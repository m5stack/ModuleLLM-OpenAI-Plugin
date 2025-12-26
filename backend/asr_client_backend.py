import time
import asyncio
import weakref
import base64
import logging
from .base_model_backend import BaseModelBackend
from client.asr_client import ASRClient
from concurrent.futures import ThreadPoolExecutor
from services.memory_check import MemoryChecker

class ASRClientBackend(BaseModelBackend):
    def __init__(self, model_config):
        super().__init__(model_config)
        self._client_pool = []
        self._active_clients = {}
        self._pool_lock = asyncio.Lock()
        self.logger = logging.getLogger("api.asr")
        self.POOL_SIZE = 1
        self._inference_executor = ThreadPoolExecutor(max_workers=self.POOL_SIZE)
        self._active_tasks = weakref.WeakSet()
        self.memory_checker = MemoryChecker(
            host=self.config["host"],
            port=self.config["port"]
        )
        
    async def _get_client(self, language: str):
        try:
            await asyncio.wait_for(self._pool_lock.acquire(), timeout=30.0)
            
            start_time = time.time()
            timeout = 30.0
            retry_interval = 3

            while True:
                if self._client_pool:
                    client = self._client_pool.pop()
                    self.logger.debug(f"Reusing client from pool | ID:{id(client)}")
                    return client
                
                if len(self._active_clients) < self.POOL_SIZE:
                    break
                
                for task in self._active_tasks:
                    task.cancel()
                
                self._pool_lock.release()
                await asyncio.sleep(retry_interval)
                await asyncio.wait_for(self._pool_lock.acquire(), timeout=timeout - (time.time() - start_time))
                
            if "memory_required" in self.config:
                await self.memory_checker.check_memory(self.config["memory_required"])
            self.logger.debug("Creating new LLM client")
            client = ASRClient(
                host=self.config["host"],
                port=self.config["port"]
            )
            self._active_clients[id(client)] = client

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._inference_executor,
                lambda: client.setup(
                    self.config["object"],
                    {
                    "model": self.config["model_name"],
                    "response_format": "asr.utf-8",
                    "input": "whisper.asr.wav.stream.base64",
                    "language": language,
                    "enoutput": True
                    }
                )
            )
            return client
        except asyncio.TimeoutError:
            raise RuntimeError("Server busy, please try again later.")
        finally:
            if self._pool_lock.locked():
                self._pool_lock.release()

    async def _release_client(self, client):
        async with self._pool_lock:
            self._client_pool.append(client)
            self.logger.debug(f"Returned client to pool | ID:{id(client)}")
 
    async def close(self):
        for task in self._active_tasks:
            task.cancel()
        if self._active_tasks:
            await asyncio.wait(self._active_tasks, timeout=2)
        for client in self._client_pool:
            client.exit()
        self._client_pool.clear()
        self._active_clients.clear()
        self._inference_executor.shutdown(wait=False)

    async def _inference(self, client, audio_b64: str):
        loop = asyncio.get_event_loop()
        for chunk in await loop.run_in_executor(
            self._inference_executor,
            client.inference,
            audio_b64,
            "asr.base64"
        ):
            full_result = chunk
        return full_result

    async def create_transcription(self, audio_data: bytes, language: str = "zh", prompt: str = "") -> str:
        client = await self._get_client(language)
        task = asyncio.current_task()
        self._active_tasks.add(task)
        try:
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            chunk_size = 4096
            audio_chunks = [
                audio_b64[i:i + chunk_size] for i in range(0, len(audio_b64), chunk_size)
            ]
            
            transcription = ""
            for index, chunk in enumerate(audio_chunks):
                finish = index == len(audio_chunks) - 1
                
                responses = list(client.inference_stream(
                    delta=chunk,
                    index=index,
                    finish=finish,
                    object_type="asr.wav.stream.base64"
                ))
                transcription += "".join(responses)
                await asyncio.sleep(0.002)
            
            return transcription
        except asyncio.CancelledError:
            self.logger.warning("Inference task cancelled, stopping...")
            client.stop_inference()
            raise
        except Exception as e:
            self.logger.error(f"Inference error: {str(e)}")
            raise RuntimeError(f"[ERROR: {str(e)}")
        finally:
            self._active_tasks.discard(task)
            await self._release_client(client)