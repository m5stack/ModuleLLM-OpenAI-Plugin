import uuid
import time
import asyncio
import weakref
from concurrent.futures import ThreadPoolExecutor
from .base_model_backend import BaseModelBackend
from .chat_schemas import ChatCompletionRequest, Message, ContentItem
from client.llm_client import LLMClient
import aiohttp
import base64
import logging
from fastapi import HTTPException
from typing import Union, List
from services.memory_check import MemoryChecker
import tiktoken

class LlmClientBackend(BaseModelBackend):
    def __init__(self, model_config):
        super().__init__(model_config)
        self._client_pool = []
        self._active_clients = {}
        self._pool_lock = asyncio.Lock()
        self.logger = logging.getLogger("api.llm")
        self.MAX_CONTEXT_LENGTH = model_config.get("max_context_length", 128)
        self.POOL_SIZE = model_config.get("pool_size", 2)
        self._inference_executor = ThreadPoolExecutor(max_workers=self.POOL_SIZE)
        self._active_tasks = weakref.WeakSet()
        self.memory_checker = MemoryChecker(
            host=self.config["host"],
            port=self.config["port"]
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def _parse_content(self, content: Union[str, List[ContentItem]], base64_images: list) -> str:
        text_parts = []
        
        if isinstance(content, list):
            for item in content:
                if item.type == "text" and item.text:
                    text_parts.append(item.text.strip())
                elif item.type == "image_url" and item.image_url:
                    url = item.image_url.get("url", "")
                    if url.startswith("data:image"):
                        base64_data = url.split(",", 1)[1]
                        base64_images.append(base64_data)
                    else:
                        base64_str = await self.download_image(url)
                        if base64_str:
                            base64_images.append(base64_str)
        else:
            text_parts.append(str(content).strip())
            
        return " ".join(text_parts).strip()

    async def _get_client(self, request):
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
                # Will interrupt the activated client inference

                self._pool_lock.release()
                await asyncio.sleep(retry_interval)
                await asyncio.wait_for(self._pool_lock.acquire(), timeout=timeout - (time.time() - start_time))
                
            if "memory_required" in self.config:
                await self.memory_checker.check_memory(self.config["memory_required"])

            self.logger.debug("Creating new LLM client")
            client = LLMClient(
                host=self.config["host"],
                port=self.config["port"]
            )
            self._active_clients[id(client)] = client

            system_content = next(
                (m.content for m in request.messages if m.role == "system"),
                self.config.get("system_prompt", "You are a helpful assistant")
            )
            parsed_prompt = await self._parse_content(system_content, [])

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: client.setup(
                    self.config["object"],
                    {
                        "model": self.config["model_name"],
                        "response_format": self.config["response_format"],
                        "input": self.config["input"],
                        "enoutput": True,
                        "max_token_len": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "prompt": parsed_prompt
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

    async def inference_stream(self, query: str, base64_images: list, request: ChatCompletionRequest):
        client = await self._get_client(request)
        task = asyncio.current_task()
        self._active_tasks.add(task)
        try:
            self.logger.debug(f"Starting inference | ClientID:{id(client)} Query length:{len(query)}")
            
            loop = asyncio.get_event_loop()
            for i, image_data in enumerate(base64_images):
                client.send_jpeg(image_data, object_type="vlm.jpeg.base64")
            
            sync_gen = client.inference_stream(
                query,
                object_type="llm.utf-8"
            )

            while True:
                if task.cancelled():
                    client.stop_inference()
                    break
                    
                def get_next():
                    try:
                        return next(sync_gen)
                    except StopIteration:
                        return None
                        
                chunk = await loop.run_in_executor(
                    self._inference_executor, 
                    get_next
                )
                if chunk is None:
                    break
                yield chunk
        except asyncio.CancelledError:
            self.logger.warning("Inference task cancelled, stopping...")
            client.stop_inference()
            raise
        except Exception as e:
            self.logger.error(f"Inference error: {str(e)}")
            yield f"[ERROR: {str(e)}]"
        finally:
            self._active_tasks.discard(task)
            await self._release_client(client)
            self.logger.debug(f"Inference stopped | ClientID:{id(client)}")

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a given text."""
        return len(self.tokenizer.encode(text))

    def _truncate_history(self, messages: List[Message]) -> List[Message]:
        """Truncate history to fit model context window"""
        total_length = 0
        keep_messages = []
        
        for msg in reversed(messages):
            if msg.role == "system":
                total_length += self._count_tokens(msg.content)
                total_length += 16

        # Process in reverse to keep latest messages
        for msg in reversed(messages):
            msg_length = 0
            if isinstance(msg.content, list):
                for item in msg.content:
                    if item.type == "text":
                        msg_length += self._count_tokens(item.text)
                total_length += msg_length
                keep_messages.insert(0, msg)
                break
            else:
                msg_length = self._count_tokens(msg.content)
            if msg.role == "user":
                msg_length += 3
            if msg.role == "assistant":
                msg_length += 3
            if total_length + msg_length > self.MAX_CONTEXT_LENGTH:
                break
            total_length += msg_length
            keep_messages.insert(0, msg)  # Maintain original order
            
        return keep_messages

    async def download_image(self, url):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        return base64.b64encode(image_data).decode('utf-8')
                    self.logger.error(f"Image download failed, status code: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Image download error: {str(e)}")
            return None

    async def generate(self, request: ChatCompletionRequest):
        try:
            truncated_messages = self._truncate_history(request.messages)
            
            if not truncated_messages:
                raise HTTPException(
                status_code=400,
                detail="The input content exceeds the maximum length supported by the model."
                )

            query_lines = []
            base64_images = []
            system_prompt = ""

            for m in truncated_messages:
                if m.role == "system":
                    system_content = await self._parse_content(m.content, base64_images)
                    system_prompt += f"{system_content}\n"
                    continue
                
                message_content = await self._parse_content(m.content, base64_images)
                if message_content:
                    query_lines.append(f"{m.role}: {message_content}")

            final_query = []
            if system_prompt:
                final_query.append(system_prompt.strip())
            if base64_images:
                pass
            final_query.append("\n".join(query_lines))
            
            query = "\n\n".join(filter(None, final_query))

            self.logger.debug(
                f"Processed query | System prompt: {len(system_prompt)} chars | "
                f"Images: {len(base64_images)} | Dialogue lines: {len(query_lines)}"
            )

            if request.stream:
                async def chunk_generator():
                    try:
                        async for chunk in self.inference_stream(query, base64_images, request):
                            yield {
                                "id": f"chatcmpl-{uuid.uuid4()}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None
                                }]
                            }
                        # Add normal completion marker
                        yield {
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                    except Exception as e:
                        self.logger.error(f"Stream generation error: {str(e)}")
                        yield {
                            "error": {
                                "message": f"Stream generation failed: {str(e)}",
                                "type": "api_error"
                            }
                        }
                        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                        raise
                return chunk_generator()
            else:
                full_response = ""
                async for chunk in self.inference_stream(query, base64_images, request):
                    full_response += chunk
                return {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": full_response
                        }
                    }]
                }
        except RuntimeError as e:
            self.logger.error(f"Connection error: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Model service connection failed: {str(e)}"
            )