import json
import socket
import time
import uuid
from typing import Generator
import logging
import threading
import base64

logger = logging.getLogger("sys_client")
logger.setLevel(logging.DEBUG)

class SYSClient:
    def __init__(self, host: str = "localhost", port: int = 10001):
        self._lock = threading.Lock()
        self.host = host
        self.port = port
        self.sock = None
        self.work_id = None
        self._initialized = False
        self._connect()
        
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    def _connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.host, self.port))
        except ConnectionRefusedError as e:
            raise RuntimeError(f"Failed to connect to {self.host}:{self.port}") from e

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def _send_request(self, action: str, object: str, data: dict) -> str:
        request_id = str(uuid.uuid4())
        object_type = "sys"
        payload = {
            "request_id": request_id,
            "work_id": self.work_id or object_type,
            "action": action,
            "object": object,
            "data": data
        }
        
        logger.debug(
            f"Sending request: [ID:{request_id}] "
            f"Action:{action} WorkID:{payload['work_id']}\n"
            f"Data: {str(data)[:100]}..."
        )
        
        self.sock.sendall(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
        return request_id

    def setup(self, object: str, model_config: dict) -> dict:
        if not self.sock:
            self._connect()
        request_id = self._send_request("setup", object, model_config)
        return self._wait_response(request_id)

    def inference_stream(self, query: str, object_type: str = "asr.base64") -> Generator[str, None, None]:
        request_id = self._send_request("inference", object_type, query)
        
        while True:
            response = json.loads(self.sock.recv(4096).decode())
            if response["request_id"] != request_id:
                continue
                
            yield response["data"]
            break

    def stop_inference(self) -> dict:
        request_id = self._send_request("pause", "llm.utf-8", {})
        return request_id

    def exit(self) -> dict:
        request_id = self._send_request("exit", "llm.utf-8", {})
        result = self._wait_response(request_id)
        self._initialized = False
        return result

    def cmminfo(self) -> dict:
        request_id = self._send_request("cmminfo", "", {})
        return self._wait_response(request_id)
    
    def axclcmminfo(self) -> dict:
        request_id = self._send_request("axcl_cmminfo", "", {})
        return self._wait_response(request_id)

    def hwinfo(self) -> dict:
        request_id = self._send_request("hwinfo", "", {})
        return self._wait_response(request_id)
    
    def model_list(self) -> dict:
        request_id = self._send_request("lsmode", "", {})
        return self._wait_response(request_id)
    
    def _wait_response(self, request_id: str) -> dict:
        start_time = time.time()
        buffer = b""
        while time.time() - start_time < 10:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            buffer += chunk
            try:
                response = json.loads(buffer.decode('utf-8'))
                if response["request_id"] == request_id:
                    if response["error"]["code"] != 0:
                        raise RuntimeError(f"Server error: {response['error']['message']}")
                    self.work_id = response["work_id"]
                    return response
            except json.JSONDecodeError:
                continue
        raise TimeoutError("No valid response from server")

    def connect(self):
        with self._lock:
            if not self.sock:
                self._connect()

    def create_transcription(self, audio_data: bytes, language: str = "zh") -> str:
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        self.setup("whisper.setup", {
            "model": "whisper-tiny",
            "response_format": "asr.utf-8",
            "input": "whisper.base64",
            "language": language,
            "enoutput": True,
        })
        
        full_text = ""
        for chunk in self.inference_stream(audio_b64, object_type="asr.base64"):
            full_text += chunk
            
        return full_text