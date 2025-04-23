import io
import os
import uuid
import yaml
import logging
import time
import json
import asyncio

from pydub import AudioSegment
from fastapi import FastAPI, Request, HTTPException, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from backend import (
    OpenAIProxyBackend,
    LlmClientBackend,
    VisionModelBackend,
    ASRClientBackend,
    TtsClientBackend,
    ChatCompletionRequest,
    CompletionRequest,
    Message,
)

from services.model_list import GetModelList

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("api")

app = FastAPI(title="OpenAI Compatible API Server")

class Config:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "config.yaml")
        with open(config_path) as f:
            self.data = yaml.safe_load(f)
        
        tiktoken_cache_dir = os.path.join(current_dir, "cache")
        os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

config = Config()

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/v1"):
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        # if api_key != os.getenv("API_KEY"):
        #     return JSONResponse(
        #         status_code=401,
        #         content={"error": "Invalid authentication credentials"}
        #     )
    return await call_next(request)

class ModelDispatcher:
    def __init__(self):
        self.backends = {}
        self.llm_models = set()
        self.asr_models = set()
        self.tts_models = set()
        self.lock = asyncio.Lock()

    async def get_backend(self, model_name):
        async with self.lock:
            if model_name not in self.backends:
                model_config = config.data["models"].get(model_name)
                if model_config is None:
                    return None
                if model_config["type"] == "openai_proxy":
                    self.backends[model_name] = OpenAIProxyBackend(model_config)
                elif model_config["type"] in ("llm", "vlm"):
                    if model_name not in self.llm_models:
                        for old_model_name in list(self.llm_models):
                            old_instance = self.backends.pop(old_model_name, None)
                            if old_instance:
                                await old_instance.close()
                        self.llm_models.clear()
                    self.backends[model_name] = LlmClientBackend(model_config)
                    self.llm_models.add(model_name)
                elif model_config["type"] == "vision_model":
                    self.backends[model_name] = VisionModelBackend(model_config)
                elif model_config["type"] == "tts":
                    if model_name not in self.tts_models:
                        for old_model_name in list(self.tts_models):
                            old_instance = self.backends.pop(old_model_name, None)
                            if old_instance:
                                await old_instance.close()
                        self.asr_models.clear()
                    self.backends[model_name] = TtsClientBackend(model_config)
                    self.tts_models.add(model_name)
                elif model_config["type"] == "asr":
                    if model_name not in self.asr_models:
                        for old_model_name in list(self.asr_models):
                            old_instance = self.backends.pop(old_model_name, None)
                            if old_instance:
                                await old_instance.close()
                        self.asr_models.clear()
                    self.backends[model_name] = ASRClientBackend(model_config)
                    self.asr_models.add(model_name)
                else:
                    return None
            return self.backends.get(model_name)

async def initialize():
    global config
    model_list = GetModelList(
        host=config.data["server"]["host"],
        port=config.data["server"]["port"]
    )
    await model_list.get_model_list(required_mem=0)
    config = Config() 
    dispatcher = ModelDispatcher()
    return dispatcher

_dispatcher = asyncio.run(initialize()) 

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    backend = await _dispatcher.get_backend(body.model)
    if not backend:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {body.model}"
        )
    
    try:        
        if body.stream:
            chunk_generator = await backend.generate(body)
            if not chunk_generator:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate stream response"
                )
            
            async def format_stream():
                try:
                    async for chunk in chunk_generator:
                        if isinstance(chunk, dict):
                            chunk_dict = chunk
                        else:
                            chunk_dict = chunk.model_dump()
                            
                        json_chunk = json.dumps(chunk_dict, ensure_ascii=False)
                        yield f"data: {json_chunk}\n\n"
                except asyncio.CancelledError:
                    logger.warning("Client disconnected early, terminating inference...")
                    if backend and isinstance(backend, LlmClientBackend):
                        current_task = asyncio.current_task()
                        if current_task in backend._active_tasks:
                            current_task.cancel()
                    raise
                finally:
                    logger.debug("Stream connection closed")

            return StreamingResponse(
                format_stream(),
                media_type="text/event-stream"
            )
        else:
            response = await backend.generate(body)
            return JSONResponse(content=response)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def create_completion(request: Request, body: CompletionRequest):
    chat_request = ChatCompletionRequest(
        model=body.model,
        messages=[Message(role="user", content=body.prompt)],
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        top_p=body.top_p,
        stream=body.stream
    )
    
    backend = await _dispatcher.get_backend(chat_request.model)
    if not backend:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {chat_request.model}")

    try:
        if body.stream:
            chunk_generator = await backend.generate(chat_request)
            
            async def convert_stream():
                async for chunk in chunk_generator:
                    # Convert format and serialize to JSON string
                    completion_chunk = {
                        "id": chunk.get("id", f"cmpl-{uuid.uuid4()}"),
                        "object": "text_completion.chunk",
                        "created": chunk.get("created", int(time.time())),
                        "model": chat_request.model,
                        "choices": [{
                            "text": chunk["choices"][0]["delta"].get("content", ""),
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": chunk["choices"][0].get("finish_reason")
                        }]
                    }
                    yield f"data: {json.dumps(completion_chunk)}\n\n"
                
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                convert_stream(),
                media_type="text/event-stream"
            )
        else:
            chat_response = await backend.generate(chat_request)
            return JSONResponse({
                "id": f"cmpl-{uuid.uuid4()}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": chat_request.model,
                "choices": [{
                    "text": chat_response["choices"][0]["message"]["content"],
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": chat_response.get("usage", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                })
            })
            
    except Exception as e:
        logger.error(f"Completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/speech")
async def create_speech(
    request: Request,
):
    try:
        request_data = await request.json()
        model = request_data.get("model")
        voice = request_data.get("voice", "alloy")
        response_format = request_data.get("response_format", "mp3")

        if not model:
            raise HTTPException(
                status_code=400,
                detail="Model is required for speech generation"
            )

        backend = await _dispatcher.get_backend(model)
        if not backend:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {model}"
            )

        input_text = request_data.get("input")
        if not input_text:
            raise HTTPException(
                status_code=400,
                detail="Input text is required for speech generation"
            )

        audio_stream = backend.generate_speech(
            input_text=input_text,
            voice=voice,
            format=response_format
        )

        return StreamingResponse(
            audio_stream,
            media_type=f"audio/{response_format}",
            headers={"Content-Disposition": f'attachment; filename="speech.{response_format}"'}
        )
    except Exception as e:
        logger.error(f"Speech generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str = Form(None),
    prompt: str = Form(""),
    response_format: str = Form("json")
):
    backend = await _dispatcher.get_backend(model)
    if not backend:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model}"
        )

    try:
        audio_data = await file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format=file.filename.split('.')[-1])
        
        target_sample_rate = 16000
        target_channels = 1
        target_sample_width = 2
        
        if audio.frame_rate != target_sample_rate or audio.channels != target_channels or audio.sample_width != target_sample_width:
            audio = audio.set_frame_rate(target_sample_rate).set_channels(target_channels).set_sample_width(target_sample_width)
        
        segment_duration_ms = 30 * 1000
        segments = [audio[i:i + segment_duration_ms] for i in range(0, len(audio), segment_duration_ms)]
        
        transcription_results = []
        for segment in segments:
            segment_data = io.BytesIO()
            segment.export(segment_data, format="wav")
            segment_data.seek(0)
            
            transcription = await backend.create_transcription(
                segment_data.read(),
                language=language,
                prompt=prompt
            )
            transcription_results.append(transcription)
        
        full_transcription = " ".join(transcription_results)
        
        return JSONResponse(content={
            "text": full_transcription,
            "task": "transcribe",
            "language": language,
            "duration": len(audio) / 1000.0,
            "segments": len(segments),
            "sample_rate": target_sample_rate,
            "channels": target_channels,
            "bit_depth": target_sample_width * 8
        })
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form(...), 
    prompt: str = Form(""),
    response_format: str = Form("json")
):
    try:
        backend = await _dispatcher.get_backend(model)
        if not backend:
            raise HTTPException(status_code=400, detail="Unsupported model")

        audio_data = await file.read()
        
        translation = await backend.create_translation(
            audio_data,
            prompt=prompt
        )

        return JSONResponse(content={
            "text": translation,
            "task": "translate",
            "duration": 0
        })
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    models_info = []
    for model_name in config.data["models"].keys():
        model_config = config.data["models"].get(model_name, {})
        models_info.append({
            "id": model_name,
            "object": "model",
            "created": model_config.get("created", 0),
            "owned_by": model_config.get("owner", "user"),
            "permission": [],
            "root": model_config.get("root", "")
        })
    
    return {
        "data": models_info,
        "object": "list"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

logging.getLogger().handlers[0].flush()