import os
import logging
import asyncio
import yaml
from typing import Optional
from client.sys_client import SYSClient

class GetModelList:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.logger = logging.getLogger("get_model_list")
        self._sys_client: Optional[SYSClient] = None
        
    async def get_model_list(self, required_mem: int) -> None:
        try:
            if not self._sys_client:
                self._sys_client = SYSClient(host=self.host, port=self.port)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            config_path = os.path.join(parent_dir, "config", "config.yaml")

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config['models'] = {}
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            models_config = config.get('models', {})
            model_list = await self._get_model_list()

            for model_data in model_list["data"]:
                mode = model_data.get("mode")
                model_type = model_data.get("type")

                if not mode or not model_type:
                    continue 

                if model_type not in ['llm', 'vlm', 'tts', 'asr']:
                    continue
                
                if mode not in models_config:
                    new_entry = {
                        "host": self.host,
                        "port": self.port,
                        "type": model_type,
                        "input": f"{model_type}.utf-8",
                        "model_name": mode,
                    }

                    if model_type in ['llm', 'vlm']:
                        new_entry.update({
                            "response_format": f"{model_type}.utf-8.stream",
                            "object": f"{model_type}.setup",
                            "system_prompt": "You are a helpful assistant."
                        })

                        mode_param = model_data.get("mode_param", {})
                        precompute_len = None
                        cmm_size = None
                        if isinstance(mode_param, dict):
                            precompute_len = mode_param.get("precompute_len")
                            cmm_size = mode_param.get("cmm_size")
                        if '-1.5B-' in mode:
                            new_entry['memory_required'] = 1782579
                            new_entry['pool_size'] = 1
                        elif '-1B-' in mode:
                            new_entry['memory_required'] = 1363148
                            new_entry['pool_size'] = 2
                        elif '-0.5B-' in mode:
                            new_entry['memory_required'] = 560460
                            new_entry['pool_size'] = 2
                        else:
                            new_entry['memory_required'] = 1363148
                            new_entry['pool_size'] = 2
                        if cmm_size is not None:
                            new_entry['memory_required'] = cmm_size

                        if '-p256-' in mode:
                            new_entry['max_context_length'] = 256

                        if '-p512-' in mode:
                            new_entry['max_context_length'] = 512

                        if '-p2048-' in mode:
                            new_entry['max_context_length'] = 2048

                        if '-ctx-' in mode:
                            new_entry['max_context_length'] = 1024

                        if precompute_len is not None:
                            new_entry['max_context_length'] = precompute_len

                        if '-vl-' in mode:
                            new_entry['max_context_length'] = 1152

                    elif model_type == 'tts':
                        mode_param = model_data.get("mode_param", {})
                        precompute_len = None
                        sample_rate = None
                        cmm_size = None
                        if isinstance(mode_param, dict):
                            precompute_len = mode_param.get("precompute_len")
                            cmm_size = mode_param.get("cmm_size")
                            sample_rate = mode_param.get("sample_rate")

                        if 'melotts' in mode.lower():
                            obj = 'melotts.setup'
                            new_entry['memory_required'] = 59764
                            new_entry['sample_rate'] = 16000
                        elif 'cosyvoice' in mode.lower():
                            obj = 'cosy_voice.setup'
                            new_entry['memory_required'] = 1185772
                            new_entry['sample_rate'] = 48000
                        else:
                            obj = 'tts.setup'

                        if cmm_size is not None:
                            new_entry['memory_required'] = cmm_size
                        if precompute_len is not None:
                            new_entry['max_context_length'] = precompute_len
                        if sample_rate is not None:
                            new_entry['max_context_length'] = sample_rate

                        new_entry.update({
                            "response_format": "wav.base64",
                            "object": obj
                        })
                    elif model_type == 'asr':
                        mode_param = model_data.get("mode_param", {})
                        precompute_len = None
                        cmm_size = None
                        if isinstance(mode_param, dict):
                            precompute_len = mode_param.get("precompute_len")
                            cmm_size = mode_param.get("cmm_size")

                        if 'whisper' in mode.lower():
                            obj = 'whisper.setup'
                            if 'tiny' in mode:
                                new_entry['memory_required'] = 263860
                            elif 'base' in mode.lower():
                                new_entry['memory_required'] = 448212
                            elif 'small' in mode.lower():
                                new_entry['memory_required'] = 1132748
                            elif 'turbo' in mode.lower():
                                new_entry['memory_required'] = 2048000
                            else:
                                new_entry['memory_required'] = 1500000
                        else:
                            obj = 'asr.setup'

                        if cmm_size is not None:
                            new_entry['memory_required'] = cmm_size
                        if precompute_len is not None:
                            new_entry['max_context_length'] = precompute_len

                        new_entry.update({
                            "input": "pcm.base64",
                            "response_format": "asr.utf-8",
                            "object": obj
                        })
                    else:
                        continue

                    models_config[mode] = new_entry
                    config['models'] = models_config
                    with open(config_path, 'w') as f:
                        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        except Exception as e:
            self.logger.error(f"Get model failed: {str(e)}")
            raise

    async def _get_model_list(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._sys_client.model_list
        )