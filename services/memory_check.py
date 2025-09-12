import logging
import asyncio
from typing import Optional
from client.sys_client import SYSClient

class MemoryChecker:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.logger = logging.getLogger("memory_check")
        self._sys_client: Optional[SYSClient] = None
        
    async def check_memory(self, required_mem: int) -> None:
        try:
            if not self._sys_client:
                self._sys_client = SYSClient(host=self.host, port=self.port)
                
            cmm_info = await self._get_cmminfo()
            remain_mem = cmm_info["data"]["remain"]
            
            self.logger.debug(f"Memory check - Required: {required_mem}, Available: {remain_mem}")
            
            if remain_mem < required_mem:
                raise RuntimeError(
                    f"Insufficient memory: {remain_mem} < {required_mem}"
                )
                
        except Exception as e:
            self.logger.error(f"Memory check failed: {str(e)}")
            raise

    async def _get_cmminfo(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._sys_client.axclcmminfo
        ) 