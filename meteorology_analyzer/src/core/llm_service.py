from typing import Optional
import logging
from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_TIMEOUT
)

logger = logging.getLogger("meteorology_analyzer")

try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:
    ChatDeepSeek = None

class LLMService:
    def __init__(self):
        self.enabled = False
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        if not DEEPSEEK_API_KEY:
            logger.warning("DeepSeek API Key not found. LLM disabled.")
            return
        
        if not ChatDeepSeek:
            logger.warning("langchain_deepseek not installed. LLM disabled.")
            return

        try:
            self.llm = ChatDeepSeek(
                model=DEEPSEEK_MODEL,
                api_key=DEEPSEEK_API_KEY,
                temperature=DEEPSEEK_TEMPERATURE,
                base_url=DEEPSEEK_BASE_URL,
                timeout=DEEPSEEK_TIMEOUT
            )
            self.enabled = True
            logger.info("LLM Service initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")

    def query(self, prompt: str) -> str:
        if not self.enabled or not self.llm:
            return "LLM service is not available."
        
        try:
            response = self.llm.invoke(prompt)
            return getattr(response, "content", str(response))
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return f"Error communicating with AI: {e}"
