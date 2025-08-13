import asyncio
import os.path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import ModelFamily
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from loguru import logger

from openevolve.config import LLMModelConfig
from openevolve.llm import LLMInterface


class AgentLLM(LLMInterface):
    def __init__(
            self,
            model_cfg: Optional[LLMModelConfig] = None,
            system_prompt: str = None,
    ):
        self.model = model_cfg.name
        self.system_message = system_prompt
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, 'random_seed', None)
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        self.client = OpenAIChatCompletionClient(
            model=self.model,
            api_key=self.api_key,
            base_url=self.api_base,
            temperature=self.temperature,
            top_p=self.top_p,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "structured_output": True,
                "family": ModelFamily.UNKNOWN,
            }
        )
        self._load_agent_tool_resources()
        self.agent = AssistantAgent(
            "assistant",
            model_client=self.client,
            tools=self.tools,
            reflect_on_tool_use=True,
            max_tool_iterations=5,
            system_message=self.system_message,
            model_context=BufferedChatCompletionContext(buffer_size=5)
        )

    def _load_agent_tool_resources(self):
        self.validation_mcp_server = StdioServerParams(
            command="uv",
            args=[
                "run",
                "fastmcp",
                "run",
                "evaluator_mcp_server.py:mcp"
            ],
            env={
                "INITIAL_FILE_PATH": os.getenv("INITIAL_FILE_PATH"),
            },
            read_timeout_seconds=60)
        with ThreadPoolExecutor() as pool:
            self.tools = pool.submit(lambda: asyncio.run(mcp_server_tools(self.validation_mcp_server))).result()

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
            self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str | None:
        """Generate text using a system message and conversational context"""
        # Prepare messages with system message
        formatted_messages = [TextMessage(content=system_message, source="system")]
        messages = [TextMessage(content=msg.get('content'), source=msg.get('role')) for msg in messages]
        formatted_messages.extend(messages)
        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(formatted_messages), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _send_message(self, message: List[TextMessage]):
        async for message in self.agent.on_messages_stream(
                messages=message,
                cancellation_token=CancellationToken()
        ):
            logger.info(f"Received message: {message}")
            if isinstance(message, Response):
                logger.info(f"Response: {message}")
                return message
        return None

    async def _call_api(self, messages) -> str:
        """Make the actual API call"""
        response = await self._send_message(messages)
        # Logging of system prompt, user message and response content
        # logger = logging.getLogger(__name__)
        logger.info(f"API response: {response.chat_message.content}")
        return response.chat_message.content
