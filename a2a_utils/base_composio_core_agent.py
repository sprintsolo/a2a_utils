import logging
import os
import asyncio # asyncio 임포트
import functools # functools 임포트
from typing import Any, Dict, List, Optional, Callable # Callable 임포트

from composio_langchain import ComposioToolSet # App은 여기서 직접 사용 안 함
from langchain_core.tools import BaseTool # BaseTool 임포트
from langchain_core.messages import AIMessage

from .base_langgraph_core_agent import BaseLangGraphCoreAgent
from .composio_util import TemporaryComposioCwd # 컨텍스트 매니저 임포트

logger = logging.getLogger(__name__)

class BaseComposioCoreAgent(BaseLangGraphCoreAgent):
    """
    Core LangGraph agent specialized for using Composio tools.
    Inherits from BaseLangGraphCoreAgent and handles Composio tool initialization.
    """

    DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful assistant. Your primary function is to assist the user with their tasks using the available Composio tools."

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        composio_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        apps: Optional[List[str]] = None, # 특정 앱 ID 리스트 (예: ["gmail", "github"])
        llm_provider: str = BaseLangGraphCoreAgent.DEFAULT_LLM_PROVIDER,
        *args,
        **kwargs
    ):
        """
        Initializes the BaseComposioCoreAgent.

        Args:
            openai_api_key: API key for OpenAI. Passed to superclass.
            google_api_key: API key for Google. Passed to superclass.
            composio_api_key: Composio API key. If None, it attempts to use an environment variable.
            model_name: Name of the language model to use. Passed to superclass.
            apps: Optional list of Composio app IDs (strings) to enable.
                  If None or empty, default Composio behavior (all connected apps) is used.
            llm_provider: The LLM provider to use ("openai" or "google"). Passed to superclass.
        """
        super().__init__(
            llm_provider=llm_provider,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            model_name=model_name,
            *args,
            **kwargs
        )

        self.composio_toolset: Optional[ComposioToolSet] = None
        self.apps_to_enable = apps if apps else [] 
        self.composio_api_key = composio_api_key
        self.available_tool_names_and_descriptions: List[Dict[str, str]] = []

        # ComposioToolSet 초기화는 CWD 변경 없이 여기서 수행
        if ComposioToolSet:
            self.composio_toolset = ComposioToolSet(api_key=self.composio_api_key)
        else:
            logger.error("ComposioToolSet is not available. Cannot initialize Composio tools.")

    def _get_system_instruction(self) -> str:
        """Returns the base system instruction for the Composio agent."""
        return self.DEFAULT_SYSTEM_INSTRUCTION

    def _get_tool_processors(self) -> Optional[Dict[str, Any]]:
        """
        Returns processor configurations for Composio tools.
        Subclasses can override this to provide specific processors.
        Example: return {"post": {Action.GMAIL_FETCH_EMAILS: my_post_processor}}
        (Note: Action needs to be imported where this is implemented)
        """
        return None

    async def _execute_tool_fetch_in_context(
        self, 
        sync_tool_fetch_func: Callable[[], List[BaseTool]], 
        description: str
    ) -> List[BaseTool]:
        """
        Executes a synchronous tool fetching function within the TemporaryComposioCwd context
        using run_in_executor.
        """
        logger.info(f"Preparing to fetch tools: {description}")
        
        def wrapped_sync_fetch():
            # TemporaryComposioCwd는 기본값 (/tmp, .composio.lock)을 사용
            with TemporaryComposioCwd():
                logger.info(f"Executing sync_tool_fetch_func for: {description} within CWD context.")
                return sync_tool_fetch_func()

        loop = asyncio.get_running_loop()
        try:
            fetched_tools = await loop.run_in_executor(None, wrapped_sync_fetch)
            return fetched_tools if fetched_tools else []
        except Exception as e:
            logger.error(f"Error during _execute_tool_fetch_in_context for '{description}': {e}", exc_info=True)
            return []

    async def _fetch_and_prepare_tools(self, external_user_id: Optional[str] = None) -> List[BaseTool]:
        """
        Fetches and prepares tools using ComposioToolSet.
        This method is intended to be called by subclasses (e.g., in their _initialize_graph_if_needed).
        """
        tools: List[BaseTool] = []
        if not self.composio_toolset:
            logger.warning("ComposioToolSet not initialized. Cannot fetch tools.")
            return tools

        if not external_user_id and self.apps_to_enable:
            logger.warning(
                "external_user_id not provided for _fetch_and_prepare_tools. "
                "If any Composio apps require entity-specific connections, tool fetching might fail or use default entities."
            )

        tool_processors = self._get_tool_processors()
        
        sync_get_tools_call = functools.partial(
            self.composio_toolset.get_tools,
            apps=self.apps_to_enable, 
            entity_id=external_user_id, 
            processors=tool_processors
        )
        
        fetch_description = f"Composio tools for apps: {self.apps_to_enable or 'all connected'}"
        if external_user_id:
            fetch_description += f", entity_id: {external_user_id}"
        if tool_processors and tool_processors.get("post"):
             fetch_description += f", with post-processors for: {list(tool_processors.get('post', {}).keys())}"

        tools = await self._execute_tool_fetch_in_context(sync_get_tools_call, fetch_description)
        
        if tools:
            # available_tool_names_and_descriptions는 이제 호출하는 쪽(예: GmailCoreAgent)에서 직접 설정하거나,
            # 또는 이 정보가 필요한 특정 지점에서 self.tools를 기반으로 생성할 수 있습니다.
            # 여기서는 일단 로깅만 수행합니다.
            logger.info(f"Successfully fetched {len(tools)} Composio tools. Names: {[tool.name for tool in tools]}")
            # self.available_tool_names_and_descriptions 업데이트 로직은 제거하여 유연성 확보
        else:
            logger.warning("No Composio tools were fetched or an error occurred in _fetch_and_prepare_tools.")
        
        return tools

    async def _initialize_graph_if_needed(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Base implementation for graph initialization. 
        Ensures the model is ready and calls the superclass's (BaseLangGraphCoreAgent) 
        _initialize_graph_if_needed. Subclasses are expected to override this to populate 
        self.tools (e.g., by calling self._fetch_and_prepare_tools) and then call this super method, 
        or build their graph entirely.
        """
        if self.graph is not None:
            logger.debug(f"{self.__class__.__name__}: Graph already initialized. Skipping.")
            return
        
        if not self.model:
            raise ValueError(f"{self.__class__.__name__}: Language model (self.model) must be set before graph initialization.")
        
        external_user_id = metadata.get("external_user_id") if metadata else None
        
        logger.info(f"{self.__class__.__name__}: Fetching Composio tools...")
        fetched_tools = await self._fetch_and_prepare_tools(external_user_id=external_user_id)
        self.tools = fetched_tools # self.tools를 여기서 설정

        if self.tools:
            self.available_tool_names_and_descriptions = [
                {"name": tool.name, "description": tool.description} for tool in self.tools
            ]
            logger.info(f"{self.__class__.__name__}: Updated available_tool_names_and_descriptions with {len(self.tools)} tools.")
        else:
            self.available_tool_names_and_descriptions = [] # 비어있는 경우도 명시적 초기화
            logger.warning(f"{self.__class__.__name__}: No Composio tools fetched. available_tool_names_and_descriptions is empty.")

        # BaseLangGraphCoreAgent의 기본 그래프 빌드 로직을 호출하여, 여기서 가져온 self.tools를 사용하도록 함.
        # 만약 하위 클래스(예: GmailCoreAgent)가 이 기본 그래프 빌드 대신 자체 그래프를 만들고 싶다면,
        # 이 _initialize_graph_if_needed 메서드를 재정의하고, self.tools 설정 후 super() 호출 없이 직접 그래프를 빌드해야 함.
        logger.info(f"{self.__class__.__name__}: Calling super()._initialize_graph_if_needed to build graph with fetched Composio tools.")
        await super()._initialize_graph_if_needed(metadata) 
        
        # super() 호출 후, self.graph가 설정되었는지 확인 (BaseLangGraphCoreAgent의 기본 빌더가 성공했다면)
        if self.graph:
            logger.info(
                f"{self.__class__.__name__}._initialize_graph_if_needed finished. "
                f"Graph built by BaseLangGraphCoreAgent using Composio tools: {[tool.name for tool in self.tools] if self.tools else 'None'}"
            )
        else:
            # 이 경우는 BaseLangGraphCoreAgent의 기본 빌더가 어떤 이유로 그래프를 생성하지 못했거나,
            # 혹은 BaseLangGraphCoreAgent의 _initialize_graph_if_needed가 재정의되어 그래프를 빌드하지 않은 경우
            logger.warning(
                f"{self.__class__.__name__}._initialize_graph_if_needed finished, but self.graph is still None. "
                f"This might be expected if a subclass (like GmailCoreAgent) overrides _initialize_graph_if_needed "
                f"to build its own graph without calling super() or if the base graph builder failed."
            )

    async def parse_agent_final_output(self, agent_output: AIMessage) -> Dict[str, Any]:
        """
        Parses the final AIMessage from LangGraph into a structured dictionary for A2A.
        For Composio agents, the output is typically in the AIMessage.content.
        """
        content = agent_output.content
        # Further parsing can be done here if the content is structured (e.g., JSON)
        return {
            "status": "completed", 
            "message_content": content if isinstance(content, str) else str(content)
        }

    # Example of how to add more specific methods for Composio, if needed:
    # def add_composio_app_auth(self, app_id: str, auth_detail: Dict[str, Any]):
    #     """Configures authentication for a specific Composio app."""
    #     if not self.composio_toolset or not AppAuth:
    #         logger.error("ComposioToolSet or AppAuth not available. Cannot configure app authentication.")
    #         return
    #     try:
    #         auth = AppAuth(**auth_detail)
    #         self.composio_toolset.app_auth.add(app_id, auth)
    #         logger.info(f"Authentication configured for Composio app: {app_id}")
    #         # May need to re-initialize graph if tools change based on auth
    #         self.graph = None 
    #         logger.info("Graph will be re-initialized on next execution due to auth change.")
    #     except Exception as e:
    #         logger.error(f"Error configuring Composio app auth for {app_id}: {e}", exc_info=True) 