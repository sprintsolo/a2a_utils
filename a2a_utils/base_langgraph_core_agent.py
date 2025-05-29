import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterable
import contextlib

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# LangGraph specific imports, assuming create_react_agent and MemorySaver are standard
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# LLM imports
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from .tool_usage_tracking_mixin import ToolUsageTrackingMixin

logger = logging.getLogger(__name__)

class BaseLangGraphCoreAgent(ToolUsageTrackingMixin, ABC):
    """
    Core logic for an agent using LangGraph. This class is NOT an A2A AgentExecutor.
    It handles LangGraph execution, tool usage tracking (via mixin),
    and provides events to an adapter (AgentExecutor).
    This class is responsible for all LLM-related logic including system instructions,
    input formatting, and output parsing.
    """

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"] # Can be overridden by subclasses
    DEFAULT_LLM_PROVIDER = "google"
    DEFAULT_OPENAI_MODEL = "gpt-4.1-mini" # OpenAI 기본 모델
    DEFAULT_GOOGLE_MODEL = "gemini-2.5-flash-preview-04-17" # Google 기본 모델

    def __init__(
        self,
        llm_provider: str = DEFAULT_LLM_PROVIDER,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        *args,
        **kwargs
    ):
        """
        Initializes the BaseLangGraphCoreAgent.
        Sets up the language model based on the provider.
        """
        super().__init__(*args, **kwargs)  # Initialize ToolUsageTrackingMixin
        
        self.llm_provider = llm_provider.lower()
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        self.model: Optional[Any] = None # 요청별로 생성될 수 있으므로 초기에는 None 또는 기본값
        self.configured_model_name: Optional[str] = model_name # 제공된 모델 이름 저장
        
        # __init__에서 모델을 즉시 생성하지 않도록 변경 (run_graph_stream에서 생성)
        # 다만, llm_provider에 따른 기본 모델명은 여기서 결정해둘 수 있음
        if not self.configured_model_name:
            if self.llm_provider == "openai":
                self.configured_model_name = self.DEFAULT_OPENAI_MODEL
            elif self.llm_provider == "google":
                self.configured_model_name = self.DEFAULT_GOOGLE_MODEL

        # API 키 유효성 검사는 여기서 수행 가능
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("openai_api_key must be provided if llm_provider is 'openai'.")
        if self.llm_provider == "google" and not self.google_api_key:
            raise ValueError("google_api_key must be provided if llm_provider is 'google'.")

        logger.info(f"BaseLangGraphCoreAgent configured for llm_provider: {self.llm_provider}, model_name: {self.configured_model_name}")
        self.tools: List[Any] = []
        self.graph: Optional[Any] = None
        self.memory = MemorySaver()

    @abstractmethod
    def _get_system_instruction(self) -> str:
        """Returns the base system instruction for the agent."""
        pass

    async def _format_input_for_graph(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Formats the user query into the input format for LangGraph.
        This includes adding system instructions and creating the message history.
        """
        system_instruction = self._get_system_instruction()
        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=query))
        return {"messages": messages}

    @abstractmethod
    async def parse_agent_final_output(self, agent_output: AIMessage) -> Dict[str, Any]:
        """
        Parses the final AIMessage from LangGraph into a structured dictionary.
        This dictionary will be used by the AgentExecutor to create the final A2A Message.
        Example: {"status": "completed", "message_content": "Agent's final response."}
        """
        pass

    async def _initialize_graph_if_needed(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the LangGraph (self.graph) if it's not already set.
        This method is called by the run_graph_stream template method after the LLM model is set up.
        Subclasses like BaseComposioCoreAgent will override this to fetch tools and build the graph.
        By default, it assumes self.model is set and self.tools might be populated by a subclass
        or this method itself if overridden (e.g., by BaseComposioCoreAgent).
        """
        if self.graph is not None:
            logger.debug(f"Graph already initialized for {self.__class__.__name__}. Skipping re-initialization within this session.")
            return

        if not self.model:
            raise ValueError("Language model (self.model) instance is not available for graph initialization. It should be set by _managed_llm_session before calling _initialize_graph_if_needed.")
        
        # Default implementation creates a simple ReAct agent.
        # Subclasses (like BaseComposioCoreAgent) are expected to override this method
        # to set self.tools (e.g., by fetching from Composio) and then build their specific graph.
        # If self.tools is not populated by a subclass overriding this method, a warning will be issued.
        if not self.tools and not isinstance(__import__("a2a_utils.base_composio_core_agent").base_composio_core_agent.BaseComposioCoreAgent): # Avoid circular import for type check
            logger.warning(f"No tools found in self.tools for {self.__class__.__name__}. \n                              If this agent uses tools, ensure self.tools is populated in an overridden \n                              _initialize_graph_if_needed or before graph compilation. Initializing graph without tools.")

        self.graph = create_react_agent(self.model, tools=self.tools, checkpointer=self.memory)
        tool_names = [tool.name for tool in self.tools] if self.tools else "None"
        logger.info(f"Default LangGraph graph compiled for {self.__class__.__name__} with tools: {tool_names}")

    async def _process_langgraph_event_for_tracking(self, event: Dict[str, Any]) -> None:
        """
        Processes a LangGraph event specifically for tool usage tracking.
        Updates self.used_tools via track_tool_usage from the mixin.
        """
        kind = event.get("event")
        if kind == "on_tool_end":
            tool_name = event.get("name", "unnamed_tool")
            tool_input = event.get("data", {}).get("input", "")
            tool_output = event.get("data", {}).get("output", "")
            await self.track_tool_usage(tool_name, tool_input, tool_output)
            logger.debug(f"CoreAgent tracked tool usage for {tool_name} via LangGraph event.")

    @contextlib.asynccontextmanager
    async def _managed_llm_session(self) -> AsyncIterable[None]:
        original_model = self.model
        original_graph = self.graph
        request_specific_model: Optional[Any] = None

        try:
            if self.llm_provider == "google":
                if not self.google_api_key: raise ValueError("Google API key not configured.")
                logger.info(f"Creating new ChatGoogleGenerativeAI instance for session (model: {self.configured_model_name})")
                request_specific_model = ChatGoogleGenerativeAI(
                    model=self.configured_model_name,
                    google_api_key=self.google_api_key,
                    disable_streaming=True,
                    convert_system_message_to_human=False,
                    temperature=0.1,
                )
            elif self.llm_provider == "openai":
                if not self.openai_api_key: raise ValueError("OpenAI API key not configured.")
                logger.info(f"Creating new ChatOpenAI instance for session (model: {self.configured_model_name})")
                request_specific_model = ChatOpenAI(
                    model_name=self.configured_model_name,
                    openai_api_key=self.openai_api_key,
                    streaming=True
                )
            else:
                raise ValueError(f"Unsupported llm_provider: {self.llm_provider}")
            
            self.model = request_specific_model
            self.graph = None # Force graph re-initialization with the new model
            yield
        finally:
            if request_specific_model:
                logger.info(f"Cleaning up request-specific model instance for {self.llm_provider}.")
                if hasattr(request_specific_model, "aclose") and callable(request_specific_model.aclose):
                    try:
                        await request_specific_model.aclose()
                        logger.info("Called aclose() on request-specific model.")
                    except Exception as close_exc:
                        logger.error(f"Error calling aclose() on request-specific model: {close_exc}", exc_info=True)
                elif self.llm_provider == "google" and hasattr(request_specific_model, "_async_client") and \
                       request_specific_model._async_client is not None and \
                       hasattr(request_specific_model._async_client, "close") and \
                       callable(request_specific_model._async_client.close):
                    try:
                        await request_specific_model._async_client.close()
                        logger.info("Closed _async_client of request-specific Google model.")
                    except Exception as close_exc:
                        logger.error(f"Error closing _async_client of Google model: {close_exc}", exc_info=True)
            self.model = original_model
            self.graph = original_graph

    @abstractmethod
    async def _prepare_graph_input(self, query: str, session_id: str, metadata: Optional[Dict[str, Any]]) -> Any:
        """
        Prepares the initial input/state for the LangGraph.
        Called by run_graph_stream within _managed_llm_session, after _initialize_graph_if_needed.
        Subclasses must implement this to return the graph's expected input format (e.g., ReActState, dict).
        """
        pass

    @abstractmethod
    async def _execute_prepared_graph(self, graph_input: Any, config: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> AsyncIterable[Dict[str, Any]]:
        """
        Executes the LangGraph with the prepared input and yields events.
        Called by run_graph_stream. Subclasses implement this to call self.graph.astream or astream_events.
        """
        pass

    async def run_graph_stream(
        self,
        query: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncIterable[Dict[str, Any]]:
        """
        Template method for running the LangGraph agent.
        Handles LLM client and graph lifecycle, delegates input preparation and graph execution to abstract methods.
        """
        logger.info(f"CoreAgent run_graph_stream (template) for session {session_id} - Query: {query[:50]}...")
        graph_initialized_successfully = False
        async with self._managed_llm_session():
            # 1. Initialize graph (tools, compilation) - self.model is now set by _managed_llm_session
            # metadata is passed to _initialize_graph_if_needed for potential use (e.g., external_user_id for Composio).
            await self._initialize_graph_if_needed(metadata=metadata) # Ensures self.tools and self.graph are set up.
            if self.graph is None:
                logger.error(f"Graph could not be initialized for session {session_id} by _initialize_graph_if_needed. Stream will be empty.")
                # graph_initialized_successfully remains False, loop below won't run
            else:
                graph_initialized_successfully = True

            # Only proceed if graph was initialized
            if graph_initialized_successfully:
                # 2. Prepare graph input using subclass logic
                graph_input_data = await self._prepare_graph_input(query, session_id, metadata)
                # 3. Configure LangGraph execution
                config = {"configurable": {"thread_id": str(session_id)}}
                # 4. Execute graph and stream events using subclass logic
                logger.info(f"CoreAgent proceeding to _execute_prepared_graph for session {session_id}")
                async for event in self._execute_prepared_graph(graph_input_data, config, metadata):
                    yield event
            # If graph_initialized_successfully is False, this block is skipped, yielding an empty stream.
        
        logger.info(f"CoreAgent run_graph_stream (template) finished for session {session_id}. Graph init success: {graph_initialized_successfully}") 