import logging
import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterable

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# LangGraph specific imports, assuming create_react_agent and MemorySaver are standard
from langgraph.prebuilt import create_react_agent 
from langgraph.checkpoint.memory import MemorySaver

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

    def __init__(self, *args, **kwargs):
        """
        Initializes the BaseLangGraphCoreAgent.
        Subclasses are responsible for initializing self.model, and potentially self.tools and self.graph
        if they are not dynamically created per execution.
        """
        super().__init__(*args, **kwargs)  # Initialize ToolUsageTrackingMixin
        self.model: Optional[Any] = None # e.g., ChatOpenAI instance, to be set by subclass
        self.tools: List[Any] = []
        self.graph: Optional[Any] = None
        self.memory = MemorySaver() # Default memory, can be configured

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
        This method is a placeholder and should be implemented or overridden by subclasses 
        (like BaseComposioCoreAgent) if graph/tools need dynamic initialization based on metadata.
        By default, it assumes self.model and self.tools are already configured.
        """
        if self.graph is None:
            if not self.model:
                raise ValueError("Language model (self.model) must be initialized before the graph.")
            # Default graph creation without dynamic tools from metadata
            self.graph = create_react_agent(
                self.model,
                tools=self.tools, # Assumes self.tools is populated by subclass
                checkpointer=self.memory
            )
            logger.info(f"Default LangGraph graph initialized for {self.__class__.__name__}")

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

    async def run_graph_stream(
        self,
        query: str,
        session_id: str, # thread_id for LangGraph
        metadata: Optional[Dict[str, Any]] = None # For potential dynamic graph/tool initialization
    ) -> AsyncIterable[Dict[str, Any]]: # Yields raw LangGraph events
        """
        Runs the LangGraph agent and yields raw events from the graph execution stream.
        
        Args:
            query: The user query.
            session_id: The session ID, used as thread_id for LangGraph.
            metadata: Optional metadata, which might be used by _initialize_graph_if_needed.
            
        Yields:
            Raw event dictionaries from LangGraph's astream_events.
        """
        try:
            # Ensure graph is initialized (subclasses like Composio might do more here)
            await self._initialize_graph_if_needed(metadata)

            if self.graph is None:
                # This should ideally be caught by _initialize_graph_if_needed or an earlier setup phase
                raise ValueError("LangGraph (self.graph) is not initialized.")

            self.clear_used_tools() # Clear any previous tool usage for this run
            
            input_data = await self._format_input_for_graph(query)
            config = {"configurable": {"thread_id": str(session_id)}}

            logger.info(f"CoreAgent starting LangGraph stream for session {session_id} with query: {query[:100]}...")
            
            async for event in self.graph.astream_events(input_data, config, version="v1"):
                # Process the event for internal tool tracking before yielding
                await self._process_langgraph_event_for_tracking(event)
                yield event # Yield the raw LangGraph event
            
            logger.info(f"CoreAgent finished LangGraph stream for session {session_id}.")
        except Exception as e:
            logger.error(f"Error in run_graph_stream: {e}", exc_info=True)
            raise 