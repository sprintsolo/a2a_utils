import logging
import uuid
from pprint import pformat
from a2a.types import TaskStatusUpdateEvent, TaskState, Message, TextPart, TaskStatus, Artifact # Import A2A types and Artifact
from a2a.server.agent_execution import RequestContext # For type hinting in _initialize_graph
from .base_langgraph_agent_executor import BaseLangGraphAgentExecutor
from .base_composio_core_agent import BaseComposioCoreAgent # Import the new core agent
from a2a.server.events import EventQueue

logger = logging.getLogger(__name__)

class BaseComposioAgentExecutor(BaseLangGraphAgentExecutor):
    """
    A2A AgentExecutor that wraps a BaseComposioCoreAgent.
    This class specializes BaseLangGraphAgentExecutor for Composio-powered agents.
    Most of the LangGraph event handling and A2A event adaptation is done by the parent class.
    This class is primarily for type specialization and potentially minor Composio-specific
    overrides in the executor layer if needed in the future, though most Composio logic
    should now reside in BaseComposioCoreAgent.
    """

    def __init__(self, core_agent: BaseComposioCoreAgent):
        """
        Initializes the BaseComposioAgentExecutor.

        Args:
            core_agent: An instance of BaseComposioCoreAgent.
        """
        if not isinstance(core_agent, BaseComposioCoreAgent):
            raise TypeError(
                f"core_agent must be an instance of BaseComposioCoreAgent, got {type(core_agent)}"
            )
        super().__init__(core_agent) # Pass the BaseComposioCoreAgent to the parent constructor
        logger.info(f"BaseComposioAgentExecutor initialized with core agent: {core_agent.__class__.__name__}")

    # async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
    #     """
    #     Executes the LangGraph agent based on the request context and publishes events to the event queue.
    #     This method handles the execution flow and event processing only.
    #     The actual LLM interaction is delegated to the CoreAgent.
    #     """
    #     logger.info("Executing BaseComposioAgentExecutor with context:")
    #     logger.info(f"Context Details:\n{pformat(vars(context), indent=2, width=120)}")
    #     logger.info(f"Executing BaseComposioAgentExecutor for task {context}")
    #     try:
    #         # Initialize graph through core_agent
    #         await self.core_agent._initialize_graph_if_needed(
    #             metadata=getattr(context._params, 'metadata', {})
    #         )
    #     except Exception as e:
    #         logger.error(f"Error during graph initialization for task {context.task_id}: {e}", exc_info=True)
    #         error_message = Message(
    #             messageId=uuid.uuid4().hex,
    #             role="agent", 
    #             parts=[TextPart(text=f"Graph initialization failed: {str(e)}")]
    #         )
    #         event_queue.enqueue_event(error_message)
    #         event_queue.enqueue_event(
    #             TaskStatusUpdateEvent(
    #                 taskId=getattr(context, 'task_id', str(uuid.uuid4())),
    #                 contextId=getattr(context, 'context_id', str(uuid.uuid4())),
    #                 status=TaskStatus(state=TaskState.failed, message=error_message),
    #                 final=True
    #             )
    #         )
    #         return

    #     await super().execute(context, event_queue) 