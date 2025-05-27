import logging
from typing import Optional

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor # For type hint
from a2a.server.tasks import TaskStore, PushNotifier # For type hints
from a2a.server.events import QueueManager # For type hint, if customizing queue_manager
from a2a.server.agent_execution import RequestContextBuilder # For type hint

# Import the custom components we've defined
# Assuming they are in the same directory or accessible via python path
from .base_langgraph_agent_executor import BaseLangGraphAgentExecutor # Specific type for our agent executor
from .custom_push_notifier import CustomPushNotifier # Our custom push notifier
from .custom_request_context_builder import CustomRequestContextBuilder # Our custom context builder

logger = logging.getLogger(__name__)

class CustomLangGraphRequestHandler(DefaultRequestHandler):
    """
    Custom RequestHandler for LangGraph-based agents.
    It uses a BaseLangGraphAgentExecutor, CustomPushNotifier, and CustomRequestContextBuilder.
    """

    def __init__(
        self,
        # agent_executor should be an instance of our BaseLangGraphAgentExecutor or its subclasses
        agent_executor: BaseLangGraphAgentExecutor,
        task_store: TaskStore,
        # Optional: if a custom queue_manager is needed, its type can be specified
        queue_manager: Optional[QueueManager] = None, 
        # push_notifier should be an instance of our CustomPushNotifier
        push_notifier: Optional[CustomPushNotifier] = None, 
        # request_context_builder should be an instance of our CustomRequestContextBuilder
        request_context_builder: Optional[CustomRequestContextBuilder] = None,
    ):
        """
        Initializes the CustomLangGraphRequestHandler.

        Args:
            agent_executor: The agent executor, expected to be a BaseLangGraphAgentExecutor.
            task_store: The task store instance.
            queue_manager: The queue manager instance. Defaults to InMemoryQueueManager if None.
            push_notifier: The custom push notifier instance.
            request_context_builder: The custom request context builder instance.
        """
        
        # Ensure the correct types are passed for our custom components if they are provided
        if push_notifier is not None and not isinstance(push_notifier, CustomPushNotifier):
            raise TypeError(f"push_notifier must be an instance of CustomPushNotifier, got {type(push_notifier)}")
        
        if request_context_builder is not None and not isinstance(request_context_builder, CustomRequestContextBuilder):
            raise TypeError(f"request_context_builder must be an instance of CustomRequestContextBuilder, got {type(request_context_builder)}")

        super().__init__(
            agent_executor=agent_executor,
            task_store=task_store,
            queue_manager=queue_manager, # DefaultRequestHandler will init InMemoryQueueManager if None
            push_notifier=push_notifier,
            request_context_builder=request_context_builder 
            # DefaultRequestHandler will init SimpleRequestContextBuilder if None, 
            # so we must provide our custom one if we want its behavior.
        )
        logger.info(
            f"CustomLangGraphRequestHandler initialized with "
            f"AgentExecutor: {type(agent_executor).__name__}, "
            f"PushNotifier: {type(push_notifier).__name__ if push_notifier else 'Default'}, "
            f"RequestContextBuilder: {type(request_context_builder).__name__ if request_context_builder else 'Default'}"
        )

    # At this point, we are mostly relying on the DefaultRequestHandler's implementation
    # for on_message_send, on_message_send_stream, on_get_task, on_cancel_task, etc.
    # If specific validation logic from the old BaseTaskManager needs to be re-introduced
    # (e.g., modality checks, specific metadata validation beyond external_user_id),
    # those methods (like on_message_send) could be overridden here.

    # Example of overriding for additional validation (if needed):
    # async def on_message_send(
    #     self,
    #     params: MessageSendParams,
    #     context: ServerCallContext | None = None,
    # ) -> Message | Task:
    #     logger.debug(f"CustomLangGraphRequestHandler received on_message_send for task_id: {params.message.taskId}")
    #     # --- Add custom validation logic here ---
    #     # Example: Check params.message.acceptedOutputModes against agent_executor.SUPPORTED_CONTENT_TYPES
    #     # if not utils.are_modalities_compatible(
    #     #     params.message.acceptedOutputModes,
    #     #     self.agent_executor.SUPPORTED_CONTENT_TYPES 
    #     # ):
    #     #     logger.warning(f"Unsupported output modes requested.")
    #     #     # This error handling would need to be more robust, returning a proper JSONRPCErrorResponse structure.
    #     #     # For simplicity, A2A SDK often handles this with AgentCard capabilities.
    #     #     # However, direct validation can be an additional layer.
    #     #     raise ServerError(error=InvalidParamsError(message="Unsupported output modes"))
    #     
    #     # --- End custom validation ---
    #     return await super().on_message_send(params, context)

    # Similarly, on_message_send_stream could be overridden if stream-specific validation is needed.

# If a CustomComposioRequestHandler is needed, it would typically inherit from this class:
# class CustomComposioRequestHandler(CustomLangGraphRequestHandler):
#     def __init__(
#         self,
#         agent_executor: BaseComposioAgentExecutor, # Note the more specific type
#         task_store: TaskStore,
#         queue_manager: Optional[QueueManager] = None,
#         push_notifier: Optional[CustomPushNotifier] = None,
#         request_context_builder: Optional[CustomRequestContextBuilder] = None,
#     ):
#         if not isinstance(agent_executor, BaseComposioAgentExecutor):
#             raise TypeError(
#                 f"agent_executor must be an instance of BaseComposioAgentExecutor for CustomComposioRequestHandler, "
#                 f"got {type(agent_executor)}"
#             )
#         super().__init__(
#             agent_executor=agent_executor,
#             task_store=task_store,
#             queue_manager=queue_manager,
#             push_notifier=push_notifier,
#             request_context_builder=request_context_builder
#         )
#         logger.info("CustomComposioRequestHandler initialized.")

    #     # Override methods here if Composio-specific request handling logic is needed,
    #     # e.g., ensuring external_user_id was set by the RequestContextBuilder if required.
    #     # async def on_message_send(...):
    #     #    # Check if context.state contains external_user_id, if not, raise error.
    #     #    return await super().on_message_send(params, context) 