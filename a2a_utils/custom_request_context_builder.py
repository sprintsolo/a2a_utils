import logging
from typing import Optional, Dict, Any

from a2a.server.agent_execution import RequestContext, SimpleRequestContextBuilder
from a2a.server.context import ServerCallContext # For type hinting
from a2a.types import MessageSendParams, Task, TaskStore # For type hinting

logger = logging.getLogger(__name__)

class CustomRequestContextBuilder(SimpleRequestContextBuilder):
    """
    Custom RequestContextBuilder that extracts 'external_user_id' from request metadata
    and adds it to the RequestContext.state.
    """

    def __init__(self, task_store: TaskStore, should_populate_referred_tasks: bool = False, external_user_id_key: str = "external_user_id"):
        """
        Initializes the CustomRequestContextBuilder.
        Args:
            task_store: The TaskStore instance.
            should_populate_referred_tasks: Whether to populate referred tasks in the context.
            external_user_id_key: The key to look for in the metadata for the external user ID.
        """
        super().__init__(task_store=task_store, should_populate_referred_tasks=should_populate_referred_tasks)
        self.external_user_id_key = external_user_id_key
        logger.info(f"CustomRequestContextBuilder initialized. External user ID key: '{self.external_user_id_key}'")

    async def build(
        self,
        params: MessageSendParams,
        task_id: Optional[str],
        context_id: Optional[str],
        task: Optional[Task],
        context: Optional[ServerCallContext],
    ) -> RequestContext:
        """
        Builds the RequestContext, adding external_user_id to its state if found in params.metadata.
        """
        # Call the parent class's build method to get the base RequestContext
        request_context = await super().build(
            params=params,
            task_id=task_id,
            context_id=context_id,
            task=task,
            context=context,
        )

        # Extract external_user_id from params.metadata
        external_user_id: Optional[str] = None
        if params.metadata and isinstance(params.metadata, dict):
            external_user_id = params.metadata.get(self.external_user_id_key)
            if external_user_id:
                if not isinstance(external_user_id, str):
                    logger.warning(f"Found '{self.external_user_id_key}' in metadata but it's not a string: {external_user_id}. Will not be set.")
                    external_user_id = None # Ensure it's not set if invalid type
                else:
                    logger.info(f"Extracted '{self.external_user_id_key}': {external_user_id} from request metadata.")
            else:
                logger.debug(f"Key '{self.external_user_id_key}' not found in request metadata.")
        else:
            logger.debug("Request metadata is missing or not a dictionary.")

        # Add external_user_id to the RequestContext.state
        # RequestContext.state is initialized as an empty dict if None by its constructor.
        if request_context.state is None: # Should not happen if RequestContext initializes it
            request_context._state = {} # type: ignore[attr-defined] # Accessing protected member for robust init
        
        if external_user_id:
            request_context.state[self.external_user_id_key] = external_user_id
            logger.debug(f"Added '{self.external_user_id_key}' to RequestContext.state.")
        
        # Example: If Composio agent requires external_user_id, you might want to raise an error here
        # if not external_user_id and <condition_for_composio_agent>:
        #     raise InvalidParamsError(message=f"'{self.external_user_id_key}' is required for this agent.")

        return request_context 