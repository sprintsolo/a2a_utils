import logging
import httpx # For sending HTTP requests
from typing import Dict, Optional, Any

from a2a.server.tasks import PushNotifier
from a2a.types import PushNotificationConfig, Task, TaskState

# Assuming PushNotificationSenderAuth is not directly part of the new a2a SDK,
# we might need to replicate its core sending logic or assume a similar utility exists.
# For this example, let's define a simple auth mechanism (e.g., a bearer token).
# If PushNotificationSenderAuth from the old utils is to be reused, 
# it would need to be made available here.

logger = logging.getLogger(__name__)

class CustomPushNotifier(PushNotifier):
    """
    Custom PushNotifier implementation.
    Manages push notification configurations and sends notifications.
    """

    def __init__(self, default_auth_token: Optional[str] = None):
        """
        Initializes the CustomPushNotifier.
        Args:
            default_auth_token: An optional default bearer token for push notifications.
                                Specific tokens can still be part of PushNotificationConfig.
        """
        self._task_push_configs: Dict[str, PushNotificationConfig] = {}
        self._default_auth_token = default_auth_token
        # For more complex auth, PushNotificationSenderAuth from a2a.utils could be adapted
        # or a more sophisticated auth management system used.
        logger.info("CustomPushNotifier initialized.")

    async def set_info(self, task_id: str, config: PushNotificationConfig) -> None:
        """Stores the push notification configuration for a given task ID."""
        if not task_id or not config or not config.url:
            logger.warning(f"Invalid push notification info for task {task_id}: Missing task_id, config, or URL. Config: {config}")
            return
        
        # Basic URL validation (can be more comprehensive)
        if not config.url.startswith(("http://", "https://")):
            logger.warning(f"Invalid push notification URL for task {task_id}: {config.url}. Must be http or https.")
            # Optionally, raise an error or prevent saving
            return

        self._task_push_configs[task_id] = config
        logger.info(f"Push notification info set for task {task_id} to URL: {config.url}")

    async def get_info(self, task_id: str) -> PushNotificationConfig | None:
        """Retrieves the push notification configuration for a task ID."""
        config = self._task_push_configs.get(task_id)
        if config:
            logger.debug(f"Retrieved push notification info for task {task_id}")
        else:
            logger.debug(f"No push notification info found for task {task_id}")
        return config

    async def clear_info(self, task_id: str) -> None:
        """Clears the push notification configuration for a task ID."""
        if task_id in self._task_push_configs:
            del self._task_push_configs[task_id]
            logger.info(f"Cleared push notification info for task {task_id}")
        else:
            logger.debug(f"No push notification info to clear for task {task_id}")

    async def notify(self, task: Task) -> None:
        """
        Sends a push notification if the task state is COMPLETED, FAILED, or INPUT_REQUIRED
        and a configuration exists for the task.
        """
        if task.status.state not in [TaskState.completed, TaskState.failed, TaskState.input_required]:
            logger.debug(f"Task {task.id} is not in a notifiable state ({task.status.state}). Skipping push notification.")
            return

        config = await self.get_info(task.id)
        if not config:
            logger.debug(f"No push notification config found for task {task.id}. Skipping notification.")
            return

        if not config.url:
            logger.warning(f"Push notification URL is missing for task {task.id} in stored config. Skipping.")
            return

        payload: Dict[str, Any] = {
            "id": task.id,
            "status": task.status.model_dump(mode='json', exclude_none=True) # Send the full status object
        }
        # If contextId is relevant for the receiver, include it
        if task.contextId:
            payload["contextId"] = task.contextId

        headers = {"Content-Type": "application/json"}
        # Prefer token from config if available, else use default token
        auth_token = config.authentication.get("bearer_token") if config.authentication and isinstance(config.authentication, dict) else None
        if not auth_token:
            auth_token = self._default_auth_token
        
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        logger.info(f"Sending push notification for task {task.id} to {config.url} with payload: {payload}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(config.url, json=payload, headers=headers, timeout=10.0)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                logger.info(f"Push notification sent successfully for task {task.id}. Response: {response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Error sending push notification for task {task.id} to {config.url}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred while sending push notification for task {task.id}: {e}", exc_info=True)
        finally:
            # Optionally, clear the config after attempting to send, or based on response
            # await self.clear_info(task.id) # Example: clear after attempt
            pass 