import logging
from abc import ABC
from typing import Any, Dict, List, Optional
import datetime
import uuid

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskStatusUpdateEvent, TaskArtifactUpdateEvent, TaskState, Message, TextPart, DataPart, Artifact, TaskStatus

# Assuming tool_usage_tracking_mixin.py is in the same directory
from .tool_usage_tracking_mixin import ToolUsageTrackingMixin

# Local core agent import
from .base_langgraph_core_agent import BaseLangGraphCoreAgent 

logger = logging.getLogger(__name__)

class BaseLangGraphAgentExecutor(AgentExecutor, ToolUsageTrackingMixin, ABC):
    """
    Base class for AgentExecutors using LangGraph.
    Inherits from a2a.server.agent_execution.AgentExecutor and ToolUsageTrackingMixin.
    This class is responsible for execution flow control and event handling only.
    All LLM-related logic is handled by the CoreAgent.
    """

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, core_agent: BaseLangGraphCoreAgent):
        """
        Initializes the BaseLangGraphAgentExecutor.

        Args:
            core_agent: An instance of BaseLangGraphCoreAgent (or its subclass)
                        that contains the actual LangGraph logic.
        """
        if not isinstance(core_agent, BaseLangGraphCoreAgent):
            raise TypeError(
                f"core_agent must be an instance of BaseLangGraphCoreAgent, got {type(core_agent)}"
            )
        super().__init__()  # Handles MRO for both AgentExecutor and ToolUsageTrackingMixin
        self.core_agent = core_agent
        self._active_tasks: Dict[str, bool] = {} # To keep track of active/cancelled tasks

    async def _handle_graph_event_for_client(
        self, event: Dict[str, Any], context: RequestContext, event_queue: EventQueue
    ) -> None:
        """
        Handles a single event from LangGraph stream and sends updates to the client via event_queue.
        Also handles tracking tool usage and sending TaskArtifactUpdateEvent.
        """
        kind = event.get("event")
        event_name = event.get("name", "unnamed_event")

        if kind == "on_tool_start":
            tool_name = event.get("name", "unnamed_tool")
            status_update = TaskStatusUpdateEvent(
                taskId=context.task_id,
                contextId=context.context_id,
                status=TaskStatus(state=TaskState.working, message=Message(
                    messageId=uuid.uuid4().hex,
                    role="agent", 
                    parts=[TextPart(text=f"Tool '{tool_name}' started.")]
                )),
                final=False
            )
            event_queue.enqueue_event(status_update)
            logger.info(f"Tool '{tool_name}' started for task {context.task_id}")

        elif kind == "on_tool_end":
            tool_name = event.get("name", "unnamed_tool")
            # The actual tool tracking (self.used_tools.append) is done by process_langgraph_event
            # Here we send the artifact for the *most recently completed* tool.
            
            # process_langgraph_event (from mixin) should have been called before this
            # by the main execute loop to update self.used_tools
            most_recent_tool_usage = self.get_used_tools()[-1] if self.get_used_tools() else None

            if most_recent_tool_usage:
                try:
                    tool_name = event.get("name", "unnamed_tool") # Ensure tool_name is available
                    tool_artifact = Artifact(
                        artifactId=f"tool_artifact_{context.task_id}_{tool_name}_{datetime.datetime.now(datetime.timezone.utc).isoformat()}",
                        name="tool_usage", # Standardized name
                        description=f"Details of {most_recent_tool_usage.get('name', 'unknown tool')} execution.",
                        parts=[DataPart(data=most_recent_tool_usage)] # Send the whole dict
                    )
                    artifact_event = TaskArtifactUpdateEvent(
                        taskId=context.task_id,
                        contextId=context.context_id,
                        artifact=tool_artifact # Corrected field name
                    )
                    event_queue.enqueue_event(artifact_event)
                    logger.info(f"Sent tool_usage artifact for {tool_name} for task {context.task_id}")
                except Exception as e:
                    logger.error(f"Error creating or sending tool_usage artifact: {e}", exc_info=True)
            
            status_update_after_tool = TaskStatusUpdateEvent(
                taskId=context.task_id,
                contextId=context.context_id,
                status=TaskStatus(state=TaskState.working, message=Message(
                    messageId=uuid.uuid4().hex,
                    role="agent", 
                    parts=[TextPart(text=f"Tool '{tool_name}' finished. Processing output...")]
                )),
                final=False
            )
            event_queue.enqueue_event(status_update_after_tool)
            logger.info(f"Tool '{tool_name}' finished for task {context.task_id}")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Executes the LangGraph agent based on the request context and publishes events to the event queue.
        This method handles the execution flow and event processing only.
        The actual LLM interaction is delegated to the CoreAgent.
        """
        if self.core_agent is None:
            logger.error(f"Agent graph is not initialized for task {context.task_id}. This must be done by a subclass or a setup method before execute.")
            # Send a failed status update
            error_message = Message(
                messageId=uuid.uuid4().hex,
                role="agent", 
                parts=[TextPart(text="Agent graph not initialized.")]
            )
            error_status = TaskStatusUpdateEvent(
                taskId=getattr(context, 'task_id', str(uuid.uuid4())),
                contextId=getattr(context, 'context_id', str(uuid.uuid4())),
                status=TaskStatus(state=TaskState.failed, message=error_message),
                final=True
            )
            event_queue.enqueue_event(error_status)
            return

        # Check if task attribute exists and handle accordingly
        task_id = getattr(context, 'task_id', str(uuid.uuid4()))
        context_id = getattr(context, 'context_id', str(uuid.uuid4()))
        task_metadata = getattr(context._params, 'metadata', {})

        query = context.get_user_input() # Prefer method from RequestContext if available, else context.message.text
        session_id = context_id or task_id  # Use context_id if available, else task_id

        logger.info(f"Executing BaseLangGraphAgentExecutor for task {task_id}, session {session_id} with query: {query[:100]}...")

        self.clear_used_tools()

        final_ai_message: Optional[AIMessage] = None
        
        try:
            async for event in self.core_agent.run_graph_stream(
                query=query, 
                session_id=session_id, 
                metadata=task_metadata
            ):
                if event.get("event") == "on_tool_end":
                    tool_name_event = event.get("name", "unnamed_tool")
                    tool_input_event = event.get("data", {}).get("input", "")
                    tool_output_event = event.get("data", {}).get("output", "")                    
                    await self.track_tool_usage(tool_name_event, tool_input_event, tool_output_event)
                    logger.debug(f"Called track_tool_usage for {tool_name_event} in execute loop.")

                # Handle event for client-facing updates (status, artifacts)
                await self._handle_graph_event_for_client(event, context, event_queue)

                if event.get("event") == "on_chat_model_end":
                    output_chunk = event.get("data", {}).get("chunk")
                    if hasattr(output_chunk, "content") and output_chunk.content:
                        final_ai_message = output_chunk 
                        logger.info(f"Captured final AIMessage (from chunk) for task {task_id}")
                    elif output_chunk:
                        final_ai_message = output_chunk
                        logger.info(f"Captured final AIMessage (from output) for task {task_id}")

            if final_ai_message:
                parsed_output = await self.core_agent.parse_agent_final_output(final_ai_message)
                logger.info(f"used_tools: {self.get_used_tools()}")
                response_message = Message(
                    messageId=uuid.uuid4().hex,
                    role="agent",
                    parts=[TextPart(text=str(parsed_output.get("message_content", "Agent processing complete.")))],
                    metadata={'used_tools': self.get_used_tools()}
                )

                logger.info(f"response_message: {response_message}")
                final_status = parsed_output.get("status", TaskState.completed)
                logger.info(f"final_status: {final_status}")
                # Send final message
                event_queue.enqueue_event(response_message)
                
                # Send final task status
                final_task_status_update = TaskStatusUpdateEvent(
                    taskId=task_id,
                    contextId=context_id,
                    status=TaskStatus(
                        state=final_status if isinstance(final_status, TaskState) else TaskState.completed,
                        message=response_message
                    ),
                    final=True
                )
                event_queue.enqueue_event(final_task_status_update)
                logger.info(f"Sent final response and status for task {task_id}")
            else:
                logger.warning(f"Stream completed without a final AIMessage for task {task_id}")
                # Send a generic completion
                generic_message = Message(
                    messageId=uuid.uuid4().hex,
                    role="agent", 
                    parts=[TextPart(text="Agent stream finished without a conclusive message.")]
                )
                event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        taskId=task_id,
                        contextId=context_id,
                        status=TaskStatus(state=TaskState.completed, message=generic_message),
                        final=True
                    )
                )

        except Exception as e:
            logger.error(f"Error during agent execution for task {task_id}: {e}", exc_info=True)
            error_message = Message(
                messageId=uuid.uuid4().hex,
                role="agent", 
                parts=[TextPart(text=f"Agent execution failed: {str(e)}")]
            )
            event_queue.enqueue_event(error_message)
            event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    taskId=task_id,
                    contextId=context_id,
                    status=TaskStatus(state=TaskState.failed, message=error_message),
                    final=True
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Requests the agent to cancel an ongoing task.
        Actual cancellation of LangGraph tasks depends on its capabilities (e.g., interrupting threads).
        For now, we'll log the request and send a CANCELED status.
        """
        logger.info(f"Cancellation requested for task {context.task_id}")
        
        cancellation_message = Message(
            messageId=uuid.uuid4().hex,
            role="agent", 
            parts=[TextPart(text=f"Task {context.task_id} cancellation requested.")]
        )
        event_queue.enqueue_event(cancellation_message)
        event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=context.task_id,
                contextId=context.context_id,
                status=TaskStatus(state=TaskState.canceled, message=cancellation_message),
                final=True
            )
        ) 