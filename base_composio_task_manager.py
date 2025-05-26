import asyncio
import logging
from typing import AsyncIterable, Union, Optional, List, Dict, Any, Tuple

from a2a.types import (
    Artifact,
    InternalError,
    InvalidParamsError,
    JSONRPCResponse,
    Message,
    PushNotificationConfig,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    DataPart,
    TaskNotFoundError,
    Part,
)
from a2a.server.task_manager import InMemoryTaskManager
import a2a.server.utils as utils
from a2a.utils.push_notification_auth import PushNotificationSenderAuth

# 이제 BaseComposioAgentLogic을 임포트합니다.
from .base_composio_agent import BaseComposioAgent 
from .base_task_manager import BaseTaskManager
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  # 콘솔 출력
    ]
)
logger = logging.getLogger(__name__)

class BaseComposioTaskManager(BaseTaskManager):
    """Base TaskManager for A2A servers using Composio-enabled agents."""

    def __init__(self, agent: BaseComposioAgent, notification_sender_auth: Optional[PushNotificationSenderAuth] = None):
        if not isinstance(agent, BaseComposioAgent):
            raise TypeError(f"Agent must be an instance of BaseComposioAgentLogic, got {type(agent)}")
        super().__init__(agent, notification_sender_auth)
        logger.info(f"BaseComposioA2ATaskManager initialized with agent: {type(agent).__name__}")

    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        """Extracts the user's text query from the first TextPart."""
        for part in task_send_params.message.parts:
            if isinstance(part, TextPart):
                return part.text
        logger.warning(f"No text part found in message for task {task_send_params.id}")
        raise ValueError(f"No text input found in the user message for task {task_send_params.id}. Agent currently only supports text.")

    def _extract_metadata(self, task_send_params: TaskSendParams) -> Optional[Dict[str, Any]]:
        """Extracts metadata from task_send_params if it exists and is a dict."""
        if hasattr(task_send_params, 'metadata') and isinstance(task_send_params.metadata, dict):
            return task_send_params.metadata
        logger.debug(f"No metadata found or metadata is not a dict for task {task_send_params.id}")
        return None

    def _validate_agent_specific_requirements(self, task_send_params: TaskSendParams) -> Optional[JSONRPCResponse]:
        """
        Implement Gmail-specific validation if needed.
        For example, ensure 'external_user_id' is present if Composio tools are always user-specific.
        BaseComposioAgentLogic already handles fetching tools based on external_user_id,
        but TaskManager can enforce its presence earlier.
        """
        metadata = self._extract_metadata(task_send_params)
        # Gmail 에이전트는 Composio를 사용하므로 external_user_id가 필수적일 수 있습니다.
        # self.agent (GmailAgent -> BaseComposioAgentLogic)의 tool_set 유무로 판단 가능
        if isinstance(self.agent, BaseComposioAgent) and self.agent.tool_set:
            if not metadata or not metadata.get("external_user_id"):
                logger.warning(f"'external_user_id' missing in metadata for task {task_send_params.id}. Required for Gmail Composio tools.")
                return JSONRPCResponse(id=task_send_params.id, error=InvalidParamsError(message="'external_user_id' is missing in metadata for Gmail agent."))
        
        return super()._validate_agent_specific_requirements(task_send_params)
    
    def _validate_request(
        self, request: Union[SendTaskRequest, SendTaskStreamingRequest]
    ) -> Optional[JSONRPCResponse]:
        """Validates the incoming A2A request."""
        task_send_params: TaskSendParams = request.params
        
        if not utils.are_modalities_compatible(
            task_send_params.acceptedOutputModes,
            self.agent.SUPPORTED_CONTENT_TYPES 
        ):
            logger.warning(f"Unsupported output modes requested for task {request.params.id}. Supported: {self.agent.SUPPORTED_CONTENT_TYPES}")
            return utils.new_incompatible_types_error(request.id)

        if task_send_params.pushNotification:
            if not self.notification_sender_auth:
                 logger.warning(f"Push notification requested for task {request.params.id}, but not supported/configured.")
                 return JSONRPCResponse(id=request.id, error=InternalError(message="Push notifications not configured"))
            if not task_send_params.pushNotification.url:
                logger.warning(f"Push notification URL missing for task {request.params.id}")
                return JSONRPCResponse(id=request.id, error=InvalidParamsError(message="Push notification URL is missing"))
        
        agent_specific_error = self._validate_agent_specific_requirements(task_send_params)
        if agent_specific_error:
            return agent_specific_error

        return None

    async def send_task_notification(self, task: Task):
        # push_config should only be retrieved if push was requested and stored
        push_config: Optional[PushNotificationConfig] = None
        try:
            # First, try to get the push info. This might raise KeyError if not set.
            push_config = await self.get_push_notification_info(task.id) 
        except KeyError:
            logger.debug(f"No push notification info found for task {task.id}. Notification will not be sent.")
            return # No config found, so exit the method
        except Exception as e:
            # Handle other potential errors during retrieval
            logger.error(f"Error retrieving push notification info for task {task.id}: {e}", exc_info=True)
            return

        # If we successfully retrieved push_config and auth is configured
        if push_config and self.notification_sender_auth:
            # Check if the task state is one that should trigger a notification
            if task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.INPUT_REQUIRED]:
                try:
                    await self.notification_sender_auth.send_push_notification(
                        push_url=push_config.url, 
                        task_info={"id": task.id, "status": task.status.model_dump(exclude_none=True)}
                    )
                    logger.info(f"Push notification sent for task {task.id} to {push_config.url}")
                except Exception as e:
                    logger.error(f"Failed to send push notification for task {task.id}: {e}")
        elif not push_config:
             # This case should be covered by the try-except block, but added for clarity
             logger.debug(f"Push config was None after retrieval attempt for task {task.id}. Notification not sent.")

    
    def _process_agent_response(self, agent_response_dict: Dict[str, Any], parts: List[Part]) -> Tuple[TaskState, Optional[Message], Optional[List[Artifact]]]:
        """
        부모 클래스의 _process_agent_response를 호출하기 전에 ComposioAgent 특화 로직을 추가합니다.
        필요한 경우 여기에 Composio 특화 처리를 추가할 수 있습니다.
        """
        # ComposioAgent 특화 처리가 필요한 경우 여기에 추가

        # 부모 클래스의 메서드 호출
        return super()._process_agent_response(agent_response_dict, parts) 
    
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handles synchronous 'tasks/send' requests."""
        logger.info(f"Received tasks/send request: id={request.params.id}, session={request.params.sessionId}")
        validation_error = self._validate_request(request)
        if validation_error:
            error_payload = validation_error.error
            return SendTaskResponse(id=request.id, error=error_payload if error_payload else InternalError(message="Validation failed"))

        if request.params.pushNotification and self.notification_sender_auth:
            if not await self.set_push_notification_info(request.params.id, request.params.pushNotification):
                logger.warning(f"Failed to set push notification for task {request.params.id}")
                return SendTaskResponse(id=request.id, error=InvalidParamsError(message="Push notification URL verification failed or invalid config"))
            logger.info(f"Push notification configured for task {request.params.id}")

        try:
            query = self._get_user_query(request.params)
            metadata = self._extract_metadata(request.params) # metadata 추출
        except ValueError as e:
             logger.error(f"Could not get user query for task {request.params.id}: {e}")
             return SendTaskResponse(id=request.id, error=InvalidParamsError(message=str(e)))

        await self.upsert_task(request.params)
        task = await self.update_store(
            request.params.id, TaskStatus(state=TaskState.WORKING), artifacts=None
        )

        try:
            # 에이전트 호출 (invoke 사용)
            logger.debug(f"Invoking agent with query: {query}, session_id: {request.params.sessionId}, metadata: {metadata}")
            agent_response_dict = await self.agent.invoke(query=query, session_id=request.params.sessionId, metadata=metadata)
            logger.info(f"Agent invoke completed for task {request.params.id}. Response status: {agent_response_dict.get('status', 'unknown')}")
            print(f"Agent invoke completed for task {request.params.id}. Response: {agent_response_dict}")
            
            # used_tools 정보 로깅
            if 'used_tools' in agent_response_dict:
                logger.info(f"Found {len(agent_response_dict['used_tools'])} used tools in agent response")
            else:
                logger.warning(f"No 'used_tools' found in agent response for task {request.params.id}")

            # BaseComposioA2ATaskManager의 _process_agent_response 메서드를 사용하여 응답 처리
            # 이 메서드는 agent_response_dict에서 추출한 도구 사용 정보를 아티팩트로 변환
            
            agent_status = agent_response_dict.get("status", "error")
            agent_message_content = agent_response_dict.get("message", "Agent response format incorrect.")
            agent_raw_result = agent_response_dict.get("result") 

            parts = [TextPart(text=agent_message_content)]
            if agent_raw_result is not None:
                if isinstance(agent_raw_result, (dict, list)):
                    parts.append(DataPart(data={"agent_result": agent_raw_result}))
                elif isinstance(agent_raw_result, (str, int, float, bool)):
                     parts.append(DataPart(data={"agent_result": agent_raw_result}))

            # 에이전트 응답 처리하여 상태, 메시지, 아티팩트 추출
            final_task_state, final_message, final_artifacts = self._process_agent_response(agent_response_dict, parts)
            
            # 도구 사용 아티팩트 로깅
            if final_artifacts:
                logger.info(f"Created {len(final_artifacts)} artifacts for task {request.params.id}")
                for idx, artifact in enumerate(final_artifacts):
                    logger.info(f"Artifact {idx}: name={getattr(artifact, 'name', 'unnamed')}, parts={len(artifact.parts)}")
                    # if getattr(artifact, 'name', '') == 'tool_usage':
                        # logger.info(f"Tool usage artifact found: {artifact}")
            else:
                logger.warning(f"No artifacts created for task {request.params.id}")

            final_task_status = TaskStatus(state=final_task_state, message=final_message)
            final_task = await self.update_store(
                request.params.id, final_task_status, artifacts=final_artifacts
            )
            task_result_for_response = self.append_task_history(final_task, request.params.historyLength)
            await self.send_task_notification(final_task)
            logger.info(f"Responding to tasks/send for task {request.params.id} with state: {final_task_state}")
            return SendTaskResponse(id=request.id, result=task_result_for_response)

        except Exception as e:
            logger.error(f"Error during agent invocation or processing for task {request.params.id}: {e}", exc_info=True)
            fail_status = TaskStatus(state=TaskState.FAILED, message=Message(role="agent", parts=[TextPart(text=f"Internal server error: {e}")]))
            failed_task = await self.update_store(request.params.id, fail_status, artifacts=None)
            await self.send_task_notification(failed_task)
            return SendTaskResponse(id=request.id, error=InternalError(message=f"Agent processing failed: {e}"))

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        """Handles streaming 'tasks/sendSubscribe' requests."""
        logger.info(f"Received tasks/sendSubscribe request: id={request.params.id}, session={request.params.sessionId}")
        validation_error = self._validate_request(request)
        if validation_error:
            return validation_error

        if request.params.pushNotification and self.notification_sender_auth:
             if not await self.set_push_notification_info(request.params.id, request.params.pushNotification):
                 logger.warning(f"Failed to set push notification for task {request.params.id}")
                 return JSONRPCResponse(id=request.id, error=InvalidParamsError(message="Push notification URL verification failed or invalid config"))
             logger.info(f"Push notification configured for task {request.params.id}")

        try:
            await self.upsert_task(request.params)
            sse_event_queue = await self.setup_sse_consumer(request.params.id, is_resubscribe=False)
            logger.info(f"SSE consumer setup for task {request.params.id}")
            # metadata 추출하여 _run_streaming_agent로 전달
            metadata = self._extract_metadata(request.params)
            asyncio.create_task(self._run_streaming_agent(request, metadata))
            logger.info(f"Agent streaming task started for {request.params.id}")
            return self.dequeue_events_for_sse(request.id, request.params.id, sse_event_queue)
        except Exception as e:
            logger.error(f"Error setting up SSE stream for task {request.params.id}: {e}", exc_info=True)
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(message=f"Failed to set up streaming: {e}")
            )

    async def _run_streaming_agent(self, request: SendTaskStreamingRequest, metadata: Optional[Dict[str, Any]]):
        """Runs the GmailAgent's stream method and pushes updates to SSE queue."""
        task_id = request.params.id
        session_id = request.params.sessionId
        logger.info(f"Starting agent stream processing for task {task_id}")

        try:
            query = self._get_user_query(request.params)
            working_status = TaskStatus(state=TaskState.WORKING)
            await self.update_store(task_id, working_status, artifacts=None)
            await self.send_task_notification(await self.get_task_info(task_id))
            await self.enqueue_events_for_sse(task_id, TaskStatusUpdateEvent(id=task_id, status=working_status, final=False))

            latest_task_state: Optional[TaskState] = TaskState.WORKING
            final_agent_response: Optional[dict] = None

            # metadata를 agent.stream으로 전달
            logger.debug(f"Starting agent.stream with query: {query}, session_id: {session_id}, metadata: {metadata}")
            async for agent_event in self.agent.stream(query, session_id, metadata):
                event_type = agent_event.get("type")
                logger.debug(f"Agent stream event for task {task_id}: {event_type}")

                if event_type == "llm_token": # LLM 토큰 스트리밍
                    token = agent_event.get("token", "")
                    if token:
                        artifact_part = TextPart(text=token)
                        temp_artifact = Artifact(parts=[artifact_part], index=0)
                        await self.enqueue_events_for_sse(task_id, TaskArtifactUpdateEvent(id=task_id, artifact=temp_artifact))

                elif event_type == "status_update": # 상태 업데이트
                    status_message_text = agent_event.get("message", "Working...")
                    # agent_event["status"]를 TaskState로 변환
                    agent_event_status = agent_event.get("status")
                    task_state_update = TaskState.WORKING # 중간 업데이트는 기본적으로 WORKING
                    
                    if agent_event_status == "error":
                        task_state_update = TaskState.FAILED
                    
                    status_message = Message(role="agent", parts=[TextPart(text=status_message_text)])
                    intermediate_status = TaskStatus(state=task_state_update, message=status_message)
                    
                    await self.send_task_notification(Task(id=task_id, sessionId=session_id, status=intermediate_status))
                    await self.enqueue_events_for_sse(task_id, TaskStatusUpdateEvent(id=task_id, status=intermediate_status, final=False))
                    latest_task_state = task_state_update

                elif event_type == "final_result": # 최종 결과
                    final_agent_response = agent_event
                    logger.warning(f"Received final_result from agent for task {task_id}. Status: {final_agent_response.get('status', 'unknown')}")
                    # used_tools 정보 로깅
                    if 'used_tools' in final_agent_response:
                        logger.warning(f"Found {len(final_agent_response['used_tools'])} used tools in final_result")
                        for idx, tool in enumerate(final_agent_response['used_tools']):
                            logger.warning(f"Tool {idx}: {tool.get('name', 'unnamed')}, input: {tool.get('input', 'N/A')[:50]}")
                    else:
                        logger.warning(f"No 'used_tools' found in final_result for task {task_id}")
                        # 결과 키 확인
                        logger.warning(f"final_result keys: {list(final_agent_response.keys())}")
                        # 도구 사용 정보가 없으면 에이전트에 직접 물어봄
                        if hasattr(self.agent, 'get_used_tools'):
                            direct_tools = self.agent.get_used_tools()
                            if direct_tools:
                                logger.warning(f"Retrieved {len(direct_tools)} tools directly from agent")
                                # 결과에 도구 사용 정보 추가
                                final_agent_response['used_tools'] = direct_tools
                                logger.warning(f"Added used_tools to final_agent_response manually")
                            else:
                                logger.warning("Agent reported no tools used when asked directly")
                    break
                elif event_type == "on_tool_start" or event_type == "on_tool_end":
                    # 도구 관련 이벤트를 로그에 기록
                    logger.warning(f"Tool event: {event_type} for tool '{agent_event.get('name', 'unnamed')}'")
                    logger.debug(f"Full tool event data: {agent_event}")
                    
                    # Streaming 중에 도구 사용 이벤트 발생 시, 적절한 알림을 사용자에게 보냄
                    if event_type == "on_tool_start":
                        tool_name = agent_event.get('name', 'unknown tool')
                        status_message = Message(role="agent", parts=[TextPart(text=f"도구 '{tool_name}'을 사용하고 있습니다...")])
                        intermediate_status = TaskStatus(state=TaskState.WORKING, message=status_message)
                        await self.enqueue_events_for_sse(task_id, TaskStatusUpdateEvent(id=task_id, status=intermediate_status, final=False))
                        
                    elif event_type == "on_tool_end":
                        tool_name = agent_event.get('name', 'unknown tool')
                        status_message = Message(role="agent", parts=[TextPart(text=f"도구 '{tool_name}' 사용이 완료되었습니다.")])
                        intermediate_status = TaskStatus(state=TaskState.WORKING, message=status_message)
                        await self.enqueue_events_for_sse(task_id, TaskStatusUpdateEvent(id=task_id, status=intermediate_status, final=False))

            if final_agent_response:
                agent_status = final_agent_response.get("status", "error")
                agent_message_content = final_agent_response.get("message", "Agent response format incorrect.")
                agent_raw_result = final_agent_response.get("result")
                
                parts = [TextPart(text=agent_message_content)]
                if agent_raw_result is not None:
                    if isinstance(agent_raw_result, (dict, list)):
                        parts.append(DataPart(data={"agent_result": agent_raw_result}))
                    elif isinstance(agent_raw_result, (str, int, float, bool)):
                        parts.append(DataPart(data={"agent_result": agent_raw_result}))

                # 에이전트 응답 처리하여 상태, 메시지, 아티팩트 추출
                final_task_state, final_message, final_artifacts = self._process_agent_response(final_agent_response, parts)
                
                # 도구 사용 아티팩트 로깅
                if final_artifacts:
                    logger.info(f"Created {len(final_artifacts)} artifacts for task {task_id}")
                    for idx, artifact in enumerate(final_artifacts):
                        logger.info(f"Artifact {idx}: name={getattr(artifact, 'name', 'unnamed')}, parts={len(artifact.parts)}")
                        if getattr(artifact, 'name', '') == 'tool_usage':
                            logger.info(f"Tool usage artifact found with {len(artifact.parts)} parts")
                else:
                    logger.warning(f"No artifacts created for task {task_id}")

                final_task_status = TaskStatus(state=final_task_state, message=final_message)
                latest_task = await self.update_store(task_id, final_task_status, artifacts=final_artifacts)
                await self.send_task_notification(latest_task)

                # 모든 아티팩트를 SSE 이벤트로 전송
                if final_artifacts:
                    for artifact in final_artifacts:
                        await self.enqueue_events_for_sse(task_id, TaskArtifactUpdateEvent(id=task_id, artifact=artifact))
                
                await self.enqueue_events_for_sse(task_id, TaskStatusUpdateEvent(id=task_id, status=final_task_status, final=True))
                latest_task_state = final_task_state
            else:
                 logger.error(f"Agent stream for task {task_id} ended without a final result event.")
                 fail_status = TaskStatus(state=TaskState.FAILED, message=Message(role="agent", parts=[TextPart(text="Internal error: Agent stream ended unexpectedly.")]))
                 failed_task = await self.update_store(task_id, fail_status, artifacts=None)
                 await self.send_task_notification(failed_task)
                 await self.enqueue_events_for_sse(task_id, TaskStatusUpdateEvent(id=task_id, status=fail_status, final=True))
                 latest_task_state = TaskState.FAILED

        except Exception as e:
            logger.error(f"Error during agent stream processing for task {task_id}: {e}", exc_info=True)
            try:
                fail_status = TaskStatus(state=TaskState.FAILED, message=Message(role="agent", parts=[TextPart(text=f"Internal server error during stream: {e}")]))
                failed_task = await self.update_store(task_id, fail_status, artifacts=None)
                await self.send_task_notification(failed_task)
                await self.enqueue_events_for_sse(task_id, TaskStatusUpdateEvent(id=task_id, status=fail_status, final=True))
            except Exception as inner_e:
                logger.error(f"Further error during stream cleanup for task {task_id}: {inner_e}", exc_info=True)
        finally:
            logger.info(f"Finished agent stream processing for task {task_id} with final state {latest_task_state}")
            await self.cleanup_sse_consumer(task_id)

    async def set_push_notification_info(self, task_id: str, push_notification_config: PushNotificationConfig):
        # JWKS URL 검증 등 추가 로직이 필요하다면 여기에 구현
        # MathAgentTaskManager에서는 검증 로직이 없었으므로, 여기서는 URL만 저장.
        # 실제 프로덕션에서는 이 URL이 신뢰할 수 있는지 등을 확인해야 할 수 있음.
        return await super().set_push_notification_info(task_id, push_notification_config)
    
    async def on_resubscribe_to_task(
        self, request: TaskIdParams
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        logger.info(f"Received tasks/resubscribe request for task ID: {request.id}")
        task_id = request.id
        task_info = await self.get_task_info(task_id)

        if not task_info:
            logger.warning(f"Task not found for resubscribe: {task_id}")
            return JSONRPCResponse(id=request.id, error=TaskNotFoundError(message=f"Task with ID {task_id} not found."))

        # 재구독 시에는 이미 Task가 생성되어 있고, 해당 Task의 SSE 큐에 연결합니다.
        # _run_streaming_agent를 다시 호출하지 않습니다.
        try:
            sse_event_queue = await self.setup_sse_consumer(task_id, is_resubscribe=True)
            logger.info(f"SSE consumer re-established for task {task_id}")
            # 과거 이벤트 재전송 로직은 InMemoryTaskManager에 이미 있을 수 있음 (get_past_events)
            # 여기서는 dequeue_events_for_sse가 이를 처리한다고 가정
            return self.dequeue_events_for_sse(request.id, task_id, sse_event_queue, send_past_events=True)
        except Exception as e:
            logger.error(f"Error setting up SSE resubscription for task {task_id}: {e}", exc_info=True)
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(message=f"Failed to set up resubscription: {e}")
            ) 