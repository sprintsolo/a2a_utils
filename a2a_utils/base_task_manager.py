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
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  # 콘솔 출력
    ]
)
logger = logging.getLogger(__name__)

class BaseTaskManager(InMemoryTaskManager):
    """
    도구 사용을 추적하고 아티팩트를 생성하는 기능을 제공하는 베이스 TaskManager.
    
    이 클래스는 일반 에이전트(Composio 도구를 사용하지 않는 에이전트)의 TaskManager가
    도구 사용 정보를 추적하고 호스트에 전달할 수 있도록 기본 기능을 제공합니다.
    """

    def __init__(self, agent, notification_sender_auth: Optional[PushNotificationSenderAuth] = None):
        """
        BaseToolTrackingTaskManager 초기화
        
        Args:
            agent: 작업을 처리할 에이전트 인스턴스 (ToolUsageTrackingMixin을 상속받아야 함)
            notification_sender_auth: 알림 전송을 위한 인증 정보 (선택 사항)
        """
        super().__init__()
        self.agent = agent
        self.notification_sender_auth = notification_sender_auth
        logger.info(f"BaseToolTrackingTaskManager initialized with agent: {type(agent).__name__}")

    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        """사용자 쿼리를 추출합니다."""
        for part in task_send_params.message.parts:
            if isinstance(part, TextPart):
                return part.text
        logger.warning(f"No text part found in message for task {task_send_params.id}")
        raise ValueError(f"No text input found in the user message for task {task_send_params.id}. Agent currently only supports text.")

    def _extract_metadata(self, task_send_params: TaskSendParams) -> Optional[Dict[str, Any]]:
        """메타데이터를 추출합니다."""
        if hasattr(task_send_params, 'metadata') and isinstance(task_send_params.metadata, dict):
            return task_send_params.metadata
        logger.debug(f"No metadata found or metadata is not a dict for task {task_send_params.id}")
        return None

    def _validate_agent_specific_requirements(self, task_send_params: TaskSendParams) -> Optional[JSONRPCResponse]:
        """에이전트별 특수 요구사항을 검증합니다. 서브클래스에서 오버라이드하세요."""
        return None

    def _validate_request(
        self, request: Union[SendTaskRequest, SendTaskStreamingRequest]
    ) -> Optional[JSONRPCResponse]:
        """요청을 검증합니다."""
        task_send_params: TaskSendParams = request.params
        
        supported_content_types = getattr(self.agent, "SUPPORTED_CONTENT_TYPES", ["text", "text/plain"])
        if not utils.are_modalities_compatible(
            task_send_params.acceptedOutputModes,
            supported_content_types
        ):
            logger.warning(f"Unsupported output modes requested for task {request.params.id}. Supported: {supported_content_types}")
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

    def _create_tool_usage_artifact(self, used_tools: List[Dict[str, Any]]) -> Optional[Artifact]:
        """도구 사용 정보를 아티팩트로 변환합니다.
        
        Args:
            used_tools: 사용된 도구 목록
            
        Returns:
            Artifact 객체 또는 None (도구 사용 기록이 없는 경우)
        """
        if not used_tools:
            logger.warning("No tools used, skipping tool usage artifact creation")
            return None
            
        try:
            # 도구 사용 정보를 JSON 형식으로 변환
            tools_json = {"used_tools": used_tools}
            logger.debug(f"Creating tool usage artifact with data: {tools_json}")
            
            # 도구 사용 정보 DataPart 생성
            data_part = DataPart(data=tools_json)
            logger.debug(f"Created DataPart for tool usage: {data_part}")
            
            # Artifact 객체 생성
            tools_artifact = Artifact(
                name="tool_usage",
                description="Tools used during processing",
                parts=[data_part],
                index=1
            )
            logger.info(f"Created tool usage artifact with {len(used_tools)} tools: {tools_artifact}")
            
            # DataPart 내용 확인 
            if hasattr(data_part, 'data'):
                logger.debug(f"DataPart.data exists and contains: {data_part.data}")
                if 'used_tools' in data_part.data:
                    logger.debug(f"DataPart.data contains 'used_tools' with {len(data_part.data['used_tools'])} items")
                else:
                    logger.warning("DataPart.data does not contain 'used_tools' key")
            else:
                logger.warning("DataPart does not have 'data' attribute")
                
            return tools_artifact
        except Exception as e:
            logger.error(f"Error creating tool usage artifact: {e}", exc_info=True)
            return None

    def _process_agent_response(self, agent_response_dict: Dict[str, Any], parts: List[Part]) -> Tuple[TaskState, Optional[Message], Optional[List[Artifact]]]:
        """에이전트 응답을 처리하여 상태, 메시지, 아티팩트를 반환합니다.
        
        Args:
            agent_response_dict: 에이전트의 응답 딕셔너리
            parts: 메시지 파트 목록
            
        Returns:
            (TaskState, Optional[Message], Optional[List[Artifact]]) 튜플
        """
        # agent_response_dict 구조 로깅
        logger.info(f"Processing agent response. Keys in agent_response_dict: {list(agent_response_dict.keys())}")
        
        agent_status = agent_response_dict.get("status", "error")
        print(f"Agent response status: {agent_status}")
        
        final_task_state = TaskState.UNKNOWN
        final_message: Optional[Message] = None
        final_artifacts: Optional[List[Artifact]] = None
        
        # 기본 아티팩트 생성 (텍스트 또는 데이터 파트)
        content_artifact = Artifact(parts=parts, index=0)
        
        # 도구 사용 정보 아티팩트 생성
        used_tools = agent_response_dict.get("used_tools", [])
        
        # used_tools 상세 로깅
        if "used_tools" in agent_response_dict:
            tools_count = len(used_tools) if used_tools else 0
            logger.info(f"Found 'used_tools' key in agent_response_dict with {tools_count} tools")
            if tools_count > 0:
                for idx, tool in enumerate(used_tools):
                    logger.info(f"Tool {idx+1}: {tool.get('name', 'unnamed')} at {tool.get('timestamp', 'unknown')}")
        else:
            logger.warning("'used_tools' key not found in agent_response_dict")
            # 에이전트 인스턴스에서 도구 사용 정보 직접 가져오기 시도
            if hasattr(self.agent, "get_used_tools"):
                try:
                    direct_tools = self.agent.get_used_tools()
                    if direct_tools:
                        logger.info(f"Retrieved {len(direct_tools)} tools directly from agent")
                        used_tools = direct_tools
                    else:
                        logger.info("No tools used according to agent's get_used_tools()")
                except Exception as e:
                    logger.error(f"Error getting used_tools directly from agent: {e}")
        
        tool_usage_artifact = self._create_tool_usage_artifact(used_tools)
        
        if agent_status == "completed":
            final_task_state = TaskState.COMPLETED
            # 아티팩트 구성
            if tool_usage_artifact:
                final_artifacts = [content_artifact, tool_usage_artifact]
            else:
                final_artifacts = [content_artifact]
        elif agent_status == "input_required":
            final_task_state = TaskState.INPUT_REQUIRED
            final_message = Message(role="agent", parts=parts)
        elif agent_status == "working":
            final_task_state = TaskState.WORKING
            final_message = Message(role="agent", parts=parts)
        else:  # 에러 또는 기타 상태
            final_task_state = TaskState.FAILED
            final_message = Message(role="agent", parts=[TextPart(text=f"Agent failed: {agent_response_dict.get('message', 'Unknown error')}")])
        
        return final_task_state, final_message, final_artifacts

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """동기적 'tasks/send' 요청을 처리합니다."""
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
            metadata = self._extract_metadata(request.params)
        except ValueError as e:
             logger.error(f"Could not get user query for task {request.params.id}: {e}")
             return SendTaskResponse(id=request.id, error=InvalidParamsError(message=str(e)))

        await self.upsert_task(request.params)
        await self.update_store(
            request.params.id, TaskStatus(state=TaskState.WORKING), artifacts=None
        )

        try:
            # 에이전트 호출
            agent_response_dict = await self.agent.invoke(query=query, session_id=request.params.sessionId, metadata=metadata)
            logger.info(f"Agent invoke completed for task {request.params.id}. Response: {agent_response_dict}")
            print(f"Agent invoke completed for task {request.params.id}. Response: {agent_response_dict}")

            # 응답에서 필요한 정보 추출
            agent_message_content = agent_response_dict.get("message", "Agent response format incorrect.")
            agent_raw_result = agent_response_dict.get("result") 

            # 메시지 파트 생성
            parts = [TextPart(text=agent_message_content)]
            if agent_raw_result is not None:
                if isinstance(agent_raw_result, (dict, list)):
                    parts.append(DataPart(data={"agent_result": agent_raw_result}))
                elif isinstance(agent_raw_result, (str, int, float, bool)):
                     parts.append(DataPart(data={"agent_result": agent_raw_result}))

            # 에이전트 응답 처리
            final_task_state, final_message, final_artifacts = self._process_agent_response(agent_response_dict, parts)

            # 상태 업데이트 및 응답 반환
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
        """스트리밍 'tasks/sendSubscribe' 요청을 처리합니다."""
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
        """에이전트의 스트림 메서드를 실행하고 SSE 큐에 업데이트를 푸시합니다."""
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

            # 스트리밍 이벤트 처리
            async for agent_event in self.agent.stream(query, session_id, metadata):
                event_type = agent_event.get("type")
                logger.debug(f"Agent stream event for task {task_id}: {event_type}")

                if event_type == "llm_token": 
                    token = agent_event.get("token", "")
                    if token:
                        artifact_part = TextPart(text=token)
                        temp_artifact = Artifact(parts=[artifact_part], index=0) 
                        await self.enqueue_events_for_sse(task_id, TaskArtifactUpdateEvent(id=task_id, artifact=temp_artifact))

                elif event_type == "status_update": 
                    status_message_text = agent_event.get("message", "Working...")
                    agent_event_status = agent_event.get("status")
                    task_state_update = TaskState.WORKING # 중간 업데이트는 기본적으로 WORKING
                    
                    if agent_event_status == "error":
                        task_state_update = TaskState.FAILED
                    
                    status_message = Message(role="agent", parts=[TextPart(text=status_message_text)])
                    intermediate_status = TaskStatus(state=task_state_update, message=status_message)
                    
                    await self.send_task_notification(Task(id=task_id, sessionId=session_id, status=intermediate_status))
                    await self.enqueue_events_for_sse(task_id, TaskStatusUpdateEvent(id=task_id, status=intermediate_status, final=False))
                    latest_task_state = task_state_update
                    
                elif event_type == "on_tool_start" or event_type == "on_tool_end":
                    # 도구 관련 이벤트를 로그에 기록
                    logger.info(f"Tool event: {event_type} for tool '{agent_event.get('name', 'unnamed')}'")
                    
                    # 도구 시작/종료 이벤트 발생 시 알림 전송
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

                elif event_type == "final_result":
                    final_agent_response = agent_event
                    logger.info(f"Received final_result from agent for task {task_id}. Status: {final_agent_response.get('status', 'unknown')}")
                    
                    # used_tools 정보 로깅
                    if 'used_tools' in final_agent_response:
                        logger.info(f"Found {len(final_agent_response['used_tools'])} used tools in final_result")
                        for idx, tool in enumerate(final_agent_response['used_tools']):
                            logger.debug(f"Tool {idx}: {tool.get('name', 'unnamed')}")
                    else:
                        logger.warning(f"No 'used_tools' found in final_result for task {task_id}")
                        # 결과 키 확인
                        logger.warning(f"final_result keys: {list(final_agent_response.keys())}")
                        # 도구 사용 정보가 없으면 에이전트에 직접 물어봄
                        if hasattr(self.agent, 'get_used_tools'):
                            direct_tools = self.agent.get_used_tools()
                            if direct_tools:
                                logger.info(f"Retrieved {len(direct_tools)} tools directly from agent")
                                # 결과에 도구 사용 정보 추가
                                final_agent_response['used_tools'] = direct_tools
                                logger.info(f"Added used_tools to final_agent_response manually")
                            else:
                                logger.info("Agent reported no tools used when asked directly")
                    break
            
            # 최종 응답 처리
            if final_agent_response:
                agent_message_content = final_agent_response.get("message", "Agent response format incorrect.")
                agent_raw_result = final_agent_response.get("result")
                
                parts = [TextPart(text=agent_message_content)]
                if agent_raw_result is not None:
                    if isinstance(agent_raw_result, (dict, list)):
                        parts.append(DataPart(data={"agent_result": agent_raw_result}))
                    elif isinstance(agent_raw_result, (str, int, float, bool)):
                        parts.append(DataPart(data={"agent_result": agent_raw_result}))

                final_task_state, final_message, final_artifacts = self._process_agent_response(final_agent_response, parts)

                final_task_status = TaskStatus(state=final_task_state, message=final_message)
                latest_task = await self.update_store(task_id, final_task_status, artifacts=final_artifacts)
                await self.send_task_notification(latest_task)

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

    async def send_task_notification(self, task: Task):
        """푸시 알림을 전송합니다."""
        # 푸시 정보가 구성된 경우에만 가져오기
        push_config: Optional[PushNotificationConfig] = None
        try:
            push_config = await self.get_push_notification_info(task.id) 
        except KeyError:
            logger.debug(f"No push notification info found for task {task.id}. Notification will not be sent.")
            return # 설정 없음, 메서드 종료
        except Exception as e:
            # 검색 중 발생할 수 있는 다른 오류 처리
            logger.error(f"Error retrieving push notification info for task {task.id}: {e}", exc_info=True)
            return

        # 푸시 설정 및 인증이 구성된 경우
        if push_config and self.notification_sender_auth:
            # 알림을 트리거해야 하는 작업 상태인지 확인
            if task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.INPUT_REQUIRED]:
                try:
                    await self.notification_sender_auth.send_push_notification(
                        push_url=push_config.url, 
                        task_info={"id": task.id, "status": task.status.model_dump(exclude_none=True)}
                    )
                    logger.info(f"Push notification sent for task {task.id} to {push_config.url}")
                except Exception as e:
                    logger.error(f"Failed to send push notification for task {task.id}: {e}")

    async def set_push_notification_info(self, task_id: str, push_notification_config: PushNotificationConfig):
        """푸시 알림 정보를 설정합니다."""
        return await super().set_push_notification_info(task_id, push_notification_config)
    
    async def on_resubscribe_to_task(
        self, request: TaskIdParams
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        """'tasks/resubscribe' 요청을 처리합니다."""
        logger.info(f"Received tasks/resubscribe request for task ID: {request.id}")
        task_id = request.id
        task_info = await self.get_task_info(task_id)

        if not task_info:
            logger.warning(f"Task not found for resubscribe: {task_id}")
            return JSONRPCResponse(id=request.id, error=TaskNotFoundError(message=f"Task with ID {task_id} not found."))

        try:
            sse_event_queue = await self.setup_sse_consumer(task_id, is_resubscribe=True)
            logger.info(f"SSE consumer re-established for task {task_id}")
            return self.dequeue_events_for_sse(request.id, task_id, sse_event_queue, send_past_events=True)
        except Exception as e:
            logger.error(f"Error setting up SSE resubscription for task {task_id}: {e}", exc_info=True)
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(message=f"Failed to set up resubscription: {e}")
            ) 