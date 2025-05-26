import logging
import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterable, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from .base_tool_usage_tracking import ToolUsageTrackingMixin
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  # 콘솔 출력
    ]
)
logger = logging.getLogger(__name__)

class BaseLangGraphAgent(ABC, ToolUsageTrackingMixin):
    """LangGraph를 사용하는 에이전트를 위한 기본 클래스.
    
    이 클래스는 ToolUsageTrackingMixin을 상속받아 도구 추적 기능을 제공하며,
    LangGraph를 사용하는 에이전트의 공통 기능을 구현합니다.
    
    Attributes:
        SUPPORTED_CONTENT_TYPES: 에이전트가 지원하는 콘텐츠 타입 목록
    """
    
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
    
    def __init__(self, *args, **kwargs):
        """
        BaseLangGraphAgent 초기화
        """
        # ToolUsageTrackingMixin 초기화 호출
        super().__init__(*args, **kwargs)
        
        # 이 클래스를 상속받는 클래스에서 설정해야 할 필드들
        self.graph = None  # LangGraph 객체
        self.tools = []    # 도구 목록
    
    @abstractmethod
    def _get_system_instruction(self) -> str:
        """에이전트의 기본 시스템 지침을 반환합니다.
        하위 클래스에서 특정 에이전트의 행동을 정의하기 위해 이 메소드를 오버라이드해야 합니다.
        """
        pass

    async def format_input_for_graph(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        사용자 쿼리를 LangGraph 입력 형식으로 변환합니다.
        
        Args:
            query: 사용자 입력 쿼리
            
        Returns:
            LangGraph에 전달할 형식의 입력 데이터
        """
        # 기본 구현은 LangGraph의 일반적인 입력 형식을 따름
        # return {"messages": [("user", query)]} # 이전 코드 삭제
        system_instruction = self._get_system_instruction()
        messages = []
        if system_instruction: # 시스템 지침이 비어있지 않은 경우에만 추가
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=query))
        return {"messages": messages}
    
    async def handle_stream_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        스트리밍 중 발생한 이벤트를 처리하고 필요한 경우 응답을 생성합니다.
        
        Args:
            event: LangGraph에서 생성된 이벤트
            
        Returns:
            클라이언트에게 전달할 이벤트 또는 None
        """
        kind = event.get("event")
        
        if kind == "on_tool_start":
            tool_name = event.get("name", "unnamed_tool")
            return {
                "type": "status_update",
                "status": "working",
                "message": f"도구 '{tool_name}'를 사용하는 중..."
            }
        elif kind == "on_tool_end":
            tool_name = event.get("name", "unnamed_tool")
            return {
                "type": "status_update",
                "status": "working",
                "message": f"도구 '{tool_name}' 사용 완료, 결과 처리 중..."
            }
        
        return None  # 클라이언트에게 전달할 필요가 없는 이벤트
    
    @abstractmethod
    async def parse_agent_output(self, agent_output: Any) -> Dict[str, Any]:
        """
        에이전트 출력을 파싱하여 표준 응답 형식으로 변환합니다.
        
        Args:
            agent_output: 에이전트의 원시 출력
            
        Returns:
            표준화된 응답 딕셔너리
        """
        pass
    
    async def invoke(self, query: str, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        에이전트를 동기적으로 호출합니다.
        
        Args:
            query: 사용자 입력 쿼리
            session_id: 세션 ID
            metadata: 추가 메타데이터
            
        Returns:
            에이전트 응답 딕셔너리
        """
        if self.graph is None:
            raise ValueError("Agent graph is not initialized. Make sure to set self.graph in __init__.")
        
        logger.info(f"Invoking agent for session {session_id} with query: {query[:100]}...")
        config = {"configurable": {"thread_id": session_id}}
        
        # 입력 포맷팅
        input_data = await self.format_input_for_graph(query)
        
        # 이벤트 수집 및 도구 추적
        await self.collect_events_and_track_tools(
            graph=self.graph,
            input_data=input_data,
            config=config,
            metadata=metadata
        )
        
        try:
            # 최종 결과 가져오기
            result = self.graph.invoke(input_data, config)
            
            # 결과를 표준 형식으로 변환
            parsed_result = await self.parse_agent_output(result)
            
            # 도구 사용 정보 추가
            self.add_tool_usage_to_result(parsed_result)
            
            return parsed_result
        except Exception as e:
            logger.error(f"Error during agent invocation: {e}", exc_info=True)
            error_result = {
                "status": "error",
                "message": f"에이전트 실행 중 오류 발생: {str(e)}"
            }
            self.add_tool_usage_to_result(error_result)
            return error_result
    
    async def stream(self, query: str, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> AsyncIterable[Dict[str, Any]]:
        """
        에이전트를 스트리밍 방식으로 호출합니다.
        
        Args:
            query: 사용자 입력 쿼리
            session_id: 세션 ID
            metadata: 추가 메타데이터
            
        Yields:
            스트리밍 이벤트 딕셔너리
        """
        if self.graph is None:
            raise ValueError("Agent graph is not initialized. Make sure to set self.graph in __init__.")
        
        logger.info(f"Streaming agent for session {session_id} with query: {query[:100]}...")
        config = {"configurable": {"thread_id": session_id}}
        
        # 도구 사용 기록 초기화
        self.clear_used_tools()
        
        # 메타데이터 준비
        if metadata is None:
            metadata = {}
        metadata = await self.prepare_metadata_for_tool_tracking(metadata)
        
        # 입력 포맷팅
        input_data = await self.format_input_for_graph(query)
        
        final_ai_message = None
        
        try:
            # 이벤트 스트림 처리
            async for event in self.graph.astream_events(input_data, config, version="v1"):
                # 도구 사용 이벤트 처리
                await self.process_langgraph_event(event)
                
                # 클라이언트에게 전달할 이벤트 생성
                client_event = await self.handle_stream_event(event)
                if client_event:
                    yield client_event
                
                # AI 메시지 캡처
                if event.get("event") == "on_chat_model_end":
                    output = event.get("data", {}).get("output")
                    if hasattr(output, "content") and output.content:
                        final_ai_message = output
            
            # 최종 결과 반환
            if final_ai_message:
                parsed_final_output = await self.parse_agent_output(final_ai_message)
                self.add_tool_usage_to_result(parsed_final_output)
                
                yield {
                    "type": "final_result",
                    **parsed_final_output
                }
            else:
                logger.warning("Stream completed without final AI message")
                error_output = {
                    "type": "final_result",
                    "status": "error",
                    "message": "에이전트 응답을 처리하는 중 문제가 발생했습니다."
                }
                self.add_tool_usage_to_result(error_output)
                yield error_output
                
        except Exception as e:
            logger.error(f"Error during agent streaming: {e}", exc_info=True)
            error_output = {
                "type": "final_result",
                "status": "error",
                "message": f"스트리밍 중 오류 발생: {str(e)}"
            }
            self.add_tool_usage_to_result(error_output)
            yield error_output 