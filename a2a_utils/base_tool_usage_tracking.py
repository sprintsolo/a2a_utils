import logging
import datetime
from typing import List, Dict, Any, Optional, Callable, AsyncIterable
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  # 콘솔 출력
    ]
)
logger = logging.getLogger(__name__)

class ToolUsageTrackingMixin:
    """도구 사용 추적 기능을 제공하는 믹스인 클래스.
    
    이 클래스를 상속받은 에이전트는 도구 사용 내역을 자동으로 추적하고
    결과에 포함시킬 수 있습니다.
    """
    
    def __init__(self, *args, **kwargs):
        self.used_tools = []  # 도구 사용 기록을 저장할 리스트
        super().__init__(*args, **kwargs)
    
    async def track_tool_usage(self, tool_name: str, tool_input: Any, tool_output: Any) -> Dict[str, Any]:
        """도구 사용 내역을 추적하는 메서드"""
        tool_info = {
            "name": tool_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "input": str(tool_input),
            "output": str(tool_output)
        }
        self.used_tools.append(tool_info)
        logger.info(f"Tracked tool usage: {tool_name}")
        return tool_info
    
    def get_used_tools(self) -> List[Dict[str, Any]]:
        """사용된 도구 목록 반환"""
        return self.used_tools
    
    def clear_used_tools(self) -> None:
        """도구 사용 기록 초기화"""
        self.used_tools = []
        logger.info("Cleared tool usage records")
    
    def add_tool_usage_to_result(self, result: Dict[str, Any]) -> None:
        """결과 딕셔너리에 도구 사용 정보를 추가합니다.
        
        Args:
            result: 도구 사용 정보를 추가할 결과 딕셔너리
        """
        if not isinstance(result, dict):
            logger.warning(f"Result is not a dictionary, cannot add tool usage. Type: {type(result)}")
            return
            
        try:
            used_tools = self.get_used_tools()
            logger.warning(f"Adding {len(used_tools)} used tools to result dictionary")
            
            if len(used_tools) == 0:
                logger.warning("No tools were used. Empty list will be added to result.")
            else:
                logger.warning(f"Tools used: {[tool.get('name', 'unnamed') for tool in used_tools]}")
            
            # 결과 딕셔너리 상태 확인
            logger.warning(f"Result dictionary before adding used_tools: keys={list(result.keys())}")
            
            # 결과 딕셔너리에 도구 사용 정보 추가
            result["used_tools"] = used_tools
            
            # 확인을 위한 로깅
            if "used_tools" in result:
                logger.warning(f"Successfully added 'used_tools' to result. Contains {len(result['used_tools'])} tools")
                # 첫 번째 도구 정보를 예시로 로깅 (있는 경우)
                if result["used_tools"] and len(result["used_tools"]) > 0:
                    logger.warning(f"First tool example: {result['used_tools'][0]}")
                
                # 결과 딕셔너리 최종 상태 확인
                logger.warning(f"Final result dictionary after adding used_tools: keys={list(result.keys())}")
            else:
                logger.warning("Failed to add 'used_tools' to result dictionary")
                logger.warning(f"Current result dict: {result}")
        except Exception as e:
            logger.error(f"Error adding tool usage to result: {e}", exc_info=True)
    
    async def create_tool_callback(self) -> Any:
        """도구 사용 추적을 위한 콜백 함수 생성"""
        async def tool_callback(tool_name: str, tool_input: Any, tool_output: Any):
            await self.track_tool_usage(tool_name, tool_input, tool_output)
        return tool_callback
    
    # 새로 추가되는 LangGraph 이벤트 처리 관련 메서드들
    
    async def process_langgraph_event(self, event: Dict[str, Any]) -> None:
        """LangGraph 이벤트를 처리하여 도구 사용을 추적합니다.
        
        Args:
            event: LangGraph에서 생성된 이벤트 데이터
        """
        kind = event.get("event")
        if kind == "on_tool_end":
            tool_name = event.get("name", "unnamed_tool")
            tool_input = event.get("data", {}).get("input", "")
            tool_output = event.get("data", {}).get("output", "")
            
            # 도구 사용 추적
            await self.track_tool_usage(tool_name, tool_input, tool_output)
            logger.info(f"Processed tool event: {tool_name}")
    
    async def prepare_metadata_for_tool_tracking(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """도구 추적을 위한 메타데이터를 준비합니다.
        
        Args:
            metadata: 기존 메타데이터 (없으면 빈 딕셔너리 사용)
            
        Returns:
            도구 추적을 위한 콜백이 추가된 메타데이터
        """
        if metadata is None:
            metadata = {}
        
        # 기존 콜백이 있으면 저장
        original_callback = metadata.get("tool_callback")
        
        # 새 콜백 생성
        metadata["tool_callback"] = await self.create_tool_callback()
        
        # 기존 콜백 복원 로직은 필요하면 확장 클래스에서 구현
        
        logger.info("Prepared metadata with tool tracking callback")
        return metadata
    
    async def process_langgraph_events(self, events: List[Dict[str, Any]]) -> None:
        """여러 LangGraph 이벤트를 일괄 처리합니다.
        
        Args:
            events: 처리할 이벤트 목록
        """
        for event in events:
            await self.process_langgraph_event(event)
        logger.info(f"Processed {len(events)} LangGraph events")
    
    async def collect_events_and_track_tools(self, graph, input_data: Dict[str, Any], config: Dict[str, Any], 
                                             metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """LangGraph 실행 중 이벤트를 수집하고 도구 사용을 추적합니다.
        
        Args:
            graph: LangGraph 객체
            input_data: 그래프에 전달할 입력 데이터
            config: 그래프 실행 설정
            metadata: 메타데이터 (없으면 생성됨)
            
        Returns:
            수집된 이벤트 목록
        """
        # 도구 사용 기록 초기화
        self.clear_used_tools()
        
        # 메타데이터 준비
        metadata = await self.prepare_metadata_for_tool_tracking(metadata)
        
        all_events = []
        try:
            # 이벤트 스트림 처리
            async for event in graph.astream_events(input_data, config, version="v1"):
                all_events.append(event)
                await self.process_langgraph_event(event)
            
            logger.info(f"Collected {len(all_events)} events and tracked tool usage")
            return all_events
        except Exception as e:
            logger.error(f"Error collecting events and tracking tools: {e}", exc_info=True)
            return all_events 