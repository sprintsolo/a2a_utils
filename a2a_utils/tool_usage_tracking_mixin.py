import logging
import datetime
from typing import List, Dict, Any, Optional # Python 3.10+ compatible

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
        self.used_tools: List[Dict[str, Any]] = []  # 도구 사용 기록을 저장할 리스트
        super().__init__(*args, **kwargs) # Ensure proper MRO handling
    
    async def track_tool_usage(self, tool_name: str, tool_input: Any, tool_output: Any) -> Dict[str, Any]:
        """도구 사용 내역을 추적하는 메서드"""
        tool_info: Dict[str, Any] = {
            "name": tool_name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), # Use timezone-aware UTC
            "input": str(tool_input), # Consider more structured input/output logging if possible
            "output": str(tool_output)
        }
        logger.info(f"tool_info: {tool_info}")
        self.used_tools.append(tool_info)
        logger.info(f"self.used_tools: {self.used_tools}")
        logger.info(f"Tracked tool usage: {tool_name}")
        return tool_info
    
    def get_used_tools(self) -> List[Dict[str, Any]]:
        """사용된 도구 목록 반환"""
        return self.used_tools
    
    def clear_used_tools(self) -> None:
        """도구 사용 기록 초기화"""
        self.used_tools = []
        logger.info("Cleared tool usage records")

    # add_tool_usage_to_result and LangGraph specific methods are removed for now.
    # They will be handled differently in the new architecture.
    # For example, creating TaskArtifactUpdateEvent within AgentExecutor. 