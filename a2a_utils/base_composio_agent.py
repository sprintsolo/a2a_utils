import os
import logging
import re
import time # time 모듈 임포트
from abc import ABC, abstractmethod
from typing import Any, Dict, AsyncIterable, List, Literal, Optional, Tuple
import datetime  # 타임스탬프 생성에 필요

# LangChain 디버그 모드 활성화
import langchain
# langchain.debug = True

from pydantic import BaseModel, Field
from langchain_core.messages.ai import AIMessageChunk
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
# composio_openai를 사용하거나, composio를 직접 사용할 수 있습니다. 사용자 제공 예시에 따라 composio_openai를 가정합니다.
# from composio_openai import ComposioToolSet 
from composio_langchain import ComposioToolSet # 수정된 임포트 경로
# from composio.client.enums import Action, App # 필요시 활성화

from .base_langgraph_agent import BaseLangGraphAgent
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  # 콘솔 출력
    ]
)
logger = logging.getLogger(__name__)

# .env 파일 로드는 애플리케이션 시작 지점에서 한 번 수행하는 것이 좋습니다.
# from dotenv import load_dotenv
# load_dotenv()

MAX_CONVERSATION_HISTORY_MESSAGES = 4 # 일시적으로 대화 기록 관리 비활성화

class MissingWorkflowParamsError(ValueError):
    """Custom error for missing workflow parameters."""
    pass

class BaseResponseFormat(BaseModel):
    """Defines the base response structure from an agent."""
    status: Literal["completed", "error", "working", "input_required"] = "error"
    message: str
    result: Optional[Any] = None # 다양한 결과 타입을 허용
    tool_used: Optional[str] = None
    tool_input: Optional[Dict | str] = None

class BaseComposioAgent(BaseLangGraphAgent):
    """A2A 호스트에서 Composio 도구를 사용하는 에이전트를 위한 기본 클래스.
    
    이 클래스는 BaseLangGraphAgent를 상속받아 Composio 도구 통합 기능을 추가합니다.
    """
    
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"] # 기본값, 필요시 재정의
    DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful assistant. Your primary function is to assist the user with their tasks using the available tools." # 기본 시스템 메시지

    def __init__(self, openai_api_key: str, composio_api_key: Optional[str] = None, model_name: str = "gpt-4.1-mini"):
        """
        BaseComposioAgentLogic 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            composio_api_key: Composio API 키 (선택 사항)
            model_name: 사용할 OpenAI 모델명
        """
        # 부모 클래스(BaseLangGraphAgent) 초기화 먼저 호출
        super().__init__()
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be provided.")
        
        self.openai_api_key = openai_api_key
        self.composio_api_key = composio_api_key
        self.model_name = model_name
        
        try:
            self.model = ChatOpenAI(openai_api_key=self.openai_api_key, model=self.model_name)
        except Exception as e:
            logger.error(f"Error initializing ChatOpenAI model: {e}")
            raise

        if self.composio_api_key:
            try:
                self.tool_set = ComposioToolSet(api_key=self.composio_api_key)
            except Exception as e:
                logger.error(f"Error initializing ComposioToolSet: {e}")
                # API 키가 필수적인 경우 여기서 raise 할 수 있음
                self.tool_set = None # 또는 실패 처리
                logger.warning("ComposioToolSet initialization failed. Composio tools will be unavailable.")
        else:
            self.tool_set = None
            logger.warning("COMPOSIO_API_KEY not provided. Composio tools will be unavailable.")

        self.memory = MemorySaver() # 각 에이전트 인스턴스 또는 세션별 메모리 관리 고려

        logger.info(f"BaseComposioAgentLogic initialized for {self.__class__.__name__}")

    @abstractmethod
    async def _get_tools(self, external_user_id: Optional[str] = None) -> List[BaseTool]:
        """
        Retrieves tools for the agent. 
        If external_user_id is provided and ComposioToolSet is available, 
        it should fetch tools specific to that user.
        Subclasses must implement this to provide tools.
        """
        pass

    def _extract_external_user_id(self, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extracts external_user_id from metadata."""
        if not isinstance(metadata, dict):
            logger.debug("Metadata is missing or not a dictionary.")
            return None

        external_user_id = metadata.get("external_user_id")
        if not external_user_id or not isinstance(external_user_id, str):
            logger.debug("'external_user_id' missing or invalid in metadata.")
            return None
        
        logger.info(f"Extracted external_user_id: {external_user_id}")
        return external_user_id

    # BaseLangGraphAgent의 abstractmethod 구현
    async def parse_agent_output(self, agent_output: Any) -> Dict[str, Any]:
        """
        에이전트 출력을 파싱하여 BaseResponseFormat 형식으로 변환합니다.
        
        Args:
            agent_output: LangGraph 결과
            
        Returns:
            {status, message, result, tool_used} 형식의 응답
        """
        # messages 필드에서 AIMessage 찾기
        final_message = None
        if isinstance(agent_output, dict) and "messages" in agent_output:
            messages = agent_output.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    final_message = msg
                    break
        elif isinstance(agent_output, AIMessage):
            final_message = agent_output
        
        if not final_message:
            logger.warning("No AIMessage found in agent output")
            return BaseResponseFormat(
                status="error",
                message="Agent did not provide a final message."
            ).model_dump()
        
        # 기본 응답 생성
        response = BaseResponseFormat(
            status="completed",
            message=final_message.content,
            result=None
        )
        
        # 텍스트에서 상태 추정
        # content = final_message.content.lower()
        # if any(kw in content for kw in ["error", "failed", "unable", "cannot", "can't", "sorry"]):
        #     response.status = "error"
        # elif any(kw in content for kw in ["need more information", "need additional"]):
        #     response.status = "input_required"
        
        return response.model_dump()
    
    # BaseLangGraphAgent.invoke와 stream 메서드는 그대로 상속받아 사용
    
    # 새로운 메서드: _initialize_graph (기존 코드에서 공통 부분 추출)
    async def _initialize_graph(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        LangGraph를 초기화하고 도구를 설정합니다.
        
        Args:
            metadata: 메타데이터 (external_user_id 포함 가능)
        """
        external_user_id = self._extract_external_user_id(metadata)
        tools_for_graph = await self._get_tools(external_user_id)
        
        if not tools_for_graph:
            logger.warning(f"No tools available for agent (user: {external_user_id}). Proceeding without tools.")
        else:
            logger.info(f"Initialized graph with {len(tools_for_graph)} tools")
        
        self.tools = tools_for_graph
        self.graph = create_react_agent(
            self.model,
            tools=tools_for_graph,
            checkpointer=self.memory
        )
    
    # 기존 invoke와 stream 메서드 대신 BaseLangGraphAgent의 메서드를 사용
    # 다만, 각 호출 전에 _initialize_graph를 호출하도록 메서드 오버라이드
    
    async def invoke(self, query: str, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        에이전트를 동기적으로 호출합니다.
        Composio 에이전트에 특화된 초기화 및 상세 로깅을 포함합니다.
        
        Args:
            query: 사용자 입력 쿼리
            session_id: 세션 ID
            metadata: 추가 메타데이터 (external_user_id 포함 가능)
            
        Returns:
            에이전트 응답 딕셔너리 (도구 사용 정보 포함)
        """
        agent_name = self.__class__.__name__
        logger.info(f"{agent_name}.invoke starting for session={session_id}, query='{query[:70]}...'")
        request_received_time = time.time()

        self.clear_used_tools()
        logger.debug(f"Tool usage records cleared for {agent_name} (invoke)")

        # 호출 전 그래프 초기화 (사용자별 도구 설정)
        await self._initialize_graph(metadata)
        
        try:
            # 부모 클래스(BaseLangGraphAgent)의 invoke 메서드 호출
            result = await super().invoke(query=query, session_id=session_id, metadata=metadata)
            
            # 상세 도구 사용 로깅 (BaseLangGraphAgent가 이미 result에 used_tools를 추가해야 함)
            used_tools = self.get_used_tools()
            if used_tools:
                logger.info(f"{agent_name}.invoke collected {len(used_tools)} used tools:")
                for idx, tool_info in enumerate(used_tools):
                    logger.info(f"  Tool {idx+1}: name={tool_info.get('name', 'unnamed')}, timestamp={tool_info.get('timestamp', 'unknown')}")
                
                # 결과에 used_tools가 있는지, 개수가 맞는지 확인 (디버깅/검증 목적)
                if 'used_tools' not in result:
                    logger.warning(
                        f"Result from super().invoke for {agent_name} did not contain 'used_tools'. "
                        f"This is unexpected as BaseLangGraphAgent should add it. Manually adding."
                    )
                    # BaseLangGraphAgent.invoke에서 이미 추가했어야 하지만, 안전장치로 추가
                    # self.add_tool_usage_to_result(result) # 이 호출은 result가 get_used_tools와 동기화되도록 보장
                                                        # 하지만 BaseLangGraphAgent에서 이미 처리됨. 중복일 수 있음.
                                                        # 만약 BaseLangGraphAgent의 add_tool_usage_to_result가 충분하다면 이 부분은 로깅으로 대체 가능
                elif len(result.get('used_tools', [])) != len(used_tools):
                    logger.warning(
                        f"Mismatch in tool count for {agent_name}.invoke: "
                        f"{len(result.get('used_tools', []))} in result vs {len(used_tools)} tracked. "
                        f"Result content: {result.get('used_tools')}"
                    )
            else:
                logger.info(f"No tools were used during {agent_name}.invoke")

            processing_time = time.time() - request_received_time
            logger.info(
                f"{agent_name}.invoke completed for session={session_id}, "
                f"status={result.get('status', 'unknown')}, processing_time={processing_time:.2f}s"
            )
            # BaseLangGraphAgent.invoke가 이미 used_tools를 result에 추가하므로, 여기서 재확인/추가 로직은 주로 로깅/검증용.
            # self.add_tool_usage_to_result(result)를 명시적으로 호출하는 대신, 부모 클래스를 신뢰하는 것이 일반적.


            return result
            
        except Exception as e:
            processing_time = time.time() - request_received_time
            logger.error(f"Error during {agent_name}.invoke for session={session_id}: {e}, processing_time={processing_time:.2f}s", exc_info=True)
            error_response = {
                "status": "error",
                "message": f"처리 중 오류가 발생했습니다: {str(e)}",
                "result": None
            }
            # 오류 발생 시에도 지금까지 사용된 도구 정보 추가
            self.add_tool_usage_to_result(error_response) 
            return error_response
    
    async def stream(self, query: str, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> AsyncIterable[Dict[str, Any]]:
        """
        에이전트를 스트리밍 방식으로 호출합니다.
        Composio 에이전트에 특화된 초기화 및 상세 로깅을 포함합니다.

        Args:
            query: 사용자 입력 쿼리
            session_id: 세션 ID
            metadata: 추가 메타데이터 (external_user_id 포함 가능)
            
        Yields:
            스트리밍 이벤트 딕셔너리 (최종 결과에 도구 사용 정보 포함)
        """
        agent_name = self.__class__.__name__
        logger.info(f"{agent_name}.stream starting for session={session_id}, query='{query[:70]}...'")
        stream_start_time = time.time()

        self.clear_used_tools()
        logger.debug(f"Tool usage records cleared for {agent_name} (stream)")

        # 호출 전 그래프 초기화 (사용자별 도구 설정)
        await self._initialize_graph(metadata)
        
        try:
            # 부모 클래스(BaseLangGraphAgent)의 stream 메서드 호출
            async for event in super().stream(query=query, session_id=session_id, metadata=metadata):
                if event.get("type") == "final_result":
                    # 상세 도구 사용 로깅 (BaseLangGraphAgent가 이미 event에 used_tools를 추가해야 함)
                    used_tools = self.get_used_tools()
                    if used_tools:
                        logger.info(f"{agent_name}.stream collected {len(used_tools)} used tools for final_result:")
                        for idx, tool_info in enumerate(used_tools):
                            logger.info(f"  Tool {idx+1}: name={tool_info.get('name', 'unnamed')}, timestamp={tool_info.get('timestamp', 'unknown')}")
                        
                        if 'used_tools' not in event:
                            logger.warning(
                                f"Final_result event from super().stream for {agent_name} did not contain 'used_tools'. "
                                f"This is unexpected as BaseLangGraphAgent should add it."
                            )
                        elif len(event.get('used_tools', [])) != len(used_tools):
                            logger.warning(
                                f"Mismatch in tool count for {agent_name}.stream final_result: "
                                f"{len(event.get('used_tools', []))} in event vs {len(used_tools)} tracked. "
                                f"Event content: {event.get('used_tools')}"
                            )
                    else:
                        logger.info(f"No tools were reported by get_used_tools() at the end of {agent_name}.stream")
                    
                    processing_time = time.time() - stream_start_time
                    logger.info(
                        f"{agent_name}.stream completed for session={session_id}, "
                        f"status={event.get('status', 'unknown')}, time={processing_time:.2f}s"
                    )
                yield event
                
        except Exception as e:
            processing_time = time.time() - stream_start_time
            logger.error(f"Error during {agent_name}.stream for session={session_id}: {e}, processing_time={processing_time:.2f}s", exc_info=True)
            error_response = {
                "type": "final_result",
                "status": "error",
                "message": f"스트리밍 중 오류가 발생했습니다: {str(e)}",
                "result": None
            }
            # 오류 발생 시에도 지금까지 사용된 도구 정보 추가
            self.add_tool_usage_to_result(error_response) 
            yield error_response

# 사용 예시 (실제로는 FastAPI 등에서 라우트 핸들러 내에서 사용될 것임):
# class MySpecificAgent(BaseComposioAgentLogic):
#     def _get_system_instruction(self) -> str:
#         return "You are a specialized assistant for XYZ tasks."

#     def _get_tools(self, external_user_id: Optional[str] = None) -> List[BaseTool]:
#         if self.tool_set and external_user_id:
#             try:
#                 # entity_id는 Composio에서 사용자를 식별하는 ID일 수 있습니다.
#                 # get_tools의 정확한 파라미터는 Composio SDK 문서를 참조해야 합니다.
#                 logger.info(f"Fetching Composio tools for entity_id: {external_user_id}")
#                 return self.tool_set.get_tools(entity_id=external_user_id)
#             except Exception as e:
#                 logger.error(f"Failed to fetch Composio tools for {external_user_id}: {e}")
#                 return [] # 또는 기본 도구 세트 반환
#         elif self.tool_set : # external_user_id가 없지만 tool_set은 있는 경우 (공용 Composio 도구?)
#              logger.info(f"Fetching generic Composio tools.")
#              # entity_id 없이 호출하는 것은 ComposioToolSet의 get_tools 시그니처에 따라 다름
#              # 만약 entity_id가 필수라면 이 부분은 에러를 발생시키거나 다른 처리가 필요
#              # 여기서는 get_tools()가 ID 없이도 호출 가능하다고 가정하거나, 또는 항상 ID가 필요하다고 가정하고 로직을 구성해야 함.
#              # 사용자 제공 예시에서는 entity_id가 항상 사용되므로, 해당 케이스를 기본으로 가정합니다.
#              # 일반 도구를 가져오는 명시적인 방법이 없다면, 이 케이스는 제거하거나 수정해야 합니다.
#              # return self.tool_set.get_tools() # 주석 처리 또는 수정
#         return [] # ComposioToolSet이 없거나 ID가 없는 경우 빈 리스트 반환 