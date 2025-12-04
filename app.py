"""
- tools.py should export:
  * get_tools() -> List[BaseTool] OR
  * tools: List[BaseTool]
  
- prompts.py should export:
  * get_prompt() -> ChatPromptTemplate OR
  * prompt: ChatPromptTemplate
  
- agent_builder.py should export (optional):
  * build_agent_executor(tools, prompt, llm) -> AgentExecutor OR
  * get_agent_executor(tools, prompt, llm) -> AgentExecutor
  
If these are not available, the app will use mock implementations.
"""

import streamlit as st
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# LangChain imports
try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_classic import hub
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate

# Set page config
st.set_page_config(
    page_title="✈️ 여행 계획 Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tool emoji mapping for visual appeal
TOOL_EMOJIS = {
    "날씨": "🌤️",
    "weather": "🌤️",
    "search_weather": "🌤️",
    "맛집": "🍽️",
    "restaurant": "🍽️",
    "관광": "🏛️",
    "attraction": "🏛️",
    "교통": "🚌",
    "transport": "🚌",
    "위험": "⚠️",
    "risk": "⚠️",
    "환율": "💱",
    "exchange": "💱",
    "예산": "💰",
    "budget": "💰",
    "일정": "📅",
    "schedule": "📅",
    "d-day": "📆",
    "검색": "🔍",
    "search": "🔍",
    "tavily": "🔍",
}


class StreamlitAgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to visualize agent thinking process in Streamlit"""
    
    def __init__(self, result_container=None):
        self.tool_executions = []
        self.current_tool = None
        self.result_container = result_container  # 실시간 결과 표시용 컨테이너
        
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts executing"""
        tool_name = serialized.get("name", "unknown_tool")
        self.current_tool = {
            "name": tool_name,
            "input": input_str,
            "status": "running",
            "start_time": datetime.now(),
            "output": None
        }
        self.tool_executions.append(self.current_tool)
        
        # Update session state
        if "tool_executions" not in st.session_state:
            st.session_state.tool_executions = []
        st.session_state.tool_executions.append(self.current_tool)
        
        # 실시간으로 실행 중인 tool 표시
        if self.result_container:
            with self.result_container:
                emoji = get_tool_emoji(tool_name)
                st.info(f"{emoji} **{tool_name}** 실행 중... | 입력: {str(input_str)[:100]}")
        
    def on_tool_end(self, output, **kwargs) -> None:
        """Called when a tool finishes executing"""
        if self.current_tool:
            self.current_tool["status"] = "completed"
            # output을 원본 형태로 저장 (dict일 수 있음)
            self.current_tool["output"] = output
            self.current_tool["end_time"] = datetime.now()
            
            # 실시간으로 완료된 tool 결과를 JSON 형식으로 표시
            if self.result_container:
                with self.result_container:
                    tool_name = self.current_tool["name"]
                    emoji = get_tool_emoji(tool_name)
                    
                    # JSON 형식으로 결과 표시
                    try:
                        import json
                        # output이 dict나 list인 경우 직접 st.json() 사용
                        if isinstance(output, (dict, list)):
                            with st.expander(f"{emoji} ✅ {tool_name} - 완료", expanded=True):
                                st.json(output)
                        else:
                            # 문자열인 경우 JSON 파싱 시도
                            try:
                                parsed = json.loads(str(output))
                                with st.expander(f"{emoji} ✅ {tool_name} - 완료", expanded=True):
                                    st.json(parsed)
                            except:
                                # JSON이 아닌 경우 일반 텍스트로 표시
                                with st.expander(f"{emoji} ✅ {tool_name} - 완료", expanded=True):
                                    st.text_area("**출력:**", value=str(output), height=200, disabled=True)
                    except Exception as e:
                        # 예외 발생 시 일반 텍스트로 표시
                        with st.expander(f"{emoji} ✅ {tool_name} - 완료", expanded=True):
                            st.text_area("**출력:**", value=str(output), height=200, disabled=True)
            
            self.current_tool = None
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool encounters an error"""
        if self.current_tool:
            self.current_tool["status"] = "error"
            self.current_tool["error"] = str(error)
            self.current_tool = None
    
    def get_tool_emoji(self, tool_name: str) -> str:
        """Get emoji for tool based on name"""
        tool_lower = tool_name.lower()
        for key, emoji in TOOL_EMOJIS.items():
            if key in tool_lower:
                return emoji
        return "🔧"


def get_tool_emoji(tool_name: str) -> str:
    """Helper function to get emoji for tool"""
    tool_lower = tool_name.lower()
    for key, emoji in TOOL_EMOJIS.items():
        if key in tool_lower:
            return emoji
    return "🔧"


def initialize_agent(use_mock: bool = False):
    """
    Initialize agent with graceful fallback to mock implementation
    
    Expected interface:
    - tools.py should export: get_tools() -> List[BaseTool] or tools: List[BaseTool]
    - prompts.py should export: get_prompt() -> ChatPromptTemplate or prompt: ChatPromptTemplate
    - agent_builder.py should export: build_agent_executor(tools, prompt, llm) -> AgentExecutor
    """
    # Import build_agent_executor from agent_builder (우선적으로 사용)
    try:
        from agent_builder import build_agent_executor
    except ImportError:
        build_agent_executor = None
    
    try:
        # Try to import from Team Member A's files
        # Import tools - tools.py exports ALL_TOOLS
        try:
            from tools import ALL_TOOLS
            tools = ALL_TOOLS
        except (ImportError, AttributeError):
            # Fallback: try get_tools() function
            try:
                from tools import get_tools
                tools = get_tools()
            except (ImportError, AttributeError):
                # Fallback: try tools variable
                try:
                    from tools import tools as tools_list
                    tools = tools_list
                except (ImportError, AttributeError):
                    raise ImportError("tools.py에서 tools를 가져올 수 없습니다")
        
        # Import prompts - prompts.py exports get_agent_prompt()
        try:
            from prompts import get_agent_prompt
            prompt = get_agent_prompt()
        except (ImportError, AttributeError):
            # Fallback: try get_prompt() function
            try:
                from prompts import get_prompt
                prompt = get_prompt()
            except (ImportError, AttributeError):
                # Fallback: try prompt variable
                try:
                    from prompts import prompt as prompt_template
                    prompt = prompt_template
                except (ImportError, AttributeError):
                    # Fallback to default prompt from hub
                    prompt = hub.pull("hwchase17/openai-functions-agent")
        
        # Initialize LLM
        llm = ChatOpenAI(
            model=st.session_state.get("model", "gpt-4o-mini"),
            temperature=st.session_state.get("temperature", 0),
            api_key=st.session_state.get("OPENAI_API_KEY")
        )
        
        # Use build_agent_executor from agent_builder if available
        if build_agent_executor:
            agent_executor = build_agent_executor(tools, prompt, llm)
        else:
            # Fallback: build agent executor directly
            agent = create_openai_tools_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        return agent_executor, False  # False means not using mock
        
    except (ImportError, AttributeError, TypeError) as e:
        # Fallback to mock implementation
        if use_mock:
            return create_mock_agent(), True  # True means using mock
        else:
            # Try one more time with alternative patterns
            try:
                # Try ALL_TOOLS
                try:
                    from tools import ALL_TOOLS
                    tools_list = ALL_TOOLS
                except (ImportError, AttributeError):
                    from tools import tools as tools_list
                
                # Try get_agent_prompt
                try:
                    from prompts import get_agent_prompt
                    prompt_template = get_agent_prompt()
                except (ImportError, AttributeError):
                    try:
                        from prompts import get_prompt
                        prompt_template = get_prompt()
                    except (ImportError, AttributeError):
                        from prompts import prompt as prompt_template
                
                llm = ChatOpenAI(
                    model=st.session_state.get("model", "gpt-4o-mini"),
                    temperature=st.session_state.get("temperature", 0),
                    api_key=st.session_state.get("OPENAI_API_KEY")
                )
                
                if build_agent_executor:
                    agent_executor = build_agent_executor(tools_list, prompt_template, llm)
                else:
                    agent = create_openai_tools_agent(llm, tools_list, prompt_template)
                    agent_executor = AgentExecutor(agent=agent, tools=tools_list, verbose=True)
                
                return agent_executor, False
            except:
                return create_mock_agent(), True


def create_mock_agent():
    """Create a mock agent for development when Team Member A's code is not ready"""
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    @tool
    def mock_weather_search(query: str) -> str:
        """날씨 정보를 검색합니다. 사용자가 제공한 목적지와 날짜에 대한 날씨 정보를 반환합니다."""
        return f"제주도 날씨: 맑음, 15°C, 강풍 주의 (1월 기준)"
    
    @tool
    def mock_restaurant_search(query: str) -> str:
        """맛집 정보를 검색합니다. 사용자의 선호사항(해산물 등)을 고려하여 추천합니다."""
        return f"해산물 맛집 추천: 해녀의 집, 바다향, 해물탕 전문점, 갈치조림 전문점"
    
    @tool
    def mock_attraction_search(query: str) -> str:
        """관광 명소 정보를 검색합니다. 목적지의 인기 관광지와 대중교통 정보를 제공합니다."""
        return f"관광 명소: 한라산, 성산일출봉, 우도, 섭지코지, 카멜리아힐, 아쿠아플라넷"
    
    mock_tools = [mock_weather_search, mock_restaurant_search, mock_attraction_search]
    
    llm = ChatOpenAI(
        model=st.session_state.get("model", "gpt-4o-mini"),
        temperature=st.session_state.get("temperature", 0),
        api_key=st.session_state.get("OPENAI_API_KEY")
    )
    
    # 여행 계획 전문 prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 전문 여행 계획 Agent입니다. 사용자가 제공한 정보를 바탕으로 즉시 여행 계획을 세워주세요.

사용자가 이미 제공한 정보:
- 목적지, 기간, 예산, 선호사항 등이 사용자 입력에 포함되어 있습니다.
- 추가 질문을 하지 말고, 제공된 정보를 바탕으로 바로 여행 계획을 세워주세요.

여행 계획에는 다음이 포함되어야 합니다:
1. 일정별 상세 계획 (날짜별로)
2. 추천 관광지 및 활동
3. 맛집 추천 (선호사항 반영)
4. 예산 배분
5. 날씨 정보 및 준비물
6. 교통 정보

사용 가능한 도구를 활용하여 실제 정보를 검색하고, 구체적이고 실용적인 여행 계획을 제공하세요."""),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    
    agent = create_openai_tools_agent(llm, mock_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=mock_tools, verbose=True)
    
    return agent_executor


def display_thinking_process():
    """Display the agent's thinking process with tool executions"""
    if "tool_executions" not in st.session_state or not st.session_state.tool_executions:
        return
    
    st.markdown("### 🤔 Agent가 작업 중입니다...")
    
    # Group tools by status
    running_tools = [t for t in st.session_state.tool_executions if t.get("status") == "running"]
    completed_tools = [t for t in st.session_state.tool_executions if t.get("status") == "completed"]
    error_tools = [t for t in st.session_state.tool_executions if t.get("status") == "error"]
    
    # Show running tools first
    if running_tools:
        st.markdown("#### 🔄 실행 중인 도구")
        for idx, tool_exec in enumerate(running_tools[-3:]):  # Show last 3 running
            tool_name = tool_exec.get("name", "unknown")
            emoji = get_tool_emoji(tool_name)
            st.info(f"{emoji} **{tool_name}** 실행 중... | 입력: {tool_exec.get('input', 'N/A')[:100]}")
    
    # Show completed tools
    if completed_tools:
        st.markdown("#### ✅ 완료된 도구")
    for idx, tool_exec in enumerate(completed_tools[-5:]):  # Show last 5 completed
        tool_name = tool_exec.get("name", "unknown")
        emoji = get_tool_emoji(tool_name)
        start_time = tool_exec.get("start_time", "")
        unique_key = f"completed_{tool_name}_{idx}_{hash(str(start_time))}"
        
        with st.expander(f"{emoji} ✅ {tool_name} - 완료", expanded=False, key=f"expander_{unique_key}"):
            st.info(f"**입력:** {tool_exec.get('input', 'N/A')}")
            output = tool_exec.get('output', 'N/A')
            if len(str(output)) > 500:
                st.text_area("**출력:**", value=str(output)[:500] + "...", height=100, disabled=True, key=f"output_{unique_key}")
            else:
                st.text_area("**출력:**", value=str(output), height=100, disabled=True, key=f"output_{unique_key}")
    
    # Show error tools
    if error_tools:
        st.markdown("#### ❌ 오류가 발생한 도구")
    for idx, tool_exec in enumerate(error_tools):
        tool_name = tool_exec.get("name", "unknown")
        emoji = get_tool_emoji(tool_name)
        start_time = tool_exec.get("start_time", "")
        unique_key = f"error_{tool_name}_{idx}_{hash(str(start_time))}"
        
        with st.expander(f"{emoji} ❌ {tool_name} - 오류", expanded=True, key=f"expander_error_{unique_key}"):
            st.error(f"**에러:** {tool_exec.get('error', 'Unknown error')}")
            st.info(f"**입력:** {tool_exec.get('input', 'N/A')}")


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "tool_executions" not in st.session_state:
        st.session_state.tool_executions = []
    
    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    if "TAVILY_API_KEY" not in st.session_state:
        st.session_state.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4o-mini"
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    
    if "use_mock" not in st.session_state:
        st.session_state.use_mock = False
    
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


def execute_agent_query(user_query: str):
    """Execute agent with user query and display results"""
    # Clear previous tool executions
    st.session_state.tool_executions = []
    
    # Initialize agent
    try:
        agent_executor, is_mock = initialize_agent(use_mock=st.session_state.use_mock)
        
        # Display assistant message area
        with st.chat_message("assistant"):
            if is_mock:
                st.info("⚠️ 개발 모드: Mock Agent를 사용 중입니다")
            
            # 실시간 tool 결과 표시용 컨테이너
            realtime_results = st.container()
            
            # Initialize callback handler with result container
            callback_handler = StreamlitAgentCallbackHandler(result_container=realtime_results)
            
            # Execute agent
            with st.spinner("🤔 여행 계획 생성 중..."):
                try:
                    result = agent_executor.invoke(
                        {"input": user_query},
                        {"callbacks": [callback_handler]}
                    )
                    
                    # Display final response
                    st.markdown("---")
                    st.markdown("### 📋 최종 여행 계획")
                    response = result.get("output", "응답을 생성할 수 없습니다.")
                    
                    # D-Day 정보가 tool_executions에 있는지 확인하고 추가
                    d_day_info = None
                    for tool_exec in st.session_state.tool_executions:
                        if tool_exec.get("name") == "calculate_d_day" and tool_exec.get("status") == "completed":
                            output = tool_exec.get("output")
                            if isinstance(output, dict):
                                d_day_info = output
                            elif isinstance(output, str):
                                try:
                                    import json
                                    d_day_info = json.loads(output)
                                except:
                                    pass
                            break
                    
                    # D-Day 정보가 있으면 응답 앞에 추가
                    if d_day_info and "formatted" in d_day_info:
                        d_day_section = f"""
### 📅 여행 D-Day 정보

- **출발일**: {d_day_info.get('date', 'N/A')}
- **D-Day**: {d_day_info.get('formatted', 'N/A')}
- **남은 일수**: {d_day_info.get('d_day', 'N/A')}일
- **준비 기간**: {d_day_info.get('preparation', {}).get('weeks', 'N/A')}주
- **준비 긴급도**: {d_day_info.get('preparation', {}).get('urgency', 'N/A')}

---
"""
                        response = d_day_section + response
                    
                    st.markdown(response)
                    
                    # Save to chat history (한 번만 저장)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "tool_executions": st.session_state.tool_executions.copy()
                    })
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # Check for specific error types
                    if "api" in error_msg.lower() or "key" in error_msg.lower():
                        st.error("❌ API 키 오류가 발생했습니다. 사이드바에서 API 키를 확인해주세요.")
                    elif "rate limit" in error_msg.lower():
                        st.error("⏱️ API 호출 한도에 도달했습니다. 잠시 후 다시 시도해주세요.")
                    else:
                        st.error("😔 잠시 문제가 생겼어요. 다시 시도해주세요.")
                    
                    # Show error details in expander for debugging
                    with st.expander("🔍 오류 상세 정보 (개발용)"):
                        st.exception(e)
                    
                    # Save error message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.",
                        "error": error_msg
                    })
    
    except Exception as e:
        st.error("😔 Agent 초기화 중 문제가 발생했습니다.")
        with st.expander("🔍 오류 상세 정보 (개발용)"):
            st.exception(e)


def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Title
    st.title("✈️ 여행 계획 Agent")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # API Keys
        st.subheader("🔑 API 키")
        openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.OPENAI_API_KEY,
            type="password",
            help="OpenAI API 키를 입력하세요"
        )
        if openai_key:
            st.session_state.OPENAI_API_KEY = openai_key
            os.environ["OPENAI_API_KEY"] = openai_key
        
        tavily_key = st.text_input(
            "Tavily API Key",
            value=st.session_state.TAVILY_API_KEY,
            type="password",
            help="Tavily API 키를 입력하세요 (선택사항)"
        )
        if tavily_key:
            st.session_state.TAVILY_API_KEY = tavily_key
            os.environ["TAVILY_API_KEY"] = tavily_key
        
        st.markdown("---")
        
        # Model Settings
        st.subheader("🤖 모델 설정")
        model = st.selectbox(
            "모델 선택",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            index=0 if st.session_state.model == "gpt-4o-mini" else 1
        )
        st.session_state.model = model
        
        temperature = st.slider("Temperature", 0.0, 1.0, float(st.session_state.temperature), 0.1)
        st.session_state.temperature = temperature
        
        st.markdown("---")
        
        # Development Mode
        st.subheader("🛠️ 개발 모드")
        use_mock = st.checkbox("Mock Agent 사용 (개발용)", value=st.session_state.use_mock)
        st.session_state.use_mock = use_mock
        
        if use_mock:
            st.warning("⚠️ Mock Agent 모드 활성화")
        
        st.markdown("---")
        
        # Clear Chat
        if st.button("🗑️ 채팅 기록 지우기"):
            st.session_state.messages = []
            st.session_state.tool_executions = []
            st.rerun()
    
    # Main area
    # Input form for quick start
    with st.expander("📝 빠른 시작 (여행 정보 입력)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            destination = st.text_input("목적지", placeholder="예: 제주도")
            duration = st.text_input("기간", placeholder="예: 3박 4일")
        with col2:
            budget = st.text_input("예산", placeholder="예: 50만원")
            preferences = st.text_input("선호사항", placeholder="예: 해산물 좋아해")
        
        example_query = st.button("📋 예제 쿼리 사용", help="3박 4일 제주도 여행 계획 짜줘, 예산 50만원, 해산물 좋아해")
        
        if st.button("🚀 여행 계획 요청") or example_query:
            query_parts = []
            if destination or example_query:
                if example_query:
                    user_input = "3박 4일 제주도 여행 계획 짜줘, 예산 50만원, 해산물 좋아해"
                else:
                    query_parts = []
                    if destination:
                        query_parts.append(f"목적지: {destination}")
                    if duration:
                        query_parts.append(f"기간: {duration}")
                    if budget:
                        query_parts.append(f"예산: {budget}")
                    if preferences:
                        query_parts.append(f"선호사항: {preferences}")
                    user_input = ", ".join(query_parts) if query_parts else destination
                
                # Add to chat
                st.session_state.messages.append({"role": "user", "content": user_input})
                # Store the query to process it after rerun
                st.session_state.pending_query = user_input
                st.rerun()
    
    # Display chat history (이미 완료된 메시지만 표시)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show tool executions if this is an assistant message (완료된 것만)
            if message["role"] == "assistant" and "tool_executions" in message:
                # 완료된 tool executions만 표시 (실시간 업데이트는 execute_agent_query에서 처리)
                if st.session_state.tool_executions:
                    # 이미 완료된 메시지이므로 간단히 표시
                    pass
    
    # Process pending query from button click
    if st.session_state.pending_query:
        pending = st.session_state.pending_query
        st.session_state.pending_query = None  # Clear pending query
        execute_agent_query(pending)
    
    # Chat input
    if prompt := st.chat_input("여행 계획을 요청하세요..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        execute_agent_query(prompt)


if __name__ == "__main__":
    main()
