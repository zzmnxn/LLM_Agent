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
    
    def __init__(self):
        self.tool_executions = []
        self.current_tool = None
        
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
        
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes executing"""
        if self.current_tool:
            self.current_tool["status"] = "completed"
            self.current_tool["output"] = str(output)
            self.current_tool["end_time"] = datetime.now()
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
        # Try different import patterns for tools
        try:
            from tools import get_tools
            tools = get_tools()
        except (ImportError, AttributeError):
            try:
                from tools import tools as tools_list
                tools = tools_list
            except (ImportError, AttributeError):
                raise ImportError("tools.py에서 tools를 가져올 수 없습니다")
        
        # Try different import patterns for prompts
        try:
            from prompts import get_prompt
            prompt = get_prompt()
        except (ImportError, AttributeError):
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
                from tools import tools as tools_list
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
        for tool_exec in running_tools[-3:]:  # Show last 3 running
            tool_name = tool_exec.get("name", "unknown")
            emoji = get_tool_emoji(tool_name)
            st.info(f"{emoji} **{tool_name}** 실행 중... | 입력: {tool_exec.get('input', 'N/A')[:100]}")
    
    # Show completed tools
    if completed_tools:
        st.markdown("#### ✅ 완료된 도구")
    for tool_exec in completed_tools[-5:]:  # Show last 5 completed
        tool_name = tool_exec.get("name", "unknown")
        emoji = get_tool_emoji(tool_name)
        
        with st.expander(f"{emoji} ✅ {tool_name} - 완료", expanded=False):
            st.info(f"**입력:** {tool_exec.get('input', 'N/A')}")
            output = tool_exec.get('output', 'N/A')
            if len(str(output)) > 500:
                st.text_area("**출력:**", value=str(output)[:500] + "...", height=100, disabled=True, key=f"output_{tool_name}_{id(tool_exec)}")
            else:
                st.text_area("**출력:**", value=str(output), height=100, disabled=True, key=f"output_{tool_name}_{id(tool_exec)}")
    
    # Show error tools
    if error_tools:
        st.markdown("#### ❌ 오류가 발생한 도구")
    for tool_exec in error_tools:
        tool_name = tool_exec.get("name", "unknown")
        emoji = get_tool_emoji(tool_name)
        with st.expander(f"{emoji} ❌ {tool_name} - 오류", expanded=True):
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
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Clear previous tool executions
    st.session_state.tool_executions = []
    
    # Initialize callback handler
    callback_handler = StreamlitAgentCallbackHandler()
    
    # Initialize agent
    try:
        agent_executor, is_mock = initialize_agent(use_mock=st.session_state.use_mock)
        
        if is_mock:
            st.info("⚠️ 개발 모드: Mock Agent를 사용 중입니다")
        
        # Display thinking process area
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            
            # Show thinking process
            with thinking_placeholder.container():
                display_thinking_process()
            
            # Execute agent
            with st.spinner("🤔 여행 계획 생성 중..."):
                try:
                    result = agent_executor.invoke(
                        {"input": user_query},
                        {"callbacks": [callback_handler]}
                    )
                    
                    # Update thinking process display
                    thinking_placeholder.empty()
                    with thinking_placeholder.container():
                        display_thinking_process()
                    
                    # Display response
                    response = result.get("output", "응답을 생성할 수 없습니다.")
                    st.markdown(response)
                    
                    # Save to chat history
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
    
    # Process pending query from button click
    if st.session_state.pending_query:
        pending = st.session_state.pending_query
        st.session_state.pending_query = None  # Clear pending query
        execute_agent_query(pending)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show tool executions if this is an assistant message
            if message["role"] == "assistant" and "tool_executions" in message:
                display_thinking_process()
    
    # Chat input
    if prompt := st.chat_input("여행 계획을 요청하세요..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        execute_agent_query(prompt)


if __name__ == "__main__":
    main()

여행 계획 AI 비서 - Main Application
"""

import os
from dotenv import load_dotenv
from agent_builder import build_travel_agent

# 환경 변수 로드 (.env 파일)
load_dotenv()

def main():
    # API 키 확인
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return
    if not os.environ.get("TAVILY_API_KEY"):
        print("❌ 오류: TAVILY_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return

    print("=" * 60)
    print("✈️  여행 계획 AI 비서 v2.0 시작")
    print("=" * 60)

    # 에이전트 빌드
    try:
        agent = build_travel_agent()
    except Exception as e:
        print(f"❌ 에이전트 초기화 실패: {e}")
        return

    # 사용자 입력 (MD 파일의 예시 사용)
    default_input = "3박 4일 제주도 여행 계획 짜줘, 예산 50만원, 해산물 좋아해, 1월 15일 출발"
    
    print("\n💡 예시 입력:")
    print(f'"{default_input}"')
    
    user_input = input("\n여행 요청사항을 입력하세요 (엔터 시 예시 실행): ").strip()
    
    if not user_input:
        user_input = default_input

    print(f"\n🔄 [진행 중] '{user_input}'에 대한 여행 계획을 생성하고 있습니다...\n")
    print("-" * 60)

    # 에이전트 실행
    try:
        result = agent.invoke({"input": user_input})
        
        print("\n" + "=" * 60)
        print("✅ [완료] 여행 계획 생성 결과")
        print("=" * 60 + "\n")
        print(result["output"])
        
        # 결과를 파일로 저장
        with open("result_plan.md", "w", encoding="utf-8") as f:
            f.write(result["output"])
        print("\n📄 결과가 'result_plan.md' 파일로 저장되었습니다.")

    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
