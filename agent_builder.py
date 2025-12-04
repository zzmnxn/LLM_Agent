"""
여행 계획 AI 비서 - Agent Builder
LangChain AgentExecutor를 생성합니다.
"""

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from tools import ALL_TOOLS
from prompts import get_agent_prompt
import os

def build_travel_agent(model_name="gpt-4o-mini"):
    """
    여행 계획 Agent를 생성하고 반환합니다.
    
    Args:
        model_name (str): 사용할 OpenAI 모델명 (기본값: gpt-4o-mini)
        
    Returns:
        AgentExecutor: 실행 가능한 에이전트 객체
    """
    # 1. LLM 초기화 (Tools 바인딩)
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # 2. Prompt 가져오기
    prompt = get_agent_prompt()
    
    # 3. Agent 생성 (OpenAI Tools Agent)
    agent = create_openai_tools_agent(llm, ALL_TOOLS, prompt)
    
    # 4. Executor 생성 (실행기)
    # verbose=True로 설정하면 터미널에서 생각 과정(CoT)을 볼 수 있습니다.
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=ALL_TOOLS, 
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor