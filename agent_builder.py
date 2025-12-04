"""
Agent Builder Module
Team Member A가 만든 tools와 prompts를 사용하여 AgentExecutor를 생성합니다.
"""

from typing import List
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate


def build_agent_executor(
    tools: List[BaseTool],
    prompt: ChatPromptTemplate,
    llm: ChatOpenAI,
    verbose: bool = True
) -> AgentExecutor:
    """
    Tools와 Prompt를 사용하여 AgentExecutor를 생성합니다.
    
    Args:
        tools: LangChain BaseTool 리스트
        prompt: ChatPromptTemplate
        llm: ChatOpenAI 인스턴스
        verbose: AgentExecutor의 verbose 옵션 (기본값: True)
    
    Returns:
        AgentExecutor: 구성된 AgentExecutor 인스턴스
    """
    # Agent 생성
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # AgentExecutor 생성
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,  # 파싱 오류 처리
        max_iterations=15,  # 최대 반복 횟수
        max_execution_time=120  # 최대 실행 시간 (초)
    )
    
    return agent_executor


def get_agent_executor(
    tools: List[BaseTool],
    prompt: ChatPromptTemplate,
    llm: ChatOpenAI,
    verbose: bool = True
) -> AgentExecutor:
    """
    build_agent_executor의 별칭 함수 (호환성을 위해 제공)
    """
    return build_agent_executor(tools, prompt, llm, verbose)

