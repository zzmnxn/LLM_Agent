"""
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