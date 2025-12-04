"""
여행 계획 AI 비서 - Tools 모듈
9개의 핵심 Tool을 정의합니다.
"""

from datetime import datetime
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import os

# Tavily Tool 설정 (검색 결과 5개)
tavily_tool = TavilySearchResults(
    max_results=5
)

@tool
def calculate_d_day(travel_date: str) -> dict:
    """
    여행 D-Day 및 준비 기간 계산

    Args:
        travel_date: 여행 출발 날짜 (형식: "YYYY-MM-DD")

    Returns:
        dict: D-Day 정보 (d_day, formatted, preparation)
    """
    try:
        today = datetime.now().date()
        target = datetime.strptime(travel_date, "%Y-%m-%d").date()
        delta = (target - today).days

        preparation_time = {
            "weeks": delta // 7,
            "urgency": "여유" if delta > 30 else "보통" if delta > 14 else "급함"
        }

        return {
            "d_day": delta,
            "formatted": f"D-{delta}" if delta > 0 else f"D+{abs(delta)}",
            "preparation": preparation_time,
            "date": travel_date
        }
    except Exception as e:
        return {"error": f"날짜 계산 오류: {str(e)}"}


@tool
def search_weather(location: str, date_range: str) -> dict:
    """
    특정 날짜의 날씨 예보 검색 (Tavily 사용)

    Args:
        location: 여행 지역 (예: "제주도", "오사카")
        date_range: 날짜 범위 (예: "2025년 1월 중순")

    Returns:
        dict: 날씨 정보 (temperature, precipitation, wind, warnings)
    """
    try:
        query = f"{location} {date_range} 날씨 예보 기온 강수량 바람"
        search_results = tavily_tool.invoke({"query": query})

        return {
            "location": location,
            "date_range": date_range,
            "search_results": search_results,
            "note": "이 데이터를 바탕으로 LLM이 날씨를 요약해야 합니다."
        }
    except Exception as e:
        return {"error": f"날씨 검색 오류: {str(e)}"}


@tool
def search_restaurants(location: str, food_type: str) -> dict:
    """
    맛집 정보 검색 (Tavily 사용)

    Args:
        location: 여행 지역
        food_type: 음식 종류 (예: "해산물", "라멘", "흑돼지")

    Returns:
        dict: 맛집 리스트
    """
    try:
        query = f"{location} {food_type} 맛집 추천 가격 메뉴 영업시간"
        search_results = tavily_tool.invoke({"query": query})

        return {
            "location": location,
            "food_type": food_type,
            "search_results": search_results
        }
    except Exception as e:
        return {"error": f"맛집 검색 오류: {str(e)}"}


@tool
def search_attractions(location: str, transportation: str = "대중교통") -> dict:
    """
    관광 명소 검색 (Tavily 사용)

    Args:
        location: 여행 지역
        transportation: 이동 수단 (예: "대중교통", "렌트카")

    Returns:
        dict: 명소 리스트
    """
    try:
        query = f"{location} {transportation}으로 갈 수 있는 관광 명소 추천 코스"
        search_results = tavily_tool.invoke({"query": query})

        return {
            "location": location,
            "transportation": transportation,
            "search_results": search_results
        }
    except Exception as e:
        return {"error": f"명소 검색 오류: {str(e)}"}


@tool
def search_transportation(location: str, transport_type: str = "버스") -> dict:
    """
    교통편 정보 검색 (Tavily 사용)

    Args:
        location: 여행 지역
        transport_type: 교통 수단 (예: "버스", "택시", "지하철")

    Returns:
        dict: 교통 정보 (요금, 노선, 패스 정보)
    """
    try:
        query = f"{location} {transport_type} 요금 노선 교통패스 꿀팁"
        search_results = tavily_tool.invoke({"query": query})

        return {
            "location": location,
            "transport_type": transport_type,
            "search_results": search_results
        }
    except Exception as e:
        return {"error": f"교통편 검색 오류: {str(e)}"}


@tool
def search_safety_warnings(location: str, season: str) -> dict:
    """
    현지 위험 요소 및 안전 정보 검색 (NEW)

    Args:
        location: 여행 지역
        season: 계절/월 (예: "1월", "겨울")

    Returns:
        dict: 안전 정보 (위험 요소, 비상 연락처, 대처법)
    """
    try:
        query = f"{location} {season} 여행 위험 요소 주의사항 치안 날씨위험"
        search_results = tavily_tool.invoke({"query": query})

        # 비상 연락처 검색
        emergency_query = f"{location} 여행 비상 연락처 병원 경찰서"
        emergency_results = tavily_tool.invoke({"query": emergency_query})

        return {
            "location": location,
            "season": season,
            "safety_results": search_results,
            "emergency_contacts": emergency_results
        }
    except Exception as e:
        return {"error": f"안전 정보 검색 오류: {str(e)}"}


@tool
def convert_currency_budget(budget: float, from_currency: str = "KRW", to_currency: str = "KRW") -> dict:
    """
    환율 기반 예산 환산 (NEW)

    Args:
        budget: 총 예산
        from_currency: 보유 통화 (예: "KRW")
        to_currency: 현지 통화 (예: "JPY", "USD")

    Returns:
        dict: 환산된 예산 및 수수료 정보
    """
    try:
        # 국내 여행 등 동일 통화일 경우
        if from_currency == to_currency:
            return {
                "original": budget,
                "converted": budget,
                "currency": from_currency,
                "rate": 1.0,
                "fees": 0,
                "note": "국내 여행 - 환전 불필요"
            }

        # 간단한 고정 환율 예시 (실제로는 실시간 API 연동 권장)
        exchange_rates = {
            ("KRW", "JPY"): 0.11,  # 1원 -> 0.11엔 (가정)
            ("KRW", "USD"): 0.00075, # 1원 -> 0.00075달러
            ("JPY", "KRW"): 9.1,
            ("USD", "KRW"): 1330,
        }

        # 키가 없으면 1.0으로 가정하거나 에러 처리
        rate = exchange_rates.get((from_currency, to_currency))
        
        if rate is None:
             # 역방향 계산 시도 혹은 기본값
             rate = 1.0
             
        converted = budget * rate
        fees = converted * 0.02  # 환전 수수료 2% 가정

        return {
            "original": budget,
            "converted": round(converted, 2),
            "rate": rate,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "fees": round(fees, 2),
            "total_needed": round(converted + fees, 2),
            "note": "예상 환율 적용 (수수료 2% 포함)"
        }
    except Exception as e:
        return {"error": f"환율 계산 오류: {str(e)}"}


@tool
def generate_weather_based_schedule(weather_summary: str, preferred_activities: str) -> dict:
    """
    날씨 기반 일정 최적화 가이드 제공 (NEW)
    이 툴은 실제 계산보다는 LLM에게 최적화 로직을 주입하기 위한 헬퍼입니다.

    Args:
        weather_summary: 날씨 요약 (예: "1일차 비, 2일차 맑음")
        preferred_activities: 선호 활동

    Returns:
        dict: 최적화 규칙
    """
    return {
        "optimization_rules": [
            "맑은 날: 야외 활동(등산, 오름, 해변) 우선 배정",
            "비/흐림: 실내 활동(박물관, 시장, 카페) 우선 배정",
            "강풍: 해안가 및 산행 주의, 실내 대안(Plan B) 마련",
            "동선 최적화: 가까운 지역끼리 묶어서 이동 시간 최소화"
        ],
        "input_context": {
            "weather": weather_summary,
            "activities": preferred_activities
        },
        "instruction": "위 규칙에 따라 일정을 재배치하고, 궃은 날씨에는 반드시 Plan B를 제시하세요."
    }


@tool
def calculate_budget_allocation(total_budget: float, duration_days: int, trip_type: str = "국내") -> dict:
    """
    스마트 예산 배분 계산

    Args:
        total_budget: 총 예산
        duration_days: 여행 일수
        trip_type: 여행 타입 ("국내", "해외")

    Returns:
        dict: 카테고리별 예산 배분
    """
    try:
        # 기본 배분 비율
        allocation = {
            "숙박": 0.30,
            "식비": 0.25,
            "교통": 0.15,
            "관광": 0.20,
            "쇼핑": 0.05,
            "기타": 0.05
        }

        if trip_type == "해외":
            allocation["교통"] = 0.25 # 항공권 등
            allocation["식비"] = 0.20
            allocation["관광"] = 0.15

        budget_breakdown = {}
        for category, ratio in allocation.items():
            amount = int(total_budget * ratio)
            budget_breakdown[category] = {
                "total": amount,
                "ratio": f"{int(ratio * 100)}%",
                "per_day": int(amount / duration_days) if duration_days > 0 else 0
            }

        return {
            "total_budget": total_budget,
            "duration_days": duration_days,
            "trip_type": trip_type,
            "breakdown": budget_breakdown
        }
    except Exception as e:
        return {"error": f"예산 배분 오류: {str(e)}"}


# Agent가 사용할 Tools 리스트
ALL_TOOLS = [
    calculate_d_day,
    search_weather,
    search_restaurants,
    search_attractions,
    search_transportation,
    search_safety_warnings,
    convert_currency_budget,
    generate_weather_based_schedule,
    calculate_budget_allocation
]