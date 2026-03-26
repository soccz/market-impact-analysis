"""
50건 정책 이벤트 주가 검증 스크립트
기준: 정책 발표일 또는 익영업일 기준, 최대 변동 종목/지수 절대값 2% 이상
"""

import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 50건 이벤트 정의
# ──────────────────────────────────────────────
EVENTS = [
    # 반도체 (대표주: 삼성전자 005930, SK하이닉스 000660)
    {"id": "반도체#1",  "date": "1999-05-20", "policy": "현대전자-LG반도체 빅딜",           "stocks": ["005930","000660"], "sector": "반도체", "verified": -3.57},
    {"id": "반도체#2",  "date": "2003-08-22", "policy": "차세대 성장동력 10대 산업 확정",   "stocks": ["005930","000660"], "sector": "반도체", "verified": -3.69},
    {"id": "반도체#3",  "date": "2019-07-01", "policy": "일본 수출규제 발동",               "stocks": ["005930","000660","000104"], "sector": "반도체"},
    {"id": "반도체#4",  "date": "2019-08-05", "policy": "소부장 경쟁력 강화 대책",          "stocks": ["005930","000660","000104"], "sector": "반도체"},
    {"id": "반도체#5",  "date": "2021-05-13", "policy": "K-반도체 벨트 전략",              "stocks": ["005930","000660","000104","042700"], "sector": "반도체"},
    {"id": "반도체#6",  "date": "2022-07-21", "policy": "반도체 초강대국 달성 전략",        "stocks": ["005930","000660"], "sector": "반도체", "verified": 2.15},
    {"id": "반도체#7",  "date": "2022-10-07", "policy": "미국 대중국 반도체 수출통제",      "stocks": ["005930","000660","000104"], "sector": "반도체", "보류": True},
    {"id": "반도체#8",  "date": "2023-03-22", "policy": "K칩스법 1차 (세액공제 8→15%)",   "stocks": ["005930","000660","042700","086390"], "sector": "반도체"},
    {"id": "반도체#9",  "date": "2024-01-15", "policy": "반도체 메가 클러스터",             "stocks": ["005930","000660","042700"], "sector": "반도체"},
    {"id": "반도체#10", "date": "2025-02-27", "policy": "K칩스법 2차 (세액공제 15→20%)",  "stocks": ["005930","000660","042700"], "sector": "반도체"},

    # 금융/증권 (대표주: 삼성증권 016360, 키움증권 039490, 미래에셋 006800)
    {"id": "금융#1",  "date": "1998-06-29", "policy": "금융 구조조정 5개 은행 퇴출",       "stocks": ["016360","039490","006800","105560"], "sector": "금융", "verified": -3.16},
    {"id": "금융#2",  "date": "2008-09-15", "policy": "리먼브라더스 파산",                 "stocks": ["016360","039490","006800","105560"], "sector": "금융", "verified": -9.80},
    {"id": "금융#3",  "date": "2008-10-01", "policy": "공매도 전면 금지①",                "stocks": ["016360","039490","006800","105560"], "sector": "금융", "보류": True},
    {"id": "금융#4",  "date": "2011-08-10", "policy": "공매도 일시 금지②",                "stocks": ["016360","039490","006800","105560"], "sector": "금융", "verified": -2.91},
    {"id": "금융#5",  "date": "2020-03-16", "policy": "공매도 전면 금지③ (코로나)",       "stocks": ["016360","039490","006800","105560"], "sector": "금융"},
    {"id": "금융#6",  "date": "2020-03-24", "policy": "코로나 금융안정 패키지 100조원",    "stocks": ["016360","039490","006800","105560"], "sector": "금융"},
    {"id": "금융#7",  "date": "2020-06-25", "policy": "증권거래세 단계적 인하",            "stocks": ["016360","039490","006800"], "sector": "금융"},
    {"id": "금융#8",  "date": "2023-11-06", "policy": "공매도 재금지④",                   "stocks": ["016360","039490","006800","105560"], "sector": "금융"},
    {"id": "금융#9",  "date": "2024-01-17", "policy": "기업 밸류업 프로그램",              "stocks": ["016360","039490","006800","105560"], "sector": "금융"},
    {"id": "금융#10", "date": "2024-11-04", "policy": "금투세 폐지 여야 합의",             "stocks": ["016360","039490","006800","105560"], "sector": "금융"},

    # 부동산/건설 (대표주: 현대건설 000720, GS건설 006360, DL이앤씨 000215)
    {"id": "부동산#1",  "date": "1998-05-22", "policy": "IMF 직후 주택경기 활성화 대책",   "stocks": ["000720","006360"], "sector": "건설", "verified": -2.99},
    {"id": "부동산#2",  "date": "2003-10-29", "policy": "10.29 주택시장 안정 종합대책",    "stocks": ["000720","006360","000215"], "sector": "건설", "verified": -5.01},
    {"id": "부동산#3",  "date": "2005-08-31", "policy": "8.31 부동산 종합대책",            "stocks": ["000720","006360","000215"], "sector": "건설", "verified": 4.00},
    {"id": "부동산#4",  "date": "2008-09-01", "policy": "부동산 세제 완화 대책",           "stocks": ["000720","006360","000215"], "sector": "건설", "verified": -8.35},
    {"id": "부동산#5",  "date": "2008-09-19", "policy": "보금자리주택 건설 방안",          "stocks": ["000720","006360","000215"], "sector": "건설", "verified": 6.91},
    {"id": "부동산#6",  "date": "2013-04-01", "policy": "4.1 부동산 종합대책",            "stocks": ["000720","006360","000215","047040","084690"], "sector": "건설", "verified": -4.02},
    {"id": "부동산#7",  "date": "2017-06-19", "policy": "6.19 주택시장 안정화 대책",       "stocks": ["000720","006360","000215","047040"], "sector": "건설"},
    {"id": "부동산#8",  "date": "2020-06-17", "policy": "6.17 부동산 대책",               "stocks": ["000720","006360","000215","047040"], "sector": "건설", "보류": True},
    {"id": "부동산#9",  "date": "2022-06-21", "policy": "윤석열 정부 규제 완화 패키지",    "stocks": ["000720","006360","000215","047040","084690","294870"], "sector": "건설"},
    {"id": "부동산#10", "date": "2024-01-10", "policy": "1.10 주택공급 대책",             "stocks": ["000720","006360","000215","047040","084690"], "sector": "건설"},

    # 바이오/제약 (대표주: 삼성바이오 207940, 셀트리온 068270, 한미약품 128940)
    {"id": "바이오#1",  "date": "2000-07-01", "policy": "의약분업 시행",                  "stocks": ["068270","128940","069620","000661"], "sector": "바이오", "verified": -5.45},
    {"id": "바이오#2",  "date": "2005-01-12", "policy": "황우석팀 줄기세포 정부 승인",     "stocks": ["068270","128940","069620","000661"], "sector": "바이오", "보류": True},
    {"id": "바이오#3",  "date": "2005-12-15", "policy": "황우석 논문 조작 폭로",           "stocks": ["068270","128940","069620","000661"], "sector": "바이오", "verified": -8.13},
    {"id": "바이오#4",  "date": "2009-06-30", "policy": "바이오시밀러 허가심사 가이드라인","stocks": ["068270","128940","069620"], "sector": "바이오", "verified": 3.70},
    {"id": "바이오#5",  "date": "2010-11-29", "policy": "리베이트 쌍벌제 시행 (11/28 휴일→29적용)", "stocks": ["068270","128940","069620"], "sector": "바이오", "verified": 5.26},
    {"id": "바이오#6",  "date": "2012-07-23", "policy": "셀트리온 램시마 세계최초 허가",   "stocks": ["068270","128940","207940"], "sector": "바이오", "verified": -2.30},
    {"id": "바이오#7",  "date": "2019-05-22", "policy": "바이오헬스 육성 전략",            "stocks": ["068270","128940","207940","086900","009420"], "sector": "바이오"},
    {"id": "바이오#8",  "date": "2020-12-08", "policy": "코로나19 백신 수급 계획",         "stocks": ["068270","128940","207940","086900","302440"], "sector": "바이오"},
    {"id": "바이오#9",  "date": "2024-02-06", "policy": "의대정원 2000명 증원 발표",       "stocks": ["068270","128940","207940","009420","000661"], "sector": "바이오"},
    {"id": "바이오#10", "date": "2025-12-30", "policy": "CDMO 특별법 시행",               "stocks": ["068270","128940","207940","000661"], "sector": "바이오", "verified": 6.50},

    # 2차전지/에너지 (대표주: LG에너지 373220, 삼성SDI 006400, 에코프로비엠 247540)
    {"id": "2차전지#1",  "date": "2009-10-06", "policy": "녹색성장 5개년 계획",            "stocks": ["006400","051910","009830","010060"], "sector": "2차전지", "verified": -5.57},
    {"id": "2차전지#2",  "date": "2013-06-25", "policy": "전기차 민간 보급 보조금 최초",   "stocks": ["006400","051910","247540","373220"], "sector": "2차전지", "verified": -2.90},
    {"id": "2차전지#3",  "date": "2017-09-11", "policy": "재생에너지 3020 이행계획",       "stocks": ["006400","051910","247540"], "sector": "2차전지", "verified": 5.23},
    {"id": "2차전지#4",  "date": "2019-06-11", "policy": "ESS 화재 안전강화대책",          "stocks": ["006400","051910","247540","373220"], "sector": "2차전지", "보류": True},
    {"id": "2차전지#5",  "date": "2020-07-14", "policy": "한국판 뉴딜 그린뉴딜 73.4조원", "stocks": ["006400","051910","247540","373220","009830"], "sector": "2차전지"},
    {"id": "2차전지#6",  "date": "2021-07-14", "policy": "EU CBAM + 배터리 규정안",        "stocks": ["006400","051910","247540","373220","086520"], "sector": "2차전지"},
    {"id": "2차전지#7",  "date": "2022-08-16", "policy": "미국 IRA 서명·발효",             "stocks": ["006400","051910","247540","373220","086520"], "sector": "2차전지"},
    {"id": "2차전지#8",  "date": "2022-11-01", "policy": "이차전지 산업 혁신전략",         "stocks": ["006400","051910","247540","373220"], "sector": "2차전지", "verified": 11.02},
    {"id": "2차전지#9",  "date": "2023-04-20", "policy": "첨단전략산업 지정 + 세액공제",   "stocks": ["006400","051910","247540","373220","086520"], "sector": "2차전지"},
    {"id": "2차전지#10", "date": "2025-05-22", "policy": "트럼프 IRA 전기차 세액공제 축소","stocks": ["006400","051910","247540","373220","086520"], "sector": "2차전지"},
]

THRESHOLD = 2.0  # 2% 기준

# 통일 기준:
# - 발표일(D+0) 1차 확인
# - D+0 < 2% 이면 익영업일(D+1) 확인 (장 마감 후 발표 등 고려)
# - D+1 까지만 허용 (D+2 이상은 포함하지 않음)
# - 여러 종목 중 절대값 최대를 선택 (단, 어떤 종목인지 기록)

def get_change(code, date_str):
    """발표일(D+0) 및 익영업일(D+1) 중 절대값 최대 변동률 반환"""
    try:
        start = pd.Timestamp(date_str)
        # D+0, D+1만 포함하도록 약 5일치 데이터만 가져옴
        end = start + pd.Timedelta(days=7)
        df = fdr.DataReader(code, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        if df is None or len(df) == 0:
            return None, None
        # D+0, D+1 (2거래일) 만
        rows = df[df.index >= pd.Timestamp(date_str)].head(2)
        if len(rows) == 0:
            return None, None
        best_idx = rows['Change'].abs().idxmax()
        best_chg = rows.loc[best_idx, 'Change'] * 100
        return round(best_chg, 2), str(best_idx.date())
    except Exception:
        return None, None

def verify_event(event):
    if event.get("보류"):
        return {
            "id": event["id"],
            "date": event["date"],
            "policy": event["policy"],
            "status": "🔶 보류",
            "best_stock": "-",
            "best_change": 0.0,
            "actual_date": "-",
            "note": "보류 - 종목/날짜/정책 교체 검토 필요"
        }
    if "verified" in event:
        return {
            "id": event["id"],
            "date": event["date"],
            "policy": event["policy"],
            "status": "✅ 기확정",
            "best_stock": "-",
            "best_change": event["verified"],
            "actual_date": event["date"],
            "note": "기존 검증 완료"
        }

    best_chg = 0
    best_stock = None
    best_date = None

    for code in event["stocks"]:
        chg, act_date = get_change(code, event["date"])
        if chg is not None and abs(chg) > abs(best_chg):
            best_chg = chg
            best_stock = code
            best_date = act_date

    if best_stock is None:
        status = "⚠️ 데이터없음"
        note = "FDR 범위 외 (pre-2014) - 웹검색 필요"
    elif abs(best_chg) >= THRESHOLD:
        status = "✅ 통과"
        note = f"2% 기준 충족"
    else:
        status = "❌ 미달"
        note = f"최대 {abs(best_chg):.2f}% - 종목/정책 교체 검토"

    return {
        "id": event["id"],
        "date": event["date"],
        "policy": event["policy"],
        "status": status,
        "best_stock": best_stock or "-",
        "best_change": best_chg,
        "actual_date": best_date or "-",
        "note": note
    }

if __name__ == "__main__":
    print("=" * 80)
    print("50건 정책 이벤트 주가 검증")
    print(f"기준: 발표일 또는 익영업일 기준 절대값 {THRESHOLD}% 이상")
    print("=" * 80)

    results = []
    for event in EVENTS:
        r = verify_event(event)
        results.append(r)
        symbol = r["status"].split()[0]
        print(f"{r['id']:<12} {symbol}  {r['best_change']:>7.2f}%  {r['actual_date']:<12} {r['policy'][:35]}")

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("결과 요약")
    print("=" * 80)
    confirmed = df[df["status"].str.contains("✅")]
    failed = df[df["status"].str.contains("❌")]
    no_data = df[df["status"].str.contains("⚠️")]
    pending = df[df["status"].str.contains("🔶")]

    print(f"✅ 통과/확정:  {len(confirmed)}건")
    print(f"🔶 보류:       {len(pending)}건  → 종목/날짜/정책 교체 검토 필요")
    print(f"❌ 미달:       {len(failed)}건")
    print(f"⚠️ 데이터없음: {len(no_data)}건")

    if len(pending) > 0:
        print(f"\n[보류 목록]")
        for _, row in pending.iterrows():
            print(f"  {row['id']}: {row['policy'][:40]}")

    if len(failed) > 0:
        print(f"\n[미달 목록]")
        for _, row in failed.iterrows():
            print(f"  {row['id']}: {row['policy'][:40]} ({row['best_change']:.2f}%)")

    if len(no_data) > 0:
        print(f"\n[데이터없음 목록]")
        for _, row in no_data.iterrows():
            print(f"  {row['id']}: {row['date']} {row['policy'][:40]}")

    # CSV 저장
    df.to_csv("verification_results.csv", index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: verification_results.csv")
