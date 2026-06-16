import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
# 기존에 사용하시던 스크립트 임포트 (경로에 맞춰 유지)
from scripts.processor import (
    get_account_name, get_active_ad_count, get_total_content_count,
    get_ad_period, get_content_period, get_total_keyword_count,
    get_instagram_followers, get_ctr_data, get_ctr_monthly_data, get_organic_data, get_organic_monthly_data, get_imp_threshold,
    get_content_ctr_data, get_a_content_target_ctr_data, get_profile_visits_monthly, get_content_reaction_data, get_reaction_metric_avg,
    get_target_avg_imp_ctr, get_target_avg_imp_ctr_threshold,
    get_raw_keyword_performance, filter_keywords_by_pos, get_overall_ctr,
    get_strategic_performance,get_essence_target_performance,get_variable_target_performance,
    has_purchase_data, get_purchase_roas_weekly, get_purchase_roas_monthly,  # ROAS,구매건수 데이터 추가
    get_purchase_count_weekly, get_purchase_count_monthly,
    has_purchase_content_data, get_purchase_contents_pages_data, get_a_content_target_purchase_data, get_purchase_age_gender_heatmap,get_purchase_age_gender_heatmap_page_data,  # 구매 컨텐츠 추가
    has_revenue_data, get_spend_and_revenue_weekly, get_spend_and_revenue_monthly,  # 광고/매출금액 추가
    has_follower_demographics_data, get_follower_demographics_latest_date, get_demographics_ratio, get_follower_age_gender_known_only, get_age_known_unknown_by_age, get_follower_age_gender_distribution,  # 팔로워 인구통계 추가
    get_target_spend_distribution, 
    get_ctr_follows_scatter_data, get_prev_quarter_ctr_follows_means,
    get_prev_quarter_organic_avg, get_prev_quarter_profile_visits_avg, get_prev_quarter_ctr_avg, get_quarter_info,
)

import re

def _parse_korean_caption(caption: str, max_chars: int = 7) -> str:
    """
    캡션 문자열에서 한글이 처음 등장하는 위치를 찾아,
    그 위치부터 max_chars(기본 7)글자를 추출하여 반환한다.

    - 한글이 없으면 원문 앞 max_chars 글자를 그대로 사용한다.
    - 추출 결과가 max_chars를 초과하면 '...'을 붙인다.
    - 추출 결과가 max_chars 이하면 말줄임 없이 반환한다.
    - caption이 None 또는 빈 문자열이면 빈 문자열을 반환한다.
    """
    if not caption or not caption.strip():
        return ""

    caption = caption.strip()
    
    # 한글 유니코드 범위: 가-힣 (완성형 한글)
    match = re.search(r'[가-힣]', caption)

    if match:
        # 한글이 처음 등장하는 인덱스부터 슬라이싱
        start_idx = match.start()
        text = caption[start_idx:]
    else:
        # 한글이 없으면 처음부터 사용
        text = caption

    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text

def _assign_reaction_ranks(items, metric_col):
    """
    items 리스트에 동순위를 반영한 'rank' 필드를 추가하고,
    13개를 초과하는 경우 업로드일 최신순으로 잘라냅니다.
    """
    if not items:
        return items

    # rank 계산: 같은 지표값이면 같은 순위 번호
    prev_val = None
    prev_rank = 0
    count = 0
    for item in items:
        val = float(item.get(metric_col, 0) or 0)
        count += 1
        if val != prev_val:
            prev_rank = count
            prev_val = val
        item["rank"] = prev_rank

    # 13개 초과 시: 11위 이상(rank >= 11)인 항목 중 업로드일 최신순 3개만 남김
    if len(items) > 13:
        top10 = [x for x in items if x["rank"] <= 10]
        rest  = [x for x in items if x["rank"] > 10]
        rest_sorted = sorted(rest, key=lambda x: x.get("uploaded_at") or "", reverse=True)
        items = top10 + rest_sorted[:3]

    return items


def run(target_id, fb_ad_account_id, start, end, main_age="", main_gender="", avoid_age="", avoid_gender="", currency=""):
    # 1. 기본 설정 및 파라미터

    # 실제 집계 마지막 날: date_end가 속한 주의 직전 일요일
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    actual_end = (end_dt - timedelta(days=end_dt.weekday())).strftime("%Y-%m-%d")

    # 평균선 라벨 통일용: 현재 기간(end 기준)이 속한 분기 정보
    cur_q_year, cur_q_quarter = get_quarter_info(end)
    current_quarter_info = {"year": cur_q_year, "quarter": cur_q_quarter}

    acc_name = get_account_name(target_id)
    ad_start, ad_end = get_ad_period(target_id, start, end)
    content_start, content_end = get_content_period(target_id, start, end)

    # 통화 설정 추가
    currency_symbol = "$" if currency == "dollar" else "원"

    # 2. 결과 저장용 구조 (핵심)
    final_report = {
        "meta": {
            "account_name": acc_name,
            "period": f"{start} ~ {actual_end}",
            "period_ads": f"{pd.Timestamp(ad_start).strftime('%Y-%m-%d') if ad_start else '-'} ~ {pd.Timestamp(ad_end).strftime('%Y-%m-%d') if ad_end else '-'}",
            "period_contents": f"{content_start} ~ {content_end}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },

        # 통화 설정
        "currency": currency,
        "currency_symbol": currency_symbol,

        "summary": {
            "total_ads": get_active_ad_count(target_id, start, end),
            "total_contents": get_total_content_count(target_id, start, end),
            "total_keywords": get_total_keyword_count(target_id, start, end)
        },
        "datasets": {}
    }

    # 데이터셋 추가를 위한 헬퍼 함수
    def add_ds(key, kind, title, df, unit="", x=None, ys=None, extra_meta=None):
        if df is None or df.empty: return
        
        # [수정포인트 1] 모든 함수의 인덱스 문제를 한 번에 해결
        df_c = df.copy().reset_index() if df.index.name else df.copy()
        columns = list(df_c.columns)
        
        # [수정포인트 2] X축 매칭 로직 개선 (테이블일 때는 X축을 억지로 잡지 않음)
        if kind != "table" and x not in columns:
            possible_x = [c for c in columns if any(word in c.lower() for word in ['date', 'at', 'start', 'week', 'time'])]
            x = possible_x[0] if possible_x else columns[0]

        # 날짜 변환 (x가 컬럼에 있고, 실제 날짜 성격일 때만 실행)
        if x in columns and ('date' in x.lower() or 'time' in x.lower() or 'at' in x.lower()):
            try:
                df_c[x] = pd.to_datetime(df_c[x], errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')
            except:
                pass

        data_obj = {"kind": kind, "title": title, "unit": unit}
        if extra_meta: data_obj.update(extra_meta)

        if kind == "table":
            # 테이블은 모든 컬럼을 보존하여 rows에 담음
            data_obj["rows"] = df_c.replace({pd.NA: None, pd.NaT: None, np.nan: None}).to_dict(orient='records')
        else:
            # 차트(bar, line 등)일 때만 x를 분리하여 labels로 사용
            data_obj["labels"] = df_c[x].fillna("Unknown").tolist()
            
            series_data = []
            # 지정된 Y축이 있으면 그것만, 없으면 숫자 컬럼 전체를 가져옴
            target_ys = ys if ys else df_c.select_dtypes(include=['number']).columns.tolist()
            
            for y_req in target_ys:
                matched_col = next((c for c in columns if y_req.lower() == c.lower() or y_req.lower() in c.lower()), None)
                if matched_col and matched_col != x: # X축과 중복 방지
                    clean_data = df_c[matched_col].fillna(0).tolist()
                    series_data.append({"name": matched_col, "data": clean_data})
            
            data_obj["series"] = series_data
            
        final_report["datasets"][key] = data_obj


    # --- 데이터 수집 및 변환 시작 ---

    # 1. 인스타그램 및 오가닉 추이
    print("인스타그램 및 오가닉 추이 생성 중...")
    insta_df = get_instagram_followers(fb_ad_account_id, start, end)

    # 'date' -> 'updated_at'으로 수정
    add_ds("insta_followers", "line", "팔로워 추이", insta_df, "명", "updated_at", ["follower_count"])
    
    # 주별
    prev_q_profile_visits = get_prev_quarter_profile_visits_avg(fb_ad_account_id, end)
    profile_visits_meta = {"current_quarter": current_quarter_info}
    if prev_q_profile_visits:
        profile_visits_meta["prev_quarter_avg"] = prev_q_profile_visits
    add_ds(
        "insta_profile_visits", "line", "프로필 방문 수(주별)", insta_df, "회", "updated_at", ["profile_views"],
        extra_meta=profile_visits_meta
    )

    # 월별 프로필 방문수 데이터 로드
    profile_monthly_df = get_profile_visits_monthly(fb_ad_account_id, start, end)

    add_ds(
        "insta_profile_visits_monthly",
        "line",                   # 방문수는 막대 그래프가 보기 편합니다
        "인스타그램 프로필 방문수 (월별)", 
        profile_monthly_df, 
        "회", 
        "updated_at", 
        ["profile_views"]
    )
    
    organic_df = get_organic_data(target_id, start, end)  # (주별) 추가
    prev_q_organic = get_prev_quarter_organic_avg(target_id, end)
    organic_meta = {"current_quarter": current_quarter_info}
    if prev_q_organic:
        organic_meta["prev_quarter_avg"] = prev_q_organic
    add_ds(
        "organic_trend", "line", "오가닉 조회수 추이 (주별)", organic_df, "회", "date_start", ["organic_impressions"],
        extra_meta=organic_meta
    )

    # 4주 단위 월별 데이터 바로 가져오기
    organic_monthly_df = get_organic_monthly_data(target_id, start, end)

    add_ds(
        "organic_trend_monthly", 
        "line", 
        "오가닉 조회수 추이 (월별)", 
        organic_monthly_df, 
        "회", 
        "date_start", 
        ["organic_impressions"]
    )


    # --- [추가] 팔로워 인구통계학 페이지 ---
    datasets = final_report["datasets"]

    followers_df = get_instagram_followers(fb_ad_account_id, start, end)
    current_followers = None
    if followers_df is not None and not followers_df.empty:
        follower_series = followers_df["follower_count"].dropna()
        if not follower_series.empty:
            current_followers = int(follower_series.iloc[-1])

    if has_follower_demographics_data(target_id, start, end):
        print("팔로워 인구통계학 데이터 생성 중...")

        gender_clean_df = get_demographics_ratio(target_id, start, end, "gender", "exclude_unknown")
        age_gender_clean_df = get_follower_age_gender_known_only(target_id, start, end)
        gender_unknown_df = get_demographics_ratio(target_id, start, end, "gender", "unknown_vs_known")
        age_known_unknown_df = get_age_known_unknown_by_age(target_id, start, end)

        follower_demo_latest_date = get_follower_demographics_latest_date(target_id, start, end)
        age_gender_distribution_df = get_follower_age_gender_distribution(target_id, start, end)

        # 좌상: 성별 비율 (알 수 없음 제외) + 가운데 현재 팔로워 수
        if gender_clean_df is not None:
            datasets["gender_clean"] = {
                "chart_type": "doughnut",
                "title": "팔로워 성별·연령 구성",
                "labels": gender_clean_df["category"].astype(str).tolist(),
                "series": [
                    {"name": "비율", "data": gender_clean_df["ratio"].astype(float).tolist()}
                ],
                "unit": "%",
                "center_text": f"{current_followers:,}" if current_followers is not None else None,
                "center_subtext": "팔로워"
            }

        # 좌하: 연령대별 성별 분포 (알 수 없음 제외)
        if age_gender_clean_df is not None:
            datasets["age_gender_clean"] = {
                "chart_type": "stacked_barh",
                "title": "연령대별 성별 분포 (알 수 없음 제외)",
                "labels": age_gender_clean_df["age_range"].astype(str).tolist(),
                "series": [
                    {"name": "male", "data": age_gender_clean_df["male"].astype(float).tolist()},
                    {"name": "female", "data": age_gender_clean_df["female"].astype(float).tolist()}
                ],
                "unit": "명"
            }
        
        if age_gender_distribution_df is not None:
            datasets["age_gender_distribution"] = {
                "chart_type": "stacked_barh",
                "title": "연령대별 팔로워 분포(성별 구성)",
                "labels": age_gender_distribution_df["age_range"].astype(str).tolist(),
                "series": [
                    {"name": "male", "data": age_gender_distribution_df["male"].astype(float).tolist()},
                    {"name": "female", "data": age_gender_distribution_df["female"].astype(float).tolist()},
                ],
                "unit": "명"
            }

        # 우상: 성별 데이터 식별 여부 + 가운데 unknown 비율
        if gender_unknown_df is not None:
            unknown_ratio = None
            unknown_row = gender_unknown_df[
                gender_unknown_df["category"].isin(["알 수 없음", "Unknown"])
            ]
            if not unknown_row.empty:
                unknown_ratio = float(unknown_row["ratio"].iloc[0])

            datasets["gender_unknown"] = {
                "chart_type": "doughnut",
                "title": "성별 데이터 식별 여부",
                "labels": gender_unknown_df["category"].astype(str).tolist(),
                "series": [
                    {"name": "비율", "data": gender_unknown_df["ratio"].astype(float).tolist()}
                ],
                "unit": "%",
                "center_text": f"{unknown_ratio:.1f}%" if unknown_ratio is not None else None,
                "center_subtext": "알 수 없음 비율"
            }

        # 우하: 연령 데이터 식별 여부 분포
        if age_known_unknown_df is not None:
            datasets["age_known_unknown"] = {
                "chart_type": "stacked_barh",
                "title": "연령대별 성별 데이터 식별 여부 분포",
                "labels": age_known_unknown_df["age_range"].astype(str).tolist(),
                "series": [
                    {"name": "known", "data": age_known_unknown_df["known"].astype(float).tolist()},
                    {"name": "unknown", "data": age_known_unknown_df["unknown"].astype(float).tolist()}
                ],
                "unit": "명"
            }

        final_report["follower_demographics_pages"] = {
            "is_visible": True,
            "latest_date": follower_demo_latest_date,
            "titles": {
                "section_title": "팔로워 인구통계학 분석",
                "page_1_title": "성별 및 연령대별 팔로워 분포"
            }
        }

    else:
        print("팔로워 인구통계학 데이터 없음...")
        final_report["follower_demographics_pages"] = {
            "is_visible": False
        }

    # 2. CTR 추이
    print("CTR 추이 생성 중...")
    ctr_weekly_df = get_ctr_data(target_id, start, end)
    prev_q_ctr = get_prev_quarter_ctr_avg(target_id, end)
    ctr_meta = {"current_quarter": current_quarter_info}
    if prev_q_ctr:
        ctr_meta["prev_quarter_avg"] = prev_q_ctr
    add_ds(
        "ctr_trend_weekly", "line", "주별 CTR 추이", ctr_weekly_df, "%", "week_start", ["ctr"],
        extra_meta=ctr_meta
    )
    ctr_monthly_df = get_ctr_monthly_data(target_id, start, end)
    add_ds("ctr_trend_monthly", "line", "월별 CTR 추이", ctr_monthly_df, "%", "month_start", ["ctr"])

    #  --- [추가] ROAS, 구매건수 (2페이지 분량) ---

    if has_purchase_data(target_id, start, end):
        print("ROAS, 구매건수 생성 중...")
        roas_weekly_df = get_purchase_roas_weekly(target_id, start, end)
        roas_monthly_df = get_purchase_roas_monthly(target_id, start, end)
        purchase_weekly_df = get_purchase_count_weekly(target_id, start, end)
        purchase_monthly_df = get_purchase_count_monthly(target_id, start, end)

        add_ds("purchase_roas_weekly", "line", "평균 ROAS (주별)", roas_weekly_df, "%", "week_start", ["avg_roas"])
        add_ds("purchase_roas_monthly", "line", "평균 ROAS (월별)", roas_monthly_df, "%", "month_start", ["avg_roas"])
        add_ds("purchase_count_weekly", "line", "구매전환 (주별)", purchase_weekly_df, "건", "week_start", ["purchases"])
        add_ds("purchase_count_monthly", "line", "구매전환 (월별)", purchase_monthly_df, "건", "month_start", ["purchases"])

        final_report["purchase_analysis_pages"] = {
            "is_visible": True,
            "titles": {
                "section_title": "전체 매출 데이터 분석",
                "page_1_title": "평균 ROAS",
                "page_2_title": "구매전환 건수"
            }
        }
    else:
        print("ROAS, 구매건수 없음...")
        final_report["purchase_analysis_pages"] = {
            "is_visible": False,
            "titles": {
                "section_title": "전체 매출 데이터 분석",
                "page_1_title": "평균 ROAS",
                "page_2_title": "구매전환 건수"
            }
        }

    # --- [추가] 광고비 & 매출발생 페이지 ---
    if has_revenue_data(target_id, start, end):
        print("광고비/매출발생 데이터 생성 중...")
        spend_revenue_weekly_df = get_spend_and_revenue_weekly(target_id, start, end)
        spend_revenue_monthly_df = get_spend_and_revenue_monthly(target_id, start, end)

        add_ds(
            "spend_revenue_weekly",
            "line",
            "광고비 & 매출발생 (주별)",
            spend_revenue_weekly_df,
            currency_symbol,
            "week_start",
            ["spend", "revenue"],
            extra_meta={"show_legend": True}
        )

        add_ds(
            "spend_revenue_monthly",
            "line",
            "광고비 & 매출발생 (월별)",
            spend_revenue_monthly_df,
            currency_symbol,
            "month_start",
            ["spend", "revenue"],
            extra_meta={"show_legend": True}
        )

        final_report["spend_revenue_pages"] = {
            "is_visible": True,
            "titles": {
                "section_title": "전체 매출 데이터 분석",
                "page_1_title": "광고비 & 매출 발생"
            }
        }
    else:
        print("광고비/매출발생 데이터 없음...")
        final_report["spend_revenue_pages"] = {
            "is_visible": False
        }

     #  --- [추가] 구매 발생 컨텐츠  --
    purchase_contents_data = get_purchase_contents_pages_data(target_id, start, end)

    if purchase_contents_data and purchase_contents_data.get("total_count", 0) > 0:
        print("구매 발생 콘텐츠 생성 중...")

        enriched_pages = []
        for page_items in purchase_contents_data["pages"]:
            enriched_items = []

            for item in page_items:
                detail_df = get_a_content_target_purchase_data(item["ad_ids"], start, end)
                if detail_df is not None:
                    item["target_details"] = detail_df.to_dict(orient="records")
                else:
                    item["target_details"] = []

                enriched_items.append(item)

            enriched_pages.append(enriched_items)

        final_report["purchase_contents_pages"] = {
            "is_visible": True,
            "title": purchase_contents_data["title"],
            "pages": enriched_pages,
            "total_count": purchase_contents_data["total_count"]
        }
    else:
        print("구매 발생 콘텐츠 없음...")
        final_report["purchase_contents_pages"] = {
            "is_visible": False
        }

    _, threshold = get_imp_threshold(target_id, start, end)

    # ================================
    # 구매 전환 히트맵 페이지 추가
    # ================================
    purchase_age_gender_data = get_purchase_age_gender_heatmap_page_data(target_id, start, end)

    if purchase_age_gender_data:
        heatmap_rows = purchase_age_gender_data.get("heatmap")

        if heatmap_rows is not None:
            heatmap_df = heatmap_rows.copy()

            # 🔥 purchases 숫자화
            heatmap_df["purchases"] = pd.to_numeric(
                heatmap_df["purchases"], errors="coerce"
            ).fillna(0)

            # 🔥 실제 구매 있는 데이터만
            valid_df = heatmap_df[heatmap_df["purchases"] > 0]

            heatmap_rows = heatmap_df.to_dict(orient="records")
        else:
            heatmap_rows = []
            valid_df = []

        # 🔥 핵심 조건 (여기 중요)
        if len(valid_df) >= 1:
            print("구매 전환 히트맵 생성 중...")

            final_report["purchase_age_gender_page"] = {
                "is_visible": True,
                "title": purchase_age_gender_data.get("title", "타겟별 구매전환"),
                "heatmap": heatmap_rows,
            }
        else:
            print("구매 전환 히트맵 없음...")

            final_report["purchase_age_gender_page"] = {
                "is_visible": False
            }

    else:
        print("구매 전환 히트맵 데이터 없음 ...")
        final_report["purchase_age_gender_page"] = {
            "is_visible": False
        }

    _, threshold = get_imp_threshold(target_id, start, end)

    # 3. 타겟 히트맵 데이터 (노출/CTR)
    print("타겟 히트맵 데이터 (노출/CTR) 생성 중...")
    target_df = get_target_avg_imp_ctr_threshold(target_id, start, end, threshold)
    # 히트맵은 테이블 형태가 시각화하기 좋음
    add_ds("target_heatmap", "table", "타겟별 노출 및 CTR 성과", target_df)

    # 3-1. 구매전환 히트맵 데이터
    purchase_heatmap_df = get_purchase_age_gender_heatmap(target_id, start, end)

    if purchase_heatmap_df is not None and not purchase_heatmap_df.empty:
        add_ds(
            "purchase_heatmap",
            "table",
            "타겟별 구매전환 성과",
            purchase_heatmap_df
        )

    # 4. 키워드 분석 (전체/메인/기피 + 명사/형용사)
    print("키워드 분석 (전체/메인/기피 + 명사/형용사) 생성 중...")
    def _normalize_age_selection(age_value):
        if age_value is None:
            return None
        if isinstance(age_value, str):
            age_value = age_value.strip()
            return age_value if age_value else None
        if isinstance(age_value, (list, tuple, set, np.ndarray, pd.Series)):
            ages = [str(v).strip() for v in age_value if v is not None and str(v).strip()]
            return ages if ages else None
        age_value = str(age_value).strip()
        return age_value if age_value else None

    def _normalize_gender_selection(gender_value):
        if gender_value is None:
            return None
        if isinstance(gender_value, str):
            gender_value = gender_value.strip()
            return gender_value if gender_value else None
        if isinstance(gender_value, (list, tuple, set, np.ndarray, pd.Series)):
            genders = [str(v).strip() for v in gender_value if v is not None and str(v).strip()]
            return genders if genders else None
        gender_value = str(gender_value).strip()
        return gender_value if gender_value else None

    target_configs = [
        ("overall", None, None, "전체"),
        ("main", main_age, main_gender, "메인 타겟"),
        ("avoid", avoid_age, avoid_gender, "기피 타겟")
    ]

    # 유효한 타겟 설정만 필터
    active_configs = []
    for prefix, age_raw, gen_raw, label in target_configs:
        age = _normalize_age_selection(age_raw)
        gen = _normalize_gender_selection(gen_raw)
        if prefix != "overall" and not age and not gen:
            continue
        active_configs.append((prefix, age, gen, label))

    # 병렬로 raw_kw (타겟별 1회) + strategic 쿼리 실행
    kw_futures = {}
    strat_futures = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        for prefix, age, gen, label in active_configs:
            # raw_kw: top/bottom 공용 (1회만 조회 후 Python에서 정렬)
            kw_futures[prefix] = executor.submit(
                get_raw_keyword_performance, target_id, start, end, age, gen
            )
            strat_futures[prefix] = executor.submit(
                get_strategic_performance, target_id, start, end, age, gen
            )
        # 결과 수집 (executor 블록 종료 시 모두 완료)
        kw_results   = {k: f.result() for k, f in kw_futures.items()}
        strat_results = {k: f.result() for k, f in strat_futures.items()}

    for prefix, age, gen, label in active_configs:
        raw_kw_df = kw_results[prefix]

        # 상위/하위: 같은 데이터를 Python에서 정렬만 달리 적용
        for is_top in [True, False]:
            suffix = "top" if is_top else "bottom"
            exclude_zero_ctr = not is_top
            # CTR 기준 정렬 (top=내림차순, bottom=오름차순)
            sorted_df = raw_kw_df.sort_values(
                by=["avg_ctr", "total_impressions"],
                ascending=[not is_top, False]
            )

            # 명사 필터링
            nouns = filter_keywords_by_pos(sorted_df, 'noun', exclude_zero_ctr=exclude_zero_ctr)
            add_ds(f"{prefix}_{suffix}_noun", "bar_h", f"{label} {suffix.upper()} 10 (명사)", nouns, "%", "keyword", ["ctr"])

            # 형용사 필터링
            vas = filter_keywords_by_pos(sorted_df, 'verb_adj', exclude_zero_ctr=exclude_zero_ctr)
            add_ds(f"{prefix}_{suffix}_va", "bar_h", f"{label} {suffix.upper()} 10 (형용사)", vas, "%", "keyword", ["ctr"])

        strat_df = strat_results[prefix]
        if strat_df is not None:
            strat_df = strat_df.copy()
            strat_df["combo_overall_ctr"] = pd.to_numeric(strat_df["combo_overall_ctr"], errors="coerce")
            strat_df["with_var_ctr"] = pd.to_numeric(strat_df["with_var_ctr"], errors="coerce")

            # 1. 조합별 item_count를 계산한 뒤, item_count>=2 조건에서만 CTR 상위 6개 선정
            combo_keys = ["ess_1", "ess_2", "combo_overall_ctr"]
            combo_sizes = strat_df.groupby(combo_keys, dropna=False).size().reset_index(name="item_count")
            top_combos = (
                combo_sizes[combo_sizes["item_count"] >= 2]
                .dropna(subset=["combo_overall_ctr"])
                .sort_values(by="combo_overall_ctr", ascending=False)
                .head(6)
            )

            # 2. 상세 카드용: 위에서 선별한 상위 6개 조합만 남김
            final_strat_df = strat_df.merge(top_combos[combo_keys], on=combo_keys, how="inner")

            # 3. 각 조합별 변수 키워드 성과 상위 8개만 유지
            final_strat_df = final_strat_df.sort_values(
                by=["combo_overall_ctr", "ess_1", "ess_2", "with_var_ctr"],
                ascending=[False, True, True, False]
            )
            final_strat_df = final_strat_df.groupby(combo_keys, sort=False).head(8)

            add_ds(f"{prefix}_keyword_combo_detail", "table", f"{label} 상세 분석", final_strat_df)

    # 5. 콘텐츠별 타겟 성과 (상/하위)
    print("콘텐츠별 타겟 성과 (상/하위) 생성 중...")
    for is_top in [True, False]:
        suffix = "top" if is_top else "bottom"
        contents = get_content_ctr_data(target_id, start, end, threshold, is_top=is_top)
        
        content_results = []
        for item in contents[:3]: # 상/하위 3개씩
            detail_df = get_a_content_target_ctr_data(item["ad_id"], start, end)
            if detail_df is not None:
                # 상세 타겟 데이터를 리스트로 변환하여 포함
                item["target_details"] = detail_df.to_dict(orient='records')
            content_results.append(item)
        
        final_report["datasets"][f"content_{suffix}_analysis"] = {
            "kind": "content_card",
            "title": f"성과 {suffix} 콘텐츠 분석",
            "items": content_results
        }

    # 6. 반응 기반 콘텐츠 성과 (지표별 TOP/BOTTOM 3, 6페이지)
    print("반응 기반 콘텐츠 성과 생성 중...")
    for metric in ['likes', 'saves', 'shares']:
        overall_avg = get_reaction_metric_avg(target_id, start, end, metric=metric)

    # metric 컬럼 이름 매핑: DB 반환 키와 일치시킴
        metric_key_map = {
            'likes':  'total_likes',
            'saves':  'total_saves',
            'shares': 'total_shares',
        }
        metric_col = metric_key_map[metric]

        # ── x_scale_max 산출 (상위 데이터를 기준으로 1회만 계산) ──────────
        # x_scale_max는 상위/하위 차트가 공유하는 x축 최대 기준값이다.
        # 상위 5개 콘텐츠 수치의 최대값과 overall_avg 중 더 큰 값을 사용한다.
        # 이렇게 하면 하위 차트에서 막대가 과장되어 늘어나는 현상을 방지한다.
        top_contents_for_scale = get_content_reaction_data(
            target_id, start, end, is_top=True, metric=metric
        )
        top_values_for_scale = [
            float(item.get(metric_col, 0) or 0)
            for item in (top_contents_for_scale or [])
        ]
        # overall_avg와 상위 콘텐츠 최대값 중 더 큰 쪽을 기준으로 삼는다.
        # 모두 0이면 x_scale_max는 0 (visualizer.py에서 예외 처리)
        x_scale_max = max(
            max(top_values_for_scale) if top_values_for_scale else 0,
            float(overall_avg or 0)
        )
        # ────────────────────────────────────────────────────────────────────

        for is_top in [True, False]:
            suffix = "top" if is_top else "bottom"
            reaction_contents = get_content_reaction_data(
                target_id, start, end, is_top=is_top, metric=metric
            )

            for item in (reaction_contents or []):
                raw_caption = str(item.get("caption") or "").strip()
                caption_label = _parse_korean_caption(raw_caption)
                if not caption_label:
                    uploaded = item.get("uploaded_at")
                    caption_label = f"{uploaded:%Y-%m-%d} 업로드" if uploaded else ""
                item["caption_label"] = caption_label

                raw_ctr = item.get("ctr")
                if raw_ctr is None or (isinstance(raw_ctr, float) and raw_ctr != raw_ctr):
                    # float NaN 대응: NaN != NaN 이 True이므로 이 조건으로 검출
                    item["ctr"] = None
                else:
                    item["ctr"] = float(raw_ctr)

            reaction_contents = _assign_reaction_ranks(reaction_contents, metric_col)

            final_report["datasets"][f"reaction_{metric}_{suffix}"] = {
                "kind":         "reaction_bar",
                "title":        f"반응 {suffix} 콘텐츠 ({metric})",
                "metric":       metric,
                "metric_col":   metric_col,
                "overall_avg":  overall_avg,
                "metric_avg":   overall_avg,       # 하위 호환 키 유지
                "is_top":       is_top,
                # 상위/하위 차트 양쪽 모두 이 값을 x축 최대 기준으로 사용한다.
                # 하위 차트가 이 키를 읽어 x축을 상위 차트와 동일하게 맞춘다.
                "x_scale_max":  x_scale_max,
                "items":        reaction_contents
            }


    # 7. CPPR 콘텐츠 효율 + 타겟별 광고비 분포
    print("타겟별 광고비 분포 생성 중...")
    target_spend_df = get_target_spend_distribution(target_id, start, end)
    if target_spend_df is not None:
        final_report["datasets"]["target_spend_bubble"] = {
            "kind":         "target_bubble",
            "title":        "타겟별 광고비 분포",
            "rows":         target_spend_df.replace({pd.NA: None, np.nan: None}).to_dict(orient="records"),
            "main_age":     main_age,
            "main_gender":  main_gender,
            "avoid_age":    avoid_age,
            "avoid_gender": avoid_gender,
        }


    # ── CTR × 팔로우 산점도 데이터 ───────────────────────────
    scatter_rows = get_ctr_follows_scatter_data(target_id, start, end)
    prev_means = get_prev_quarter_ctr_follows_means(target_id, start)

    if scatter_rows:
        final_report["ctr_follows_scatter"] = {
            "rows":        scatter_rows,
            "ctr_mean":    prev_means.get("ctr_mean"),      # median → mean
            "follows_mean": prev_means.get("follows_mean"), # median → mean
        }
    else:
        final_report["ctr_follows_scatter"] = {
            "rows": [], "ctr_mean": None, "follows_mean": None
        }

    # --- [추가] 별첨 자료용 키워드 상세 분석 (4페이지 분량) ---
    
    # 데이터 불러오기
    print("별첨 자료용 키워드 상세 분석 생성 중...")
    df_ess = get_essence_target_performance(target_id, start, end)
    df_var = get_variable_target_performance(target_id, start, end)

    def format_rows(df, col_indices):
        """데이터프레임에서 특정 인덱스의 컬럼만 추출하여 리스트로 반환"""
        if df is None or df.empty: return []
        # NaN 처리 후 리스트 변환
        temp_df = df.replace({pd.NA: None, pd.NaT: None, np.nan: None})
        return temp_df.iloc[:, col_indices].values.tolist()

    def build_ranked_rows(df, col_indices, start_rank, end_rank):
        """
        start_rank~end_rank 구간의 행만 뽑아서 절대 순위 번호를 붙여 반환.
        """
        rows = format_rows(df, col_indices)
        if not rows:
            return []
        start_idx = max(start_rank - 1, 0)
        end_idx = max(end_rank, 0)
        sliced = rows[start_idx:end_idx]
        return [[rank] + row for rank, row in enumerate(sliced, start=start_rank)]

    def build_appendix_split_items(base_title, subtitle, headers, df, col_indices):
        """
        별첨 표를 1~25위, 26~50위 두 개로 분할 생성.
        """
        ranges = [(1, 25), (26, 50)]
        items = []
        for start_rank, end_rank in ranges:
            ranked_rows = build_ranked_rows(df, col_indices, start_rank, end_rank)
            if not ranked_rows:
                continue
            items.append({
                "title": f"{base_title} ({start_rank}~{end_rank}위)",
                "subtitle": subtitle,
                "headers": headers,
                "rows": ranked_rows,
                "footnote": "*등장 광고 수 상위 50개 기준"
            })
        return items

    def build_appendix_full_item(base_title, subtitle, headers, df, col_indices):
        """
        별첨 표를 등장한 전체 키워드(전체 순위)로 1개 생성.
        """
        rows = format_rows(df, col_indices)
        if not rows:
            return []
        ranked_rows = [[i + 1] + row for i, row in enumerate(rows)]
        return [{
            "title": f"{base_title} (전체)",
            "subtitle": subtitle,
            "headers": headers,
            "rows": ranked_rows,
            "footnote": "*등장한 전체 키워드 기준"
        }]

    # 가공된 아이템 리스트 생성
    print("아이템 리스트 생성 중...")
    appendix_items = []
    appendix_items.extend(build_appendix_full_item(
        base_title="많이 사용한 업종 필수 키워드 - 노출",
        subtitle="키워드가 가장 많이 노출된 타겟",
        headers=["랭킹", "키워드", "등장 광고 수", "최다 노출 타겟", "타겟 노출량", "노출 비중", "총 노출량"],
        df=df_ess,
        col_indices=[0, 1, 2, 3, 4, 5]
    ))
    appendix_items.extend(build_appendix_full_item(
        base_title="많이 사용한 업종 필수 키워드 - 클릭",
        subtitle="키워드가 가장 많이 노출된 타겟",
        headers=["랭킹", "키워드", "등장 광고 수", "최다 클릭 타겟", "타겟 클릭량", "클릭 비중", "총 클릭량"],
        df=df_ess,
        col_indices=[0, 1, 6, 7, 8, 9]
    ))
    appendix_items.extend(build_appendix_split_items(
        base_title="많이 사용한 브랜드 변수 키워드 - 노출",
        subtitle="키워드가 가장 많이 노출된 타겟",
        headers=["랭킹", "키워드", "등장 광고 수", "최다 노출 타겟", "타겟 노출량", "노출 비중", "총 노출량"],
        df=df_var,
        col_indices=[0, 1, 2, 3, 4, 5]
    ))
    appendix_items.extend(build_appendix_split_items(
        base_title="많이 사용한 브랜드 변수 키워드 - 클릭",
        subtitle="키워드가 가장 많이 노출된 타겟",
        headers=["랭킹", "키워드", "등장 광고 수", "최다 클릭 타겟", "타겟 클릭량", "클릭 비중", "총 클릭량"],
        df=df_var,
        col_indices=[0, 1, 6, 7, 8, 9]
    ))

    # 최종 리포트 구조에 삽입 (HTML 템플릿의 appendix_groups 구조에 맞춤)
    final_report["appendix_groups"] = [
        {
            "title": "",
            "items": appendix_items
        }
    ]

    # --- [기존] 6. 최종 JSON 저장 ---

    # 6. 최종 JSON 저장
    output_path = "json_reports/integrated_report.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4, default=str)
    
    print(f"✅ 모든 요구사항이 반영된 리포트 생성 완료: {output_path}")

if __name__ == "__main__":
    run()
