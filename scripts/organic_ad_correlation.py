"""
organic_ad_correlation.py
==========================
브랜드별 오가닉 vs 광고 콘텐츠 성과(팔로우/좋아요/저장) 상관관계 분석.

함수 정의만 담당한다. 단계별 실행/검증은
notebooks/organic_ad_correlation_analysis.ipynb에서 수행한다(개발 검증 로그 용도 —
CLI 실행 진입점은 아래 run_all() / `python scripts/organic_ad_correlation.py --period ...`).

전체 브랜드 순회 산출물이 필요하면 run_all()을 CLI로 실행한다:
    python scripts/organic_ad_correlation.py --period all
outputs/{period}/ 아래 결과를 확인할 때는 notebooks/review_all_results.ipynb를 쓴다
(DB 연결 없이 outputs/ 파일만 읽어서 표시하는 별도 리뷰 노트북).

지표 범위(2026-08-04 확정): follows/likes/saved 3종만 organic↔ad 비교 대상으로 삼는다.
'views'는 제외했다 — ad 쪽에 organic views(콘텐츠 조회수)와 대응하는 지표가 없어
(video_views는 영상 전용이라 분석 스코프(IMAGE/CAROUSEL_ALBUM, ig_media_type NOT IN
('VIDEO','REEL'))에 연결된 광고 3915건 전부 정지 이미지 기반이라 성립하지 않고,
reach/impressions는 organic views와 집계 정의(중복 허용 여부, unique 기준)가 근본적으로
달라 대체 불가) views는 비교 대상에서 제외한다. organic 콘텐츠 자체의 조회수는 별도
진단(오가닉 단독 분석)에서 참고 가능(get_organic_ad_pairs()의 organic_views 컬럼 참고).

브랜드 식별 정책(2026-08-04 사용자 확정, DB 조회로 검증 완료):
- clients.id 5(depart_business)/31(sub_depart_business)는 client_info.brand_name이
  둘 다 [De;part, 디파트, Depart]로 동일한 메인/서브 관계라 'depart_business_merged'로
  병합한다. clients.id 4(depart_creative)는 brand_name이 NULL이지만 병합 대상이 아니므로
  ad_accounts.name 폴백으로 별도 브랜드 유지. clients.id 3(coralier, brand_name=[Coralier,...])
  은 제외 목록에 없으므로 그대로 포함.
- clients.username = 'kbooster'(clients.id=11, 실측 확인됨)는 에이전시 자체 테스트 계정으로
  분석 대상에서 제외.
- 이 파일은 scripts/processor.py, scripts/flp_keyword_analysis.py를 import하지 않는다
  (서로 다른 분석 트랙 — 참고만 하고 독립적으로 구현). 단, ad_name 정제 로직만은
  scripts/ctr_baseline.py의 clean_ad_name()을 그대로 import해서 재사용한다 (동일 로직을
  복사하면 두 파일이 따로 노는 문제가 생기므로 — ctr_baseline.py 자체는 수정하지 않음).

캠페인 필터 예외(2026-08-04 확정, DB 조회로 검증 완료):
- ad_agg(현 content_ad_agg) 집계는 원칙적으로 campaigns.name LIKE '[디파트]%%'인 캠페인만
  포함한다. 단 MERGE_CLIENT_IDS(5, 31 = depart_business_merged)는 에이전시 자체 계정이라
  전부 포함(캠페인명 규칙 예외).
- clients.id=3(username='coralier')과 clients.id=25(username='coralier_info')는
  client_info.brand_name이 우연히 동일하게 중복 저장된 서로 다른 client 레코드다.
  id=3은 business_portfolios가 아예 없어 organic_perf에 절대 매칭되지 않는 죽은 행이고,
  실제 데이터는 전부 id=25가 만든다. id=25의 ad_accounts 중 하나(coralier_official)는
  '[디파트]' 접두사가 없는 캠페인(당당플로우_시즌1차, coralier_ad, SUMMER SALE 등)이 섞여
  있지만, 이는 에이전시 자체 계정이 아니라 정상 외부 클라이언트이므로 이 캠페인들은
  "디파트가 대행하지 않은 광고"일 가능성이 높아 예외에 넣지 않는다(사용자 확정).
"""
from __future__ import annotations

import io
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as scipy_stats
from sqlalchemy import text

try:
    from scripts.db_connector import get_engine
except ImportError:
    from db_connector import get_engine

try:
    from scripts.ctr_baseline import clean_ad_name
except ImportError:
    from ctr_baseline import clean_ad_name

try:
    from scripts.visualizer import build_color_map
except ImportError:
    from visualizer import build_color_map

try:
    from scripts._local_review_html import build_html_report
except ImportError:
    from _local_review_html import build_html_report

try:
    from matplotlib_venn import venn2
    _HAS_MATPLOTLIB_VENN = True
except ImportError:
    _HAS_MATPLOTLIB_VENN = False

# main.py의 리포트 테마 컬러와 동일 — run_all() CLI 단독 실행 시 color_map을
# 내부에서 만들기 위한 고정값.
THEME_COLOR = "#C9A67F"


# ──────────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────────

# 기간 프리셋 → 개월수 변환 ('all'은 기간 제한 없음 → None)
PERIOD_TO_MONTHS: dict[str, Optional[int]] = {
    "3m": 3, "6m": 6, "12m": 12, "all": None,
}

# 병합 대상: clients.id 5(depart_business), 31(sub_depart_business).
# 둘 다 client_info.brand_name = [De;part, 디파트, Depart]로 동일한 메인/서브 관계.
MERGE_CLIENT_IDS: tuple[int, ...] = (5, 31)
MERGE_BRAND_LABEL: str = "depart_business_merged"

# clients.username 기준 제외 (ig_accounts.username 아님 — 2026-08-04 정정 확정).
# DB 실측: clients.id=11, username='kbooster'.
EXCLUDED_USERNAMES: tuple[str, ...] = ("kbooster",)

# ig_contents.ig_media_type 실측값(2026-08-04 확인): VIDEO, REEL(단수), IMAGE, CAROUSEL_ALBUM.
# 원 스펙엔 'REELS'(복수)로 적혀 있었으나 DB엔 단수형만 존재 — 복수형을 쓰면 REEL이 필터링되지
# 않는 버그가 되므로 실제 값 기준으로 정정해서 채택한다.
EXCLUDED_MEDIA_TYPES: tuple[str, ...] = ("VIDEO", "REEL")

DEPART_CAMPAIGN_PREFIX = "[디파트]"

# 'views'는 organic↔ad 비교 지표에서 제외한다 (2026-08-04 확정).
# ad 쪽에 organic views(콘텐츠 조회수)와 대응하는 지표가 없어(video_views는 영상 전용이라
# 분석 스코프(IMAGE/CAROUSEL_ALBUM, ig_media_type NOT IN ('VIDEO','REEL'))에 연결된 광고
# 3915건 전부 정지 이미지 기반이라 성립하지 않고, reach/impressions는 organic views와
# 집계 정의(중복 허용 여부, unique 기준)가 근본적으로 달라 대체 불가) views는 비교 대상에서
# 제외한다. organic 콘텐츠 자체의 조회수는 별도 진단(오가닉 단독 분석)에서 참고 가능
# (get_organic_ad_pairs()가 반환하는 organic_views 컬럼 자체는 남겨둠).
#
# compute_correlations / build_butterfly_dataset이 다루는 지표 3종.
# organic 쪽은 ig_content_insights.{follows,likes,saved} 컬럼을 그대로 사용.
# ad 쪽은 ad_performance_daily에 동일한 이름의 컬럼이 없어 다음으로 매핑한다(실측 스키마 기준):
#   ad_follows -> instagram_profile_follows
#   ad_likes   -> post_reactions (좋아요 단독 컬럼이 없어 반응 수 전체로 근사)
#   ad_saved   -> post_saves
METRICS: tuple[str, ...] = ("follows", "likes", "saved")
MIN_SAMPLE_SIZE = 10


# ──────────────────────────────────────────────────────────────────────────
# 데이터 조회
# ──────────────────────────────────────────────────────────────────────────
def get_organic_ad_pairs(period: str, engine=None) -> pd.DataFrame:
    """
    브랜드(콘텐츠) 단위로 오가닉 성과와 연결 광고 성과를 나란히 붙인 페어 데이터를 만든다.

    CTE 구성:
      - clients_scope   : EXCLUDED_USERNAMES(clients.username) 제외한 client_id 목록.
                          이 필터를 여기 한 곳에서만 적용해 아래 두 CTE에 중복 기입하지 않는다.
      - brand_map       : brand_name 계산 전용. ad_accounts는 브랜드명 폴백값(aa.name)을
                          얻기 위해서만 거치며, 콘텐츠 조회 경로와는 완전히 분리된다.
                          한 client_id에 business_portfolios/ad_accounts가 여러 개 걸릴 수
                          있어 DISTINCT ON (cl.id)로 1건만 남기고, 여러 ad_accounts 중에는
                          가장 최근 생성된(created_at DESC) 것의 name을 대표값으로 쓴다.
                          MERGE_CLIENT_IDS(5, 31)는 이 폴백 로직보다 우선 적용되어
                          'depart_business_merged'로 고정된다.
      - latest_content_insights : ig_content_insights에서 DISTINCT ON (content_id)
                          ORDER BY as_of_date DESC로 콘텐츠별 최신 스냅샷만 사용.
      - organic_perf    : 오가닉 콘텐츠 조회 전용. business_portfolios -> ig_accounts ->
                          ig_contents 경로만 사용하고 ad_accounts를 전혀 거치지 않는다
                          (scripts/flp_keyword_analysis.fetch_organic_content_base와 동일
                          경로 — 한 client_id에 ad_accounts가 여러 개 걸려도 오가닉 콘텐츠가
                          중복 집계되지 않도록 brand_map과 분리했다).
                          ig_media_type NOT IN EXCLUDED_MEDIA_TYPES, organic follows IS NOT
                          NULL 조건 적용. ic.caption도 함께 가져온다 —
                          _build_content_label()의 2차 폴백(ad_name_clean 없을 때)용.
      - content_ad_agg  : campaigns.name LIKE '[디파트]%%' 필터(+ MERGE_CLIENT_IDS 예외) +
                          ads.source_ig_media_id로 ig_contents.fb_ig_media_id와 매칭.
                          SUM 집계(ad_follows/ad_likes/ad_saved)와 대표 ad_name(가장 최근
                          fb_created_time인 광고)을 윈도우 함수로 동시에 계산한 뒤
                          DISTINCT ON (content_id)로 1행 collapse — ad_agg와 latest_ad_name을
                          같은 FROM/WHERE를 두 번 쓰지 않도록 하나로 통합했다.

    최종 결합: organic_perf(N) JOIN brand_map(1, client_id 기준) JOIN content_ad_agg
    (content_id 기준, INNER — 2026-08-04 확정). 이 분석의 대상은 애초에 "[디파트] 캠페인
    또는 우리 브랜드(MERGE_CLIENT_IDS 예외)로 집행된 광고와 연결된 콘텐츠"이므로, 광고
    자체가 매칭 안 되는 콘텐츠(ads 테이블에 해당 content_id가 없는 경우)는 결과에서 아예
    제외한다 — LEFT JOIN이던 시절엔 순수 오가닉(광고 없음) 콘텐츠까지 다 들어와 있었는데,
    compute_correlations()가 어차피 organic/ad 값이 모두 non-null인 행만 쓰기 때문에
    상관계수(rho/p_value/n) 자체는 이 변경으로 바뀌지 않는다(실측 diff 확인 완료) — 영향은
    build_butterfly_dataset/build_dominance_table의 organic_only 후보군에서만 나타난다
    (이제 그쪽도 "광고는 붙었지만 이 지표 성과가 없는" 콘텐츠만 후보가 됨).
    ad_follows/ad_likes/ad_saved/ad_name_raw는 여전히 COALESCE 없이 NULL을 남긴다 —
    광고가 매칭된 콘텐츠라도 ad_performance_daily에 해당 지표 데이터가 없으면 NULL이 되는
    정상 케이스이기 때문(광고 자체가 없는 경우와는 다름 — 이제 후자는 INNER JOIN으로
    이미 걸러졌으므로, 남은 NULL은 전부 "광고는 있는데 그 지표 데이터가 없음" 케이스다).

    SQL fetch 후 파이썬 후처리:
        df["ad_name_clean"] = df["ad_name_raw"].apply(
            lambda x: clean_ad_name(x)[0] if isinstance(x, str) else None
        )
        (ad_name_raw가 NULL이면 ad_name_clean도 NULL. clean_ad_name()이 빈 결과(None)를
        내도 그 결과 그대로 사용 — 원본 ad_name으로 대체하지 않는다.)

    Parameters
    ----------
    period : '3m'/'6m'/'12m'/'all' 중 하나. ig_timestamp(콘텐츠 업로드일) 기준 필터.
    engine : SQLAlchemy engine. None이면 get_engine()으로 생성.

    Returns
    -------
    DataFrame: content_id, brand_name, content_date, caption,
               organic_follows, organic_views, organic_likes, organic_saved,
               ad_follows, ad_likes, ad_saved, ad_name_clean
               (organic_views는 오가닉 단독 참고용으로 남겨두되, ad_views는 만들지 않는다 —
               근거는 파일 상단 METRICS 주석 참고. INNER JOIN이라 모든 행이 광고 매칭된
               콘텐츠 — ad_follows/ad_likes/ad_saved는 그래도 ad_performance_daily에 해당
               지표 데이터가 없으면 NULL일 수 있다(0으로 채우지 않음). ad_name_clean은
               광고가 있어도 clean_ad_name()이 비우면 NULL — 이때 _build_content_label()이
               caption으로 폴백한다)
    """
    if period not in PERIOD_TO_MONTHS:
        raise ValueError(f"period는 {list(PERIOD_TO_MONTHS)} 중 하나여야 합니다: {period!r}")

    engine = engine or get_engine()
    months = PERIOD_TO_MONTHS[period]

    params: dict = {
        "excluded_usernames": list(EXCLUDED_USERNAMES),
        "merge_client_ids": list(MERGE_CLIENT_IDS),
        "excluded_media_types": list(EXCLUDED_MEDIA_TYPES),
        "campaign_prefix": f"{DEPART_CAMPAIGN_PREFIX}%",
    }
    date_filter_sql = ""
    if months is not None:
        date_filter_sql = "AND ic.ig_timestamp >= (CURRENT_DATE - make_interval(months => :months))"
        params["months"] = months

    query = f"""
        WITH clients_scope AS (
            SELECT cl.id AS client_id
            FROM clients cl
            WHERE cl.username IS NULL OR cl.username <> ALL(:excluded_usernames)
        ),
        brand_map AS (
            SELECT DISTINCT ON (cl.id)
                cl.id AS client_id,
                CASE
                    WHEN cl.id = ANY(:merge_client_ids) THEN 'depart_business_merged'
                    ELSE COALESCE(ci.brand_name[1], aa.name, cl.id::text)
                END AS brand_name
            FROM clients cl
            JOIN clients_scope cs ON cs.client_id = cl.id
            LEFT JOIN client_info ci ON ci.client_id = cl.id
            LEFT JOIN business_portfolios bp ON bp.client_id = cl.id
            LEFT JOIN ad_accounts aa ON aa.business_portfolio_id = bp.id
            ORDER BY cl.id, aa.created_at DESC NULLS LAST
        ),
        latest_content_insights AS (
            SELECT DISTINCT ON (content_id)
                content_id, follows, views, likes, saved
            FROM ig_content_insights
            ORDER BY content_id, as_of_date DESC
        ),
        organic_perf AS (
            SELECT
                ic.id             AS content_id,
                bp.client_id      AS client_id,
                ic.ig_timestamp   AS content_date,
                ic.caption        AS caption,
                li.follows        AS organic_follows,
                li.views          AS organic_views,
                li.likes          AS organic_likes,
                li.saved          AS organic_saved
            FROM business_portfolios bp
            JOIN clients_scope cs ON cs.client_id = bp.client_id
            JOIN ig_accounts ia   ON ia.business_portfolio_id = bp.id
            JOIN ig_contents ic   ON ic.ig_id = ia.id
            LEFT JOIN latest_content_insights li ON li.content_id = ic.id
            WHERE ic.ig_media_type <> ALL(:excluded_media_types)
              AND li.follows IS NOT NULL
              {date_filter_sql}
        ),
        content_ad_agg AS (
            SELECT DISTINCT ON (sub.content_id)
                sub.content_id,
                sub.ad_follows,
                sub.ad_likes,
                sub.ad_saved,
                sub.ad_name_raw
            FROM (
                SELECT
                    ic.id AS content_id,
                    SUM(apd.instagram_profile_follows) OVER (PARTITION BY ic.id) AS ad_follows,
                    SUM(apd.post_reactions)             OVER (PARTITION BY ic.id) AS ad_likes,
                    SUM(apd.post_saves)                 OVER (PARTITION BY ic.id) AS ad_saved,
                    FIRST_VALUE(a.ad_name) OVER (
                        PARTITION BY ic.id ORDER BY a.fb_created_time DESC
                    ) AS ad_name_raw
                FROM ads a
                JOIN ad_sets aset ON a.ad_set_id = aset.id
                JOIN campaigns c  ON aset.campaign_id = c.id
                JOIN ad_accounts aa2 ON aa2.id = a.account_id
                JOIN business_portfolios bp2 ON bp2.id = aa2.business_portfolio_id
                JOIN ig_contents ic ON ic.fb_ig_media_id = a.source_ig_media_id
                LEFT JOIN ad_performance_daily apd ON apd.ad_id = a.id
                WHERE (c.name LIKE :campaign_prefix OR bp2.client_id = ANY(:merge_client_ids))
            ) sub
            ORDER BY sub.content_id
        )
        SELECT
            op.content_id,
            bm.brand_name,
            op.content_date,
            op.caption,
            op.organic_follows,
            op.organic_views,
            op.organic_likes,
            op.organic_saved,
            content_ad_agg.ad_follows,
            content_ad_agg.ad_likes,
            content_ad_agg.ad_saved,
            content_ad_agg.ad_name_raw
        FROM organic_perf op
        JOIN brand_map bm ON bm.client_id = op.client_id
        JOIN content_ad_agg ON content_ad_agg.content_id = op.content_id
        ORDER BY bm.brand_name, op.content_date
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    df["ad_name_clean"] = df["ad_name_raw"].apply(
        lambda x: clean_ad_name(x)[0] if isinstance(x, str) else None
    )
    return df.drop(columns=["ad_name_raw"])


# ──────────────────────────────────────────────────────────────────────────
# 상관관계 계산
# ──────────────────────────────────────────────────────────────────────────
def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    브랜드별로 organic vs ad 성과 쌍(follows/likes/saved — METRICS 참고, views는 제외)의
    Spearman 상관을 계산한다.

    - 표본(브랜드 내 콘텐츠 수, organic/ad 양쪽 값이 결측이 아닌 행 기준) 10개 미만이면
      계산을 스킵하고 note에 사유를 남긴다 (status='skipped_low_n').
    - organic 또는 ad 쪽 값이 전부 동일(분산 0 — 대개 전부 0)이면 Spearman 자체가
      정의되지 않으므로 계산하지 않고 'N/A - 오가닉/광고 {metric} 미발생'으로 표시한다
      (status='skipped_zero_variance').

    Returns
    -------
    DataFrame: brand_name, metric, n, rho, p_value, status, note
               (status: 'ok' / 'skipped_low_n' / 'skipped_zero_variance')
    """
    rows = []
    for brand_name, sub in df.groupby("brand_name"):
        for metric in METRICS:
            organic_col = f"organic_{metric}"
            ad_col = f"ad_{metric}"
            if organic_col not in sub.columns or ad_col not in sub.columns:
                continue

            organic_vals = pd.to_numeric(sub[organic_col], errors="coerce")
            ad_vals = pd.to_numeric(sub[ad_col], errors="coerce")
            valid_mask = organic_vals.notna() & ad_vals.notna()
            organic_vals = organic_vals[valid_mask]
            ad_vals = ad_vals[valid_mask]
            n = len(organic_vals)

            if n < MIN_SAMPLE_SIZE:
                rows.append({
                    "brand_name": brand_name, "metric": metric, "n": n,
                    "rho": None, "p_value": None,
                    "status": "skipped_low_n",
                    "note": f"표본 {n}건 < {MIN_SAMPLE_SIZE}건 - 상관계수 계산 스킵",
                })
                continue

            if organic_vals.nunique() <= 1 or ad_vals.nunique() <= 1:
                rows.append({
                    "brand_name": brand_name, "metric": metric, "n": n,
                    "rho": None, "p_value": None,
                    "status": "skipped_zero_variance",
                    "note": f"N/A - 오가닉/광고 {metric} 미발생",
                })
                continue

            rho, p_value = scipy_stats.spearmanr(organic_vals, ad_vals)
            rows.append({
                "brand_name": brand_name, "metric": metric, "n": n,
                "rho": round(float(rho), 4), "p_value": round(float(p_value), 4),
                "status": "ok", "note": "",
            })

    return pd.DataFrame(
        rows, columns=["brand_name", "metric", "n", "rho", "p_value", "status", "note"]
    )


# ──────────────────────────────────────────────────────────────────────────
# 콘텐츠 라벨 (build_butterfly_dataset / build_dominance_table 공용)
# ──────────────────────────────────────────────────────────────────────────
def _build_content_label(content_id, ad_name_clean, caption, content_date) -> str:
    """
    콘텐츠 하나의 표시 라벨을 만든다. 폴백 순서: ad_name_clean -> caption -> content_id+날짜.

    - ad_name_clean이 non-empty 문자열이면 그대로 반환.
    - 그 외(NULL/빈 문자열 — 매칭된 광고가 없거나 clean_ad_name()이 비운 경우)면
      caption이 non-empty 문자열일 때 caption(strip만, truncate 없음)을 반환한다.
      (INNER JOIN 이후로도 광고 자체는 매칭됐지만 ad_name이 정제 후 빈 값이 되는 경우가
      있을 수 있어 — 예: 광고 소재명이 날짜/코드 토큰뿐인 경우 — 콘텐츠 캡션으로 대체한다.)
    - caption도 없으면 "{content_id} ({content_date:%Y-%m-%d})" 폴백 (content_date가 NaT면 '?').
    """
    if isinstance(ad_name_clean, str) and ad_name_clean:
        return ad_name_clean
    if isinstance(caption, str) and caption.strip():
        return caption.strip()
    if pd.notna(content_date):
        date_str = pd.Timestamp(content_date).strftime("%Y-%m-%d")
    else:
        date_str = "?"
    return f"{content_id} ({date_str})"


# ──────────────────────────────────────────────────────────────────────────
# 나비형 차트 데이터셋 / 렌더링
# ──────────────────────────────────────────────────────────────────────────
def build_butterfly_dataset(
    df: pd.DataFrame, brand_name: str, metric: str, top_n: int = 10
) -> dict:
    """
    특정 브랜드 x 메트릭에 대해 나비형(butterfly) 차트용 데이터셋을 만든다.

    왼쪽 = organic_{metric} 내림차순 top_n, 오른쪽 = ad_{metric} 내림차순 top_n.
    두 쪽은 독립적으로 정렬되므로 같은 행(순번)이 같은 콘텐츠라는 보장은 없다
    — "각 채널에서 각각 잘 나온 콘텐츠"를 나란히 보여주는 목적.

    top_n은 이 함수(시각화)에만 적용되는 값이다. compute_correlations()의 상관계수 계산에는
    영향을 주지 않는다 — 그쪽은 브랜드 내 전체 콘텐츠(표본 MIN_SAMPLE_SIZE건 이상)를 그대로
    쓰고, 이 함수는 그중 시각화용으로 상위 top_n개만 잘라서 보여주는 역할이다.

    scripts/to_json.py의 add_ds() 스키마(kind/title/unit/labels/series)를 참고한 형태로,
    좌우가 서로 다른 정렬이라는 점만 labels_left/labels_right로 분리했다.

    Parameters
    ----------
    top_n : 좌우 각각 보여줄 콘텐츠 개수 (기본값 10). render_organic_ad_butterfly_chart()가
            이 dataset의 labels_left/labels_right 길이를 그대로 읽어 차트 높이/막대 개수를
            맞추므로, top_n을 바꿔도 렌더링 쪽을 별도로 수정할 필요가 없다.

    Returns
    -------
    dict: {"kind": "organic_ad_butterfly", "title": str, "unit": str,
           "labels_left": [ad_name 또는 "content_id (date)"...],
           "labels_right": [ad_name 또는 "content_id (date)"...],
           "series": [{"name": "organic", "data": [...]}, {"name": "ad", "data": [...]}]}
    """
    if metric not in METRICS:
        raise ValueError(f"metric은 {list(METRICS)} 중 하나여야 합니다: {metric!r}")

    organic_col = f"organic_{metric}"
    ad_col = f"ad_{metric}"
    label_cols = ["content_id", "ad_name_clean", "caption", "content_date"]
    sub = df[df["brand_name"] == brand_name]

    organic_top = (
        sub[label_cols + [organic_col]]
        .dropna(subset=[organic_col])
        .sort_values(organic_col, ascending=False)
        .head(top_n)
    )
    ad_top = (
        sub[label_cols + [ad_col]]
        .dropna(subset=[ad_col])
        .sort_values(ad_col, ascending=False)
        .head(top_n)
    )

    def _labels(rows: pd.DataFrame) -> list[str]:
        return [
            _build_content_label(r.content_id, r.ad_name_clean, r.caption, r.content_date)
            for r in rows.itertuples(index=False)
        ]

    return {
        "kind": "organic_ad_butterfly",
        "title": f"{brand_name} — {metric} (organic vs ad)",
        "unit": "",
        "labels_left": _labels(organic_top),
        "labels_right": _labels(ad_top),
        "series": [
            {"name": "organic", "data": organic_top[organic_col].astype(float).tolist()},
            {"name": "ad", "data": ad_top[ad_col].astype(float).tolist()},
        ],
    }


def build_dominance_table(
    df: pd.DataFrame, brand_name: str, metric: str, top_n: int = 10
) -> dict:
    """
    organic_{metric} 상위 top_n ∪ ad_{metric} 상위 top_n 콘텐츠에 dominance 라벨을 붙인다.

    dominance:
      - "both"         : organic top_n과 ad top_n 양쪽에 모두 있는 콘텐츠
      - "organic_only" : organic top_n에만 있는 콘텐츠
      - "ad_only"      : ad top_n에만 있는 콘텐츠

    정렬: both -> organic_only -> ad_only 순 (둘 다 잘된 케이스를 상단에 배치).
    그룹 내부는 organic_{metric} 내림차순(ad_only는 organic 값이 없으므로 ad_{metric}
    내림차순).

    콘텐츠 라벨은 build_butterfly_dataset과 동일하게 _build_content_label()로 만든다
    (ad_name_clean -> caption -> "content_id (content_date)" 순 폴백).

    Returns
    -------
    dict: scripts/to_json.py의 add_ds() kind="table" 스키마 참고.
          {"kind": "table", "title": str,
           "headers": ["콘텐츠", "organic_{metric}", "ad_{metric}", "dominance"],
           "rows": [[label, organic_value, ad_value, dominance], ...],
           "footnote": str}
    """
    if metric not in METRICS:
        raise ValueError(f"metric은 {list(METRICS)} 중 하나여야 합니다: {metric!r}")

    organic_col = f"organic_{metric}"
    ad_col = f"ad_{metric}"
    label_cols = ["content_id", "ad_name_clean", "caption", "content_date"]
    sub = df[df["brand_name"] == brand_name]

    organic_top = (
        sub[label_cols + [organic_col]]
        .dropna(subset=[organic_col])
        .sort_values(organic_col, ascending=False)
        .head(top_n)
        .set_index("content_id")
    )
    ad_top = (
        sub[label_cols + [ad_col]]
        .dropna(subset=[ad_col])
        .sort_values(ad_col, ascending=False)
        .head(top_n)
        .set_index("content_id")
    )

    both_ids = organic_top.index.intersection(ad_top.index)
    organic_only_ids = organic_top.index.difference(ad_top.index)
    ad_only_ids = ad_top.index.difference(organic_top.index)

    rows: list[dict] = []

    for cid in both_ids:
        o_row = organic_top.loc[cid]
        a_row = ad_top.loc[cid]
        rows.append({
            "content_id": cid,
            "label": _build_content_label(cid, o_row["ad_name_clean"], o_row["caption"], o_row["content_date"]),
            "organic_value": float(o_row[organic_col]),
            "ad_value": float(a_row[ad_col]),
            "dominance": "both",
        })

    for cid in organic_only_ids:
        o_row = organic_top.loc[cid]
        rows.append({
            "content_id": cid,
            "label": _build_content_label(cid, o_row["ad_name_clean"], o_row["caption"], o_row["content_date"]),
            "organic_value": float(o_row[organic_col]),
            "ad_value": None,
            "dominance": "organic_only",
        })

    for cid in ad_only_ids:
        a_row = ad_top.loc[cid]
        rows.append({
            "content_id": cid,
            "label": _build_content_label(cid, a_row["ad_name_clean"], a_row["caption"], a_row["content_date"]),
            "organic_value": None,
            "ad_value": float(a_row[ad_col]),
            "dominance": "ad_only",
        })

    dominance_order = {"both": 0, "organic_only": 1, "ad_only": 2}
    rows.sort(key=lambda r: (
        dominance_order[r["dominance"]],
        -(r["organic_value"] if r["organic_value"] is not None else r["ad_value"]),
    ))

    table_rows = [
        [r["label"], r["organic_value"], r["ad_value"], r["dominance"]] for r in rows
    ]

    return {
        "kind": "table",
        "title": f"{brand_name} — {metric} organic/ad 우위 비교",
        "headers": ["콘텐츠", f"organic_{metric}", f"ad_{metric}", "dominance"],
        "rows": table_rows,
        "footnote": f"organic/ad 각각 상위 {top_n}개 콘텐츠의 합집합, {len(table_rows)}건",
    }


def render_dominance_venn(dominance_dict: dict, color_map: dict) -> str:
    """
    build_dominance_table()의 both/organic_only/ad_only 개수를 2-circle 벤다이어그램으로
    시각화한다.

    Parameters
    ----------
    dominance_dict : build_dominance_table()의 반환값 그대로
        (rows 안의 각 항목에서 dominance 필드를 세어 both/organic_only/ad_only 카운트 산출)
    color_map : build_color_map(theme_color) 결과, 기존 렌더 함수들과 동일 컨벤션

    구현:
        - matplotlib-venn 패키지 사용 가능하면 venn2(subsets=(organic_only, ad_only, both))로
          렌더 (both=0이면 venn2가 자동으로 두 원을 겹치지 않게 떨어뜨려 그린다 — 실측 확인함).
        - 패키지 없으면 matplotlib.patches.Circle 두 개를 겹쳐 그리고, 겹치는 영역에 both
          개수, 왼쪽 전용 영역에 organic_only, 오른쪽 전용 영역에 ad_only 숫자를 직접 텍스트로
          표기하는 방식으로 자체 구현 (both=0이면 두 원의 중심 간 거리를 반지름 합보다 크게
          벌려 겹치지 않게 배치).
        - 왼쪽 원 = organic(color_map의 base), 오른쪽 원 = ad(color_map의 dark), 교집합은
          color_map의 highlight.
        - 제목: "{brand_name} · {metric} 콘텐츠 겹침" (dominance_dict["title"]을 파싱해서 구성).

    Returns
    -------
    str: SVG 문자열 (_fig_to_svg 방식과 동일하게 "<svg"부터 시작). dominance_dict가 비었거나
         "dominance" 헤더가 없으면 빈 문자열.
    """
    if not dominance_dict:
        return ""

    headers = dominance_dict.get("headers") or []
    rows = dominance_dict.get("rows") or []
    if "dominance" not in headers:
        return ""
    dominance_idx = headers.index("dominance")

    counts = {"both": 0, "organic_only": 0, "ad_only": 0}
    for row in rows:
        label = row[dominance_idx]
        if label in counts:
            counts[label] += 1
    organic_only, ad_only, both = counts["organic_only"], counts["ad_only"], counts["both"]

    title_raw = dominance_dict.get("title") or ""
    brand_metric = title_raw.split(" organic/ad")[0] if " organic/ad" in title_raw else title_raw
    if " — " in brand_metric:
        brand_part, metric_part = brand_metric.split(" — ", 1)
        venn_title = f"{brand_part} · {metric_part} 콘텐츠 겹침"
    else:
        venn_title = f"{brand_metric} 콘텐츠 겹침"

    left_color = color_map.get("base", "#4e73df")
    right_color = color_map.get("dark", "#2e59d9")
    # 교집합 색: color_map["highlight"]는 visualizer.build_color_map()에서 "형광펜용:
    # 매우 밝고 채도 낮춤"으로 설계된 색이라(테마에 따라 거의 흰색이 나올 수 있음, 실측:
    # 테마 #C9A67F에서 highlight=#ffffff) 흰 배경 위 교집합 렌즈 채우기로 쓰면 배경과
    # 구분이 안 되는 문제가 있어(2026-08-04 실측으로 발견) "darker"로 교체.
    both_color = color_map.get("darker", "#333333")
    text_color = color_map.get("title", "#111111")
    muted_color = color_map.get("muted", "#666666")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    if _HAS_MATPLOTLIB_VENN:
        v = venn2(subsets=(organic_only, ad_only, both), set_labels=("Organic", "Ad"), ax=ax)

        patch_10 = v.get_patch_by_id("10")
        if patch_10 is not None:
            patch_10.set_color(left_color)
            patch_10.set_alpha(0.75)

        patch_01 = v.get_patch_by_id("01")
        if patch_01 is not None:
            patch_01.set_color(right_color)
            patch_01.set_alpha(0.75)

        patch_11 = v.get_patch_by_id("11")
        if patch_11 is not None:
            patch_11.set_color(both_color)
            patch_11.set_alpha(0.9)

        for label_id in ("10", "01"):
            label = v.get_label_by_id(label_id)
            if label is not None:
                label.set_color(text_color)
                label.set_fontsize(13)

        label_11 = v.get_label_by_id("11")
        if label_11 is not None:
            # both_color(darker)가 어두운 배경이라 text_color(어두운 계열)로는 대비가
            # 안 나옴 — 이 라벨만 흰색으로 고정.
            label_11.set_color("#ffffff")
            label_11.set_fontsize(13)

        for set_label in v.set_labels:
            if set_label is not None:
                set_label.set_color(muted_color)
    else:
        # matplotlib-venn 미설치 시 자체 구현.
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")

        radius = 1.0
        # both=0이면 두 원이 겹치지 않도록 중심 간 거리(half_gap*2)를 반지름 합(2*radius)보다
        # 크게 벌린다 (엣지케이스: "두 원이 안 겹침" 형태로 정상 표시).
        half_gap = radius * 0.62 if both > 0 else radius * 1.15
        left_center = (-half_gap, 0.0)
        right_center = (half_gap, 0.0)

        ax.add_patch(plt.Circle(
            left_center, radius, facecolor=left_color, alpha=0.6,
            edgecolor="white", linewidth=1.5,
        ))
        ax.add_patch(plt.Circle(
            right_center, radius, facecolor=right_color, alpha=0.6,
            edgecolor="white", linewidth=1.5,
        ))

        ax.text(
            left_center[0] - radius * 0.55, 0, str(organic_only),
            ha="center", va="center", fontsize=15, fontweight="bold", color=text_color,
        )
        ax.text(
            right_center[0] + radius * 0.55, 0, str(ad_only),
            ha="center", va="center", fontsize=15, fontweight="bold", color=text_color,
        )
        if both > 0:
            # 겹치는 영역은 두 반투명 원이 곂쳐 더 짙어지므로, 밝은 organic_only/ad_only
            # 라벨과 달리 흰색 텍스트로 대비를 맞춘다 (venn2 경로의 label_11과 동일 처리).
            ax.text(
                0, 0, str(both),
                ha="center", va="center", fontsize=15, fontweight="bold", color="#ffffff",
            )

        ax.text(left_center[0], -radius - 0.25, "Organic", ha="center", fontsize=10, color=muted_color)
        ax.text(right_center[0], -radius - 0.25, "Ad", ha="center", fontsize=10, color=muted_color)

    ax.set_title(venn_title, fontsize=11, color=text_color)
    fig.tight_layout(pad=0.8)
    return _fig_to_svg(fig)


def _fig_to_svg(fig) -> str:
    """matplotlib Figure를 SVG 문자열로 변환한다 (visualizer.py의 동명 함수와 동일한
    방식 — <svg 태그부터 시작하도록 잘라낸다 — 이지만, 이 파일은 visualizer.py를
    import하지 않고 독립적으로 구현한다)."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    idx = svg.find("<svg")
    return svg[idx:] if idx != -1 else svg


def render_organic_ad_butterfly_chart(dataset: dict, color_map: dict) -> str:
    """
    build_butterfly_dataset() 결과를 좌우 대칭 가로 막대(나비형) SVG로 렌더링한다.

    좌측 서브플롯(organic)은 x축을 반전해 안쪽(중앙)을 향해 막대가 자라도록 하고,
    우측 서브플롯(ad)은 그대로 바깥쪽을 향해 자라게 해 나비 모양을 만든다.
    color_map은 visualizer.build_color_map()이 만드는 딕셔너리(base/dark/muted/title 등
    키)와 동일한 형태를 기대하지만, 이 파일 자체는 visualizer.py를 import하지 않는다.

    Returns
    -------
    str: SVG 문자열 ("<svg"부터 시작). dataset이 비어 있으면 빈 문자열.
    """
    if not dataset:
        return ""

    labels_left = dataset.get("labels_left") or []
    labels_right = dataset.get("labels_right") or []
    series = dataset.get("series") or []
    if not labels_left and not labels_right:
        return ""

    left_vals = next((s.get("data") or [] for s in series if s.get("name") == "organic"), [])
    right_vals = next((s.get("data") or [] for s in series if s.get("name") == "ad"), [])

    n = max(len(labels_left), len(labels_right), 1)

    left_color = color_map.get("base", "#4e73df")
    right_color = color_map.get("dark", "#2e59d9")
    muted_color = color_map.get("muted", "#666666")
    title_color = color_map.get("title", "#111111")

    fig, (ax_left, ax_right) = plt.subplots(
        ncols=2, figsize=(9.0, max(3.0, 0.42 * n)), sharey=False
    )

    ax_left.barh(range(len(left_vals)), left_vals, color=left_color)
    ax_left.invert_xaxis()
    ax_left.invert_yaxis()
    ax_left.set_yticks(range(len(labels_left)))
    ax_left.set_yticklabels(labels_left, fontsize=8.5)
    ax_left.set_title("Organic", fontsize=10, color=muted_color)

    ax_right.barh(range(len(right_vals)), right_vals, color=right_color)
    ax_right.invert_yaxis()
    ax_right.set_yticks(range(len(labels_right)))
    ax_right.set_yticklabels(labels_right, fontsize=8.5)
    ax_right.yaxis.tick_right()
    ax_right.set_title("Ad", fontsize=10, color=muted_color)

    for ax in (ax_left, ax_right):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(colors=muted_color, labelsize=8.5)

    title = dataset.get("title") or ""
    if title:
        fig.suptitle(title, fontsize=11, color=title_color)

    fig.tight_layout(pad=0.8)
    return _fig_to_svg(fig)


# ──────────────────────────────────────────────────────────────────────────
# 전체 브랜드 순회 CLI
# ──────────────────────────────────────────────────────────────────────────
def run_all(period: str, engine, top_n: int = 10, output_dir: str = "outputs") -> None:
    """
    이 period에 대해 전체 브랜드 순회 산출물을 일괄 생성한다.

    1. get_organic_ad_pairs(period, engine)로 페어 데이터 조회.
    2. compute_correlations(df)로 전체 브랜드x지표 상관계수 계산 후
       {output_dir}/{period}/correlation_summary.csv 저장 (skip된 것 포함 전체).
    3. status == 'ok'인 (brand_name, metric) 조합만 순회하며:
       - build_butterfly_dataset + render_organic_ad_butterfly_chart
         -> {output_dir}/{period}/{brand}_{metric}_butterfly.svg
       - build_dominance_table + render_dominance_venn
         -> {output_dir}/{period}/{brand}_{metric}_venn.svg
       - build_dominance_table()의 rows -> {output_dir}/{period}/{brand}_{metric}_dominance.csv
    4. 마지막으로 build_html_report(period, output_dir)를 호출해 위 산출물들을
       {output_dir}/{period}/report.html 하나로 묶는다.

    color_map은 build_color_map(THEME_COLOR)로 1회 생성해 모든 렌더링에 재사용한다.
    """
    period_dir = os.path.join(output_dir, period)
    os.makedirs(period_dir, exist_ok=True)

    print(f"[run_all] period={period!r} 데이터 조회 중...")
    df = get_organic_ad_pairs(period, engine)
    print(f"[run_all] rows={len(df)}, brands={df['brand_name'].nunique()}")

    corr = compute_correlations(df)
    summary_path = os.path.join(period_dir, "correlation_summary.csv")
    corr.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[run_all] saved {summary_path}")

    ok_rows = corr[corr["status"] == "ok"].reset_index(drop=True)
    total = len(ok_rows)
    color_map = build_color_map(THEME_COLOR)

    for i, row in enumerate(ok_rows.itertuples(index=False), 1):
        brand, metric = row.brand_name, row.metric
        print(f"[{i}/{total}] {brand} / {metric} 처리 중...")

        ds = build_butterfly_dataset(df, brand, metric, top_n=top_n)
        butterfly_svg = render_organic_ad_butterfly_chart(ds, color_map)
        butterfly_path = os.path.join(period_dir, f"{brand}_{metric}_butterfly.svg")
        with open(butterfly_path, "w", encoding="utf-8") as f:
            f.write(butterfly_svg)

        table = build_dominance_table(df, brand, metric, top_n=top_n)

        venn_svg = render_dominance_venn(table, color_map)
        venn_path = os.path.join(period_dir, f"{brand}_{metric}_venn.svg")
        with open(venn_path, "w", encoding="utf-8") as f:
            f.write(venn_svg)

        dominance_path = os.path.join(period_dir, f"{brand}_{metric}_dominance.csv")
        pd.DataFrame(table["rows"], columns=table["headers"]).to_csv(
            dominance_path, index=False, encoding="utf-8-sig"
        )

    report_path = build_html_report(period, output_dir=output_dir)
    print(f"[run_all] report.html 생성: {report_path}")

    n_files = 1 + total * 3 + 1
    print(f"[run_all] 완료: {total}개 조합, 총 {n_files}개 파일, 경로: {period_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="브랜드별 오가닉 vs 광고 콘텐츠 성과 상관관계 분석. 전체 브랜드 순회 산출물 생성"
    )
    parser.add_argument(
        "--period", required=True, choices=list(PERIOD_TO_MONTHS),
        help="분석 기간 (3m/6m/12m/all)",
    )
    parser.add_argument("--top-n", type=int, default=10, dest="top_n")
    parser.add_argument("--output-dir", default="outputs", dest="output_dir")
    args = parser.parse_args()

    run_all(args.period, get_engine(), top_n=args.top_n, output_dir=args.output_dir)
