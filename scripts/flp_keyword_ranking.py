"""
flp_keyword_ranking.py
=======================
F(빈도) / L(참고 지표) / P(성과) 3축 인스타그램 키워드 순위 파이프라인 — 배포용.

`scripts/flp_keyword_analysis.py`(실험 과정 전체가 담긴 원본, 폐기된 로직 포함)에서
최종 확정된 로직만 추려 독립 실행 가능한 단일 스크립트로 정리한 버전이다. 새 실험은
`flp_keyword_analysis.py`에서 하고, 이 파일은 "지금 확정된 방식으로 결과를 뽑는 것"
전용이다.

지표 정의
---------
F (Frequency, 빈도)
    브랜드 x 키워드 x POS별로, 그 키워드가 등장한 광고 소재(ad_body) 텍스트의
    distinct count. `scripts/processor.py::get_raw_keyword_performance()`의
    `HAVING COUNT(DISTINCT ad_body) >= 3` 프로덕션 CTR 로직과 동일한 단위다.
    순수 오가닉(광고 미연결) 콘텐츠의 캡션에서만 등장하는 키워드는 ad_body 자체가
    없어 F=0이 된다 — 의도된 동작이다.

L (참고 지표, 게이트 겸 백필 정렬 기준)
    브랜드별로 "오가닉 top-N 콘텐츠"(follows 기준)와 "광고 top-N"(분석 기간 내
    instagram_profile_follows 합산 기준, 개별 ad_id 단위)을 판정해, 콘텐츠 단위
    점수(둘 다 해당=2, 하나만=1, 둘 다 아님=0)를 매긴다. 키워드의 L =
    (등장 콘텐츠 점수 합) / (등장 distinct content_id 수) — 콘텐츠 1건당 평균 점수.

P (Performance, 성과)
    raw_P = SUM(등장 콘텐츠 follows) / SUM(등장 콘텐츠 reach). EB 스무딩은 쓰지
    않는다 — F_gate+L_gate를 통과한 키워드는 이미 최소 표본이 보장돼 EB 유무가
    최종 순위에 주는 영향이 미미함을 확인했다.

게이트 + 정렬
-------------
1. F_gate: F >= CONFIG['F_THRESHOLD'].
2. L_gate: L >= 브랜드x POS별 L의 median (F_gate 통과 키워드 기준, 이상 포함).
3. F_gate AND L_gate를 모두 통과한 키워드를 raw_P 내림차순 -> L 내림차순 -> keyword
   가나다순으로 정렬한다 (`sort_basis='P'`). 뒤 두 기준은 raw_P가 완전히 동률일 때
   재실행해도 항상 같은 순서가 나오도록 하는 타이브레이크용이다.
4. 그래도 CONFIG['TOP_N_KEYWORDS']에 못 미치면, F_gate만 통과·L_gate 탈락 키워드를
   L 내림차순 -> raw_P 내림차순 -> keyword 가나다순으로 부족분만큼 채운다
   (`sort_basis='L_backfill'`, `*` 표시).
5. 그래도 부족하면(F_gate 통과 표본 자체가 모자란 경우) 있는 만큼만 표시한다.

TOP_N 경계선 동점자 안내
------------------------
TOP_N번째 키워드와 (raw_P, L) 값이 완전히 동일한 키워드가 순위표 밖에도 있으면
(예: 11위 이후에 10위와 완전 동률인 키워드가 더 있는 경우), `find_top_n_boundary_ties()`가
"XX위 키워드와 동일한 F/L/P 값을 가진 키워드가 순위 밖에 N개 더 있습니다"라는 안내를
콘솔에 출력한다. `--show-ties`를 켜면 그 동점 키워드들의 F/L/raw_P 상세까지 콘솔에
추가로 출력한다(둘 다 콘솔 전용, 결과 파일에는 안 들어감).

`summarize_boundary_ties()`는 여기서 한 단계 더 나아가, 동점 키워드들의 등장
content_id가 완전히 겹치는지(구조적 동점 — 반복되는 정형 문구 등 같은 콘텐츠에서
함께 나온 경우, 예: 임상시험 고지 문구 하나가 여러 키워드를 동시에 만들어낸 케이스)
아닌지(독립적 동점 — 서로 다른 콘텐츠에서 우연히 같은 F/L/raw_P가 나온 경우)를
판별해 문장으로 만든다. CLI 실행 시 이 문장들은 결과 CSV와 별도로
`{output_stem}_ties.txt` 파일에 항상 저장된다(동점이 없으면 "동점 없음" 한 줄만
저장) — TSV 표 안에 섞으면 tab 구분이 깨지므로 Notion에는 텍스트 블록으로 따로
붙여넣는 편이 안전하다. 모듈로 import해서 `find_top_n_boundary_ties()`나
`summarize_boundary_ties()`를 직접 호출해도 안내 메시지는 항상 출력되고, 결과는
DataFrame으로 함께 반환된다.

사용법
------
CLI로 바로 실행(기간 3/6/12개월 등, 자유 지정 가능):
    python scripts/flp_keyword_ranking.py --months 3
    python scripts/flp_keyword_ranking.py --months 6 --basis content_upload
    python scripts/flp_keyword_ranking.py --months 12 --output my_ranking.csv
    python scripts/flp_keyword_ranking.py --months 3 --show-ties

또는 모듈로 import해서 사용:
    from scripts.flp_keyword_ranking import CONFIG, run_flp_pipeline
    from scripts.db_connector import get_engine

    result = run_flp_pipeline(get_engine(), {**CONFIG, "ANALYSIS_PERIOD_MONTHS": 3})
    ranking_df = result["ranking_df"]

필요 패키지
-----------
pandas, numpy, python-dateutil, kiwipiepy(형태소 분석), SQLAlchemy, psycopg2-binary
(PostgreSQL 드라이버, DB_URL에 맞춰 필요). `scripts/db_connector.py`가 `.env`의
DB_URL을 읽을 때 python-dotenv가 있으면 자동 사용하지만 필수는 아니다.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from kiwipiepy import Kiwi
from sqlalchemy import text

try:
    from scripts.db_connector import get_engine
except ImportError:
    from db_connector import get_engine

_kiwi = Kiwi()

# POS 태그 분류 기준.
NOUN_TAGS = {"NNG", "NNP"}
VERB_ADJ_TAGS = {"VA", "VV"}
VALID_KEYWORD_TAGS = NOUN_TAGS | VERB_ADJ_TAGS

# 형태소 분석 결과에서 키워드로 채택할 최소 글자 수 (조사/어미 파편 등 잡음 제거).
MIN_KEYWORD_LENGTH = 2

# 캡션/훅텍스트를 하나의 문자열로 합칠 때 쓰는 구분자 (다시 나눠 형태소 분석할 때 기준).
TEXT_SEGMENT_DELIMITER = "¶¶"

# 콘텐츠 필터 대상 미디어 타입 (캡션 기반 키워드 분석 대상만 — VIDEO/REEL 제외).
TARGET_MEDIA_TYPES = ("CAROUSEL_ALBUM", "IMAGE")

# 디파트 캠페인명 접두사.
DEPART_CAMPAIGN_PREFIX = "[디파트]"

# 기간 필터 기준값으로 허용되는 값.
#   'content_upload' : content_date(콘텐츠 업로드일)만 기준으로 필터링.
#   'ad_activity'    : content_date OR 연결 광고의 fb_created_time 중 하나라도
#                       기간 내면 포함 — "이번 기간에 타겟에게 실제로 도달한 소재"
#                       관점에 맞아 기본값으로 쓴다.
PERIOD_FILTER_BASES = ("content_upload", "ad_activity")


# ══════════════════════════════════════════════════════════════════════════
# CONFIG — 하드코딩 금지, 값 변경이 필요하면 아래 딕셔너리(또는 CLI 인자)만 수정한다.
# ══════════════════════════════════════════════════════════════════════════
CONFIG = {
    # 분석 기준 기간(개월). 분석 종료일 기준 N개월 역산해 데이터 필터링. 자유 지정 가능.
    "ANALYSIS_PERIOD_MONTHS": 3,
    # 기간 필터 기준. 'content_upload' 또는 'ad_activity'.
    "PERIOD_FILTER_BASIS": "ad_activity",
    # F 게이트 통과 기준 (키워드가 등장한 ad_body distinct count). 미만이면 표본이
    # 너무 적어 우연에 의한 등장일 가능성이 크다.
    "F_THRESHOLD": 3,
    # 최종 순위표에 노출할 키워드 개수 (POS x 브랜드별). --top-n으로 덮어쓴다.
    "TOP_N_KEYWORDS": 10,
    # L 계산용 브랜드별 "상위권" 판정 기준 — 오가닉 top-N 콘텐츠(follows 기준)와
    # 광고 top-N(개별 ad_id 단위, 기간 내 instagram_profile_follows 합산 기준)
    # 둘 다 이 값을 공유한다. TOP_N_KEYWORDS(최종 순위표 노출 개수)와는 다른
    # 파라미터 — CLI에서는 --l-top-n으로 덮어쓴다.
    "L_TOP_N": 10,
    # F/L/P 최종 분석에서 제외할 브랜드.
    #   'DepartCreative' : 채용 공고/내부 테스트 계정으로 확인됨 — "타겟에게 소재가
    #                      잘 전달됐는지" 검증이라는 분석 목적과 무관.
    "EXCLUDED_BRANDS": ("DepartCreative",),
}


def apply_brand_exclusions(df: pd.DataFrame, exclude: tuple[str, ...]) -> pd.DataFrame:
    """brand_name 컬럼이 있는 DataFrame에서 exclude에 해당하는 브랜드를 제외한다."""
    return df[~df["brand_name"].isin(exclude)].reset_index(drop=True)


def to_notion_tsv(df: pd.DataFrame) -> str:
    """DataFrame을 Notion 붙여넣기용 탭 구분 텍스트로 변환한다."""
    return df.to_csv(sep="\t", index=False)


# ══════════════════════════════════════════════════════════════════════════
# 데이터 수집 (콘텐츠 + 광고, 디파트 판별, ad_id_list 보존)
# ══════════════════════════════════════════════════════════════════════════
def get_period_bounds(months: int, end_date: date | None = None) -> tuple[date, date]:
    """분석 종료일(기본값 오늘) 기준 N개월 역산한 (시작일, 종료일)을 반환한다."""
    end = end_date or date.today()
    start = end - relativedelta(months=months)
    return start, end


def fetch_organic_content_base(engine) -> pd.DataFrame:
    """
    오가닉 콘텐츠 기준 데이터 (기간/브랜드 필터 없음). ig_content_insights는 콘텐츠당
    여러 스냅샷이 쌓이므로 콘텐츠별 최신 스냅샷(follows/reach)만 사용한다.

    Returns
    -------
    DataFrame: content_id, fb_ig_media_id, caption, content_date,
               brand_name, organic_follows, organic_reach
    """
    query = """
        WITH latest_insights AS (
            SELECT DISTINCT ON (content_id) content_id, follows, reach
            FROM ig_content_insights
            ORDER BY content_id, as_of_date DESC
        )
        SELECT
            ic.id             AS content_id,
            ic.fb_ig_media_id AS fb_ig_media_id,
            ic.caption        AS caption,
            ic.ig_timestamp   AS content_date,
            bp.business_name  AS brand_name,
            li.follows        AS organic_follows,
            li.reach          AS organic_reach
        FROM ig_contents ic
        JOIN ig_accounts ia          ON ic.ig_id = ia.id
        JOIN business_portfolios bp  ON ia.business_portfolio_id = bp.id
        LEFT JOIN latest_insights li ON li.content_id = ic.id
        WHERE ic.ig_media_type = ANY(:media_types)
    """
    with engine.connect() as conn:
        return pd.read_sql(
            text(query), conn, params={"media_types": list(TARGET_MEDIA_TYPES)}
        )


def fetch_content_ad_links(engine) -> pd.DataFrame:
    """
    콘텐츠(fb_ig_media_id)에 연결된 광고(ad_id) 단위 데이터. ad_id별 reach/
    instagram_profile_follows는 ad_performance_daily 전체 기간 합산치다.

    Returns
    -------
    DataFrame: content_id, ad_id, ad_body, ad_fb_created_time, campaign_name,
               ad_reach_sum, ad_follows_sum
    """
    query = """
        WITH ad_perf AS (
            SELECT
                ad_id,
                SUM(reach)                      AS ad_reach_sum,
                SUM(instagram_profile_follows)  AS ad_follows_sum
            FROM ad_performance_daily
            GROUP BY ad_id
        )
        SELECT
            ic.id                AS content_id,
            a.id                 AS ad_id,
            a.body                AS ad_body,
            a.fb_created_time     AS ad_fb_created_time,
            c.name                AS campaign_name,
            COALESCE(ap.ad_reach_sum, 0)   AS ad_reach_sum,
            COALESCE(ap.ad_follows_sum, 0) AS ad_follows_sum
        FROM ig_contents ic
        JOIN ads a            ON a.source_ig_media_id = ic.fb_ig_media_id
        LEFT JOIN ad_sets aset ON a.ad_set_id = aset.id
        LEFT JOIN campaigns c  ON aset.campaign_id = c.id
        LEFT JOIN ad_perf ap   ON ap.ad_id = a.id
        WHERE ic.ig_media_type = ANY(:media_types)
    """
    with engine.connect() as conn:
        return pd.read_sql(
            text(query), conn, params={"media_types": list(TARGET_MEDIA_TYPES)}
        )


def _build_content_text(caption, hook_bodies: list[str]) -> str:
    """caption + 광고 훅텍스트(ads.body, 중복 제거)를 TEXT_SEGMENT_DELIMITER로 이어붙인다."""
    parts = []
    if isinstance(caption, str) and caption.strip():
        parts.append(caption.strip())
    parts.extend(hook_bodies)
    return TEXT_SEGMENT_DELIMITER.join(parts)


def build_depart_content_scope(engine) -> pd.DataFrame:
    """
    디파트 콘텐츠 스코프 전체를 콘텐츠 단위로 구성한다 (기간 필터는 적용하지 않음 —
    filter_by_period()에서 별도로 적용해 기간을 바꿔가며 재사용할 수 있게 한다).

    디파트 콘텐츠 판별: campaigns.name이 '[디파트]'로 시작 OR 브랜드명에 'depart' 포함.
    콘텐츠별 ad_id_list/ad_dates_list를 보존해 기간 필터링과 추후 타겟별 조인에 활용한다.

    Returns
    -------
    DataFrame: content_id, brand_name, ad_id_list, ad_dates_list, content_text,
               total_follows, total_reach, content_date
    """
    organic = fetch_organic_content_base(engine)
    ad_links = fetch_content_ad_links(engine)

    ad_links = ad_links.copy()
    ad_links["is_depart_campaign"] = (
        ad_links["campaign_name"].fillna("").str.startswith(DEPART_CAMPAIGN_PREFIX)
    )

    ad_grouped = ad_links.groupby("content_id").agg(
        ad_id_list=("ad_id", lambda s: sorted({int(x) for x in s})),
        ad_dates_list=("ad_fb_created_time", lambda s: [d for d in s if pd.notna(d)]),
        hook_bodies=(
            "ad_body",
            lambda s: list(dict.fromkeys(
                b.strip() for b in s if isinstance(b, str) and b.strip()
            )),
        ),
        ad_reach_sum=("ad_reach_sum", "sum"),
        ad_follows_sum=("ad_follows_sum", "sum"),
        any_depart_campaign=("is_depart_campaign", "any"),
    ).reset_index()

    merged = organic.merge(ad_grouped, on="content_id", how="left")

    for list_col in ("ad_id_list", "ad_dates_list", "hook_bodies"):
        merged[list_col] = merged[list_col].apply(
            lambda x: x if isinstance(x, list) else []
        )
    merged["ad_reach_sum"] = merged["ad_reach_sum"].fillna(0)
    merged["ad_follows_sum"] = merged["ad_follows_sum"].fillna(0)
    merged["any_depart_campaign"] = merged["any_depart_campaign"].fillna(False)

    merged["is_depart_content"] = (
        merged["brand_name"].str.contains("depart", case=False, na=False)
        | merged["any_depart_campaign"]
    )
    scope = merged[merged["is_depart_content"]].copy()

    scope["total_follows"] = scope["organic_follows"].fillna(0) + scope["ad_follows_sum"]
    scope["total_reach"] = scope["organic_reach"].fillna(0) + scope["ad_reach_sum"]
    scope["content_text"] = scope.apply(
        lambda row: _build_content_text(row["caption"], row["hook_bodies"]), axis=1
    )

    return scope[[
        "content_id", "brand_name", "ad_id_list", "ad_dates_list", "content_text",
        "total_follows", "total_reach", "content_date",
    ]].reset_index(drop=True)


def filter_by_period(
    df: pd.DataFrame,
    months: int,
    end_date: date | None = None,
    basis: str = "ad_activity",
) -> tuple[pd.DataFrame, date, date]:
    """
    콘텐츠를 [시작일, 종료일] 기간 기준으로 필터링한다.

    basis='content_upload'면 content_date만, 'ad_activity'면 content_date OR
    ad_dates_list 중 하나라도 기간 내면 포함한다.

    Returns
    -------
    (filtered_df, start_date, end_date)
    """
    if basis not in PERIOD_FILTER_BASES:
        raise ValueError(f"basis는 {PERIOD_FILTER_BASES} 중 하나여야 합니다: {basis!r}")

    start, end = get_period_bounds(months, end_date)

    def _in_period(row) -> bool:
        cd = row["content_date"]
        if pd.notna(cd) and start <= cd.date() <= end:
            return True
        if basis == "ad_activity":
            return any(start <= d.date() <= end for d in row["ad_dates_list"])
        return False

    mask = df.apply(_in_period, axis=1)
    return df[mask].copy(), start, end


# Step 2 최종 산출물 컬럼 — ad_dates_list는 기간 필터용 중간 컬럼이라 최종 출력에서는 뺀다.
CONTENT_DATASET_COLUMNS = [
    "content_id", "brand_name", "ad_id_list", "content_text",
    "total_follows", "total_reach", "content_date",
]


def build_content_level_dataset(engine, config: dict) -> pd.DataFrame:
    """
    콘텐츠 수집 전체 파이프라인: 스코프 구성 -> 기간 필터 적용 -> 최종 컬럼만 반환.

    config['ANALYSIS_PERIOD_MONTHS'], config['PERIOD_FILTER_BASIS']만 바꿔서 다시
    호출하면 다른 기간/필터 기준으로 재실행된다.
    """
    scope = build_depart_content_scope(engine)
    filtered, _, _ = filter_by_period(
        scope,
        config["ANALYSIS_PERIOD_MONTHS"],
        basis=config.get("PERIOD_FILTER_BASIS", "ad_activity"),
    )
    return filtered[CONTENT_DATASET_COLUMNS].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# 키워드 매핑 (Kiwi 형태소 분석, noun/verb_adj 분류)
# ══════════════════════════════════════════════════════════════════════════
def split_text_segments(content_text: str) -> list[str]:
    """TEXT_SEGMENT_DELIMITER(¶¶) 기준으로 콘텐츠 텍스트를 세그먼트로 분할한다."""
    if not isinstance(content_text, str) or not content_text.strip():
        return []
    return [seg.strip() for seg in content_text.split(TEXT_SEGMENT_DELIMITER) if seg.strip()]


def extract_keywords_from_text(text_segment: str) -> list[tuple[str, str]]:
    """
    텍스트 세그먼트 하나를 Kiwi로 형태소 분석해 (키워드, pos) 리스트를 반환한다.

    - pos == 'noun'     : 형태소 태그가 NNG/NNP (명사)
    - pos == 'verb_adj' : 형태소 태그가 VA/VV (형용사/동사) — 사전형으로 읽히도록
                           어간 뒤에 '다'를 붙여 반환한다.
    MIN_KEYWORD_LENGTH(2자) 미만인 형태는 제외한다.
    """
    if not text_segment:
        return []

    results = []
    for token in _kiwi.tokenize(text_segment):
        if token.tag not in VALID_KEYWORD_TAGS:
            continue
        form = token.form.strip()
        if len(form) < MIN_KEYWORD_LENGTH:
            continue
        if token.tag in NOUN_TAGS:
            results.append((form, "noun"))
        else:
            results.append((f"{form}다", "verb_adj"))
    return results


def build_content_keyword_pos_table(content_df: pd.DataFrame) -> pd.DataFrame:
    """
    content_df의 content_text를 세그먼트 분할 + Kiwi 형태소 분석해 content-keyword-POS
    매핑 테이블(long format)을 만든다. 같은 콘텐츠 내 동일 (keyword, pos) 중복은 dedup.

    Returns
    -------
    DataFrame: content_id, brand_name, keyword, pos
    """
    rows = []
    for row in content_df.itertuples(index=False):
        seen_in_content: set[tuple[str, str]] = set()
        for segment in split_text_segments(row.content_text):
            for keyword, pos in extract_keywords_from_text(segment):
                seen_in_content.add((keyword, pos))
        for keyword, pos in seen_in_content:
            rows.append({
                "content_id": row.content_id,
                "brand_name": row.brand_name,
                "keyword": keyword,
                "pos": pos,
            })
    return pd.DataFrame(rows, columns=["content_id", "brand_name", "keyword", "pos"])


# ══════════════════════════════════════════════════════════════════════════
# F (Frequency) 계산 — ad_body distinct count
# ══════════════════════════════════════════════════════════════════════════
def compute_keyword_frequency(engine, keyword_pos_df: pd.DataFrame, content_df: pd.DataFrame) -> pd.DataFrame:
    """
    브랜드 x 키워드 x POS 기준 F(빈도)를 계산한다.

    F = 키워드가 등장한 ad_body(광고 소재 텍스트)의 distinct count —
    `scripts/processor.py::get_raw_keyword_performance()`의 프로덕션 CTR 로직
    (`HAVING COUNT(DISTINCT ad_body) >= 3`)과 동일한 단위다. 같은 ad_body가 여러
    캠페인 목적으로 중복 집행돼도 1개로 카운트한다. 순수 오가닉 콘텐츠의 캡션에서만
    등장하는 키워드는 ad_body가 없어 F=0이 된다.

    keyword_pos_df(콘텐츠 전체 텍스트 기준 키워드 카탈로그)는 브랜드에 존재하는
    키워드의 완전한 목록을 얻는 데만 쓰고, 실제 F 값은 ad_body 텍스트를 독립적으로
    다시 형태소 분석해 계산한다.

    Returns
    -------
    DataFrame: brand_name, keyword, pos, F, ad_id_list (F=0인 행은 ad_id_list=[])
    """
    ad_links = fetch_content_ad_links(engine)
    scope_content_ids = set(content_df["content_id"])
    ad_links = ad_links[ad_links["content_id"].isin(scope_content_ids)]
    ad_links = ad_links.dropna(subset=["ad_body"])
    ad_links = ad_links[ad_links["ad_body"].str.strip() != ""]

    brand_map = content_df[["content_id", "brand_name"]].drop_duplicates()
    ad_links = ad_links.merge(brand_map, on="content_id", how="left")

    unique_bodies = ad_links["ad_body"].drop_duplicates()
    body_kw_rows = []
    for body in unique_bodies:
        for keyword, pos in extract_keywords_from_text(body.strip()):
            body_kw_rows.append({"ad_body": body, "keyword": keyword, "pos": pos})
    body_kw_df = pd.DataFrame(
        body_kw_rows, columns=["ad_body", "keyword", "pos"]
    ).drop_duplicates()

    merged = ad_links.merge(body_kw_df, on="ad_body", how="inner")

    f_counts = (
        merged.groupby(["brand_name", "keyword", "pos"])["ad_body"]
        .nunique()
        .rename("F")
        .reset_index()
    )
    ad_id_union = (
        merged.groupby(["brand_name", "keyword", "pos"])["ad_id"]
        .apply(lambda s: sorted({int(x) for x in s}))
        .rename("ad_id_list")
        .reset_index()
    )
    f_by_ad = f_counts.merge(ad_id_union, on=["brand_name", "keyword", "pos"], how="left")

    catalog = keyword_pos_df[["brand_name", "keyword", "pos"]].drop_duplicates()
    result = catalog.merge(f_by_ad, on=["brand_name", "keyword", "pos"], how="left")
    result["F"] = result["F"].fillna(0).astype(int)
    result["ad_id_list"] = result["ad_id_list"].apply(lambda x: x if isinstance(x, list) else [])

    return result.sort_values(
        ["brand_name", "pos", "F"], ascending=[True, True, False]
    ).reset_index(drop=True)


def apply_f_gate(df: pd.DataFrame, f_threshold: int) -> pd.DataFrame:
    """F >= f_threshold 여부를 F_gate 컬럼으로 추가한다."""
    result = df.copy()
    result["F_gate"] = result["F"] >= f_threshold
    return result


# ══════════════════════════════════════════════════════════════════════════
# L (참고 지표) 계산 — 브랜드별 오가닉/광고 top-N 콘텐츠 점수의 콘텐츠당 평균
# ══════════════════════════════════════════════════════════════════════════
def compute_organic_top_n_flags(engine, content_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    브랜드별로 콘텐츠를 오가닉 follows(ig_content_insights 최신 스냅샷) 내림차순
    정렬해 상위 top_n을 organic_top_n_flag=True로 표시한다.

    Returns
    -------
    DataFrame: content_id, brand_name, organic_follows, organic_rank, organic_top_n_flag
    """
    organic_raw = fetch_organic_content_base(engine)
    scope = content_df[["content_id", "brand_name"]].drop_duplicates()
    merged = scope.merge(organic_raw[["content_id", "organic_follows"]], on="content_id", how="left")
    merged["organic_follows"] = merged["organic_follows"].fillna(0)

    merged["organic_rank"] = merged.groupby("brand_name")["organic_follows"].rank(
        method="first", ascending=False
    )
    merged["organic_top_n_flag"] = merged["organic_rank"] <= top_n
    return merged[["content_id", "brand_name", "organic_follows", "organic_rank", "organic_top_n_flag"]]


def compute_ad_top_n_flags(engine, content_df: pd.DataFrame, config: dict, top_n: int) -> pd.DataFrame:
    """
    브랜드별로 "개별 광고"(ad_id, 콘텐츠 단위 합산 아님)를 분석 기간
    (config['ANALYSIS_PERIOD_MONTHS']) 내 instagram_profile_follows 합산 기준
    내림차순 정렬해 상위 top_n을 ad_top_n_flag=True로 표시한다.

    Returns
    -------
    DataFrame: ad_id, brand_name, period_follows, ad_rank, ad_top_n_flag
    """
    start, end = get_period_bounds(config["ANALYSIS_PERIOD_MONTHS"])

    ad_id_to_brand = (
        content_df[["content_id", "brand_name", "ad_id_list"]]
        .explode("ad_id_list")
        .dropna(subset=["ad_id_list"])
        .rename(columns={"ad_id_list": "ad_id"})
    )
    ad_id_to_brand["ad_id"] = ad_id_to_brand["ad_id"].astype(int)
    ad_id_to_brand = ad_id_to_brand.drop_duplicates(subset=["ad_id"])
    ad_ids = sorted(ad_id_to_brand["ad_id"].unique().tolist())

    if not ad_ids:
        return pd.DataFrame(columns=["ad_id", "brand_name", "period_follows", "ad_rank", "ad_top_n_flag"])

    query = text("""
        SELECT ad_id, SUM(instagram_profile_follows) AS period_follows
        FROM ad_performance_daily
        WHERE ad_id = ANY(:ad_ids) AND as_of_date BETWEEN :start AND :end
        GROUP BY ad_id
    """)
    with engine.connect() as conn:
        period_perf = pd.read_sql(query, conn, params={"ad_ids": ad_ids, "start": start, "end": end})

    merged = ad_id_to_brand[["ad_id", "brand_name"]].merge(period_perf, on="ad_id", how="left")
    merged["period_follows"] = merged["period_follows"].fillna(0)

    merged["ad_rank"] = merged.groupby("brand_name")["period_follows"].rank(method="first", ascending=False)
    merged["ad_top_n_flag"] = merged["ad_rank"] <= top_n
    return merged[["ad_id", "brand_name", "period_follows", "ad_rank", "ad_top_n_flag"]]


def compute_content_ad_top_n_flag(content_df: pd.DataFrame, ad_top_n_df: pd.DataFrame) -> pd.DataFrame:
    """콘텐츠별 ad_top_n_flag = 연결된 ad_id_list 중 하나라도 광고 top_n에 포함되면 True."""
    flagged_ad_ids = set(ad_top_n_df.loc[ad_top_n_df["ad_top_n_flag"], "ad_id"])
    result = content_df[["content_id", "ad_id_list"]].copy()
    result["ad_top_n_flag"] = result["ad_id_list"].apply(
        lambda ids: any(a in flagged_ad_ids for a in ids)
    )
    return result[["content_id", "ad_top_n_flag"]]


def compute_content_score(
    organic_top_n_df: pd.DataFrame, content_ad_top_n_df: pd.DataFrame
) -> pd.DataFrame:
    """
    콘텐츠 단위 점수: organic_top_n_flag AND ad_top_n_flag 둘 다 True -> 2,
    하나만 True -> 1, 둘 다 False -> 0.

    Returns
    -------
    DataFrame: content_id, brand_name, organic_top_n_flag, ad_top_n_flag, content_score
    """
    merged = organic_top_n_df[["content_id", "brand_name", "organic_top_n_flag"]].merge(
        content_ad_top_n_df, on="content_id", how="left"
    )
    merged["ad_top_n_flag"] = merged["ad_top_n_flag"].fillna(False)
    merged["content_score"] = np.select(
        [
            merged["organic_top_n_flag"] & merged["ad_top_n_flag"],
            merged["organic_top_n_flag"] | merged["ad_top_n_flag"],
        ],
        [2, 1],
        default=0,
    )
    return merged[["content_id", "brand_name", "organic_top_n_flag", "ad_top_n_flag", "content_score"]]


def compute_keyword_l_score(keyword_pos_df: pd.DataFrame, content_score_df: pd.DataFrame) -> pd.DataFrame:
    """
    브랜드 x 키워드 x POS 기준 L을 계산한다.

    L = L_sum(등장 콘텐츠의 content_score 합산) / F_content(등장 distinct content_id 수)
      — "이 키워드가 등장한 콘텐츠 1건당 평균 점수". F_content는 F_gate에 쓰이는 F
      (ad_body 기준)와 별개로, L의 분모 전용이다.

    Returns
    -------
    DataFrame: brand_name, keyword, pos, F_content, L_sum, L
    """
    merged = keyword_pos_df.merge(
        content_score_df[["content_id", "content_score"]], on="content_id", how="left"
    )
    agg = merged.groupby(["brand_name", "keyword", "pos"]).agg(
        F_content=("content_id", "nunique"),
        L_sum=("content_score", "sum"),
    ).reset_index()
    agg["L"] = (agg["L_sum"] / agg["F_content"]).round(4)
    return agg.sort_values(["brand_name", "pos", "L"], ascending=[True, True, False]).reset_index(drop=True)


def apply_l_gate(keyword_scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    L_gate: L >= 브랜드x POS별 median(F_gate 통과 키워드 기준)을 판정한다 (이상 포함).

    median은 반드시 브랜드x POS 단위로 계산한다 — 브랜드 단위(POS 무관)로 계산하면
    noun/verb_adj 분포가 크게 다른 브랜드에서 한쪽 POS 풀이 거의 붕괴할 수 있다.
    F_gate를 통과하지 못한 키워드는 L_gate 판정 대상이 아니므로 L_gate=False.

    Returns
    -------
    keyword_scored_df + bp_L_median, L_gate 컬럼
    """
    gated = keyword_scored_df[keyword_scored_df["F_gate"]]
    median = gated.groupby(["brand_name", "pos"])["L"].median().rename("bp_L_median")
    result = keyword_scored_df.merge(median, on=["brand_name", "pos"], how="left")
    result["L_gate"] = result["F_gate"] & (result["L"] >= result["bp_L_median"])
    return result


# ══════════════════════════════════════════════════════════════════════════
# P (Performance) 계산
# ══════════════════════════════════════════════════════════════════════════
def compute_keyword_performance(keyword_pos_df: pd.DataFrame, content_df: pd.DataFrame) -> pd.DataFrame:
    """
    브랜드 x 키워드 x POS 기준 raw_P를 계산한다.

    raw_P = SUM(등장 콘텐츠 total_follows) / SUM(등장 콘텐츠 total_reach)

    Returns
    -------
    DataFrame: brand_name, keyword, pos, SUM_follows, SUM_reach, raw_P, ad_id_list
    """
    merged = keyword_pos_df.merge(
        content_df[["content_id", "total_follows", "total_reach", "ad_id_list"]],
        on="content_id",
        how="left",
    )

    agg = merged.groupby(["brand_name", "keyword", "pos"]).agg(
        SUM_follows=("total_follows", "sum"),
        SUM_reach=("total_reach", "sum"),
    ).reset_index()
    agg["raw_P"] = np.where(agg["SUM_reach"] > 0, agg["SUM_follows"] / agg["SUM_reach"], np.nan)

    exploded = merged[["brand_name", "keyword", "pos", "ad_id_list"]].explode("ad_id_list")
    ad_id_union = (
        exploded.dropna(subset=["ad_id_list"])
        .groupby(["brand_name", "keyword", "pos"])["ad_id_list"]
        .apply(lambda s: sorted({int(x) for x in s}))
        .rename("ad_id_list")
    )

    result = agg.merge(ad_id_union, on=["brand_name", "keyword", "pos"], how="left")
    result["ad_id_list"] = result["ad_id_list"].apply(lambda x: x if isinstance(x, list) else [])
    return result[["brand_name", "keyword", "pos", "SUM_follows", "SUM_reach", "raw_P", "ad_id_list"]]


# ══════════════════════════════════════════════════════════════════════════
# 최종 순위표 통합 — F_gate -> L_gate(median) -> raw_P 정렬 -> backfill
# ══════════════════════════════════════════════════════════════════════════
def build_final_ranking(
    keyword_scored_df: pd.DataFrame, p_df: pd.DataFrame, top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    브랜드 x POS별 최종 순위표를 만든다. keyword_scored_df는 apply_l_gate()를 거쳐
    F_gate/L_gate/bp_L_median 컬럼을 가진 상태여야 한다.

    1. F_gate AND L_gate 모두 통과한 키워드 -> raw_P 내림차순, sort_basis='P'.
    2. top_n에 못 미치면 F_gate만 통과·L_gate 탈락 키워드를 L 내림차순으로 부족분만큼
       채운다 -> sort_basis='L_backfill', is_backfill=True(Notion 표시 '*').
    3. 그래도 부족하면(F_gate 통과 표본 자체가 top_n보다 적은 경우) 있는 만큼만
       반환하고 shortage_df에 남긴다 — 게이트 밖 키워드로는 채우지 않는다.

    동점(raw_P 또는 L이 완전히 같은 경우) 발생 시 keyword 가나다순(오름차순)을
    3차 정렬 키로 사용해 재실행해도 항상 같은 순서가 나오도록 한다(결정론성).

    Returns
    -------
    (ranking_df, shortage_df)
      ranking_df  : brand_name, pos, rank, keyword, sort_basis, is_backfill,
                    F, L, raw_P, SUM_follows, SUM_reach, ad_id_list
      shortage_df : brand_name, pos, available_count, note (부족한 조합만 행으로 존재)
    """
    merged = keyword_scored_df[["brand_name", "keyword", "pos", "F", "F_gate", "L", "L_gate"]].merge(
        p_df[["brand_name", "keyword", "pos", "raw_P", "SUM_follows", "SUM_reach", "ad_id_list"]],
        on=["brand_name", "keyword", "pos"],
        how="left",
    )

    rank_blocks = []
    shortage_rows = []

    for (brand, pos), grp in merged[merged["F_gate"]].groupby(["brand_name", "pos"]):
        primary = grp[grp["L_gate"]].sort_values(
            ["raw_P", "L", "keyword"], ascending=[False, False, True]
        ).copy()
        primary["sort_basis"] = "P"
        selected = primary.head(top_n).copy()

        if len(selected) < top_n:
            need = top_n - len(selected)
            backfill = grp[~grp["L_gate"]].sort_values(
                ["L", "raw_P", "keyword"], ascending=[False, False, True]
            ).head(need).copy()
            backfill["sort_basis"] = "L_backfill"
            selected = pd.concat([selected, backfill], ignore_index=True)

        if len(selected) < top_n:
            shortage_rows.append({
                "brand_name": brand,
                "pos": pos,
                "available_count": len(selected),
                "note": f"{brand}/{pos}는 backfill 후에도 {len(selected)}개만 표시됨(F_gate 통과 표본 부족)",
            })

        selected["rank"] = range(1, len(selected) + 1)
        selected["brand_name"] = brand
        selected["pos"] = pos
        rank_blocks.append(selected)

    ranking_df = pd.concat(rank_blocks, ignore_index=True) if rank_blocks else pd.DataFrame()
    ranking_df["is_backfill"] = ranking_df["sort_basis"] == "L_backfill"

    cols = [
        "brand_name", "pos", "rank", "keyword", "sort_basis", "is_backfill",
        "F", "L", "raw_P", "SUM_follows", "SUM_reach", "ad_id_list",
    ]
    ranking_df = ranking_df[cols].sort_values(["brand_name", "pos", "rank"]).reset_index(drop=True)
    shortage_df = pd.DataFrame(
        shortage_rows, columns=["brand_name", "pos", "available_count", "note"]
    )
    return ranking_df, shortage_df


def find_top_n_boundary_ties(
    ranking_df: pd.DataFrame,
    keyword_scored_df: pd.DataFrame,
    p_df: pd.DataFrame,
    top_n: int,
    show_ties: bool = False,
) -> pd.DataFrame:
    """
    TOP_N 경계선 동점자를 탐지하고 안내 메시지를 출력한다.

    브랜드 x POS별로 TOP_N번째(rank == top_n) 키워드의 (raw_P, L) 값과 완전히
    동일한 값을 가진 키워드가 최종 순위표 밖(선발되지 못한 나머지)에도 있는지
    확인한다. 경계 행의 sort_basis가 'P'면 F_gate&L_gate 통과 풀(raw_P 기준 선발)
    에서, 'L_backfill'이면 F_gate만 통과·L_gate 탈락 풀(L 기준 선발)에서 탐색한다
    — build_final_ranking()이 실제로 선발에 쓴 풀과 동일한 풀을 봐야 "정말 순위
    밖으로 밀려난 동점자"만 잡아낸다. backfill로도 top_n을 못 채운 조합(F_gate
    통과 표본 자체가 부족한 경우)은 경계선 개념이 성립하지 않아 대상에서 뺀다.

    동점자가 있는 브랜드x POS 조합마다 안내 메시지를 print한다(모듈로 import해서
    단독 호출해도 항상 출력됨). show_ties=True면 동점 키워드의 F/L/raw_P 상세까지
    추가로 print한다(CLI --show-ties에 대응).

    Returns
    -------
    DataFrame: brand_name, pos, boundary_rank, boundary_keyword, sort_basis,
               tied_keyword, F, L, raw_P (동점 후보가 없는 조합은 행이 없음)
    """
    merged = keyword_scored_df[["brand_name", "keyword", "pos", "F", "F_gate", "L", "L_gate"]].merge(
        p_df[["brand_name", "keyword", "pos", "raw_P"]],
        on=["brand_name", "keyword", "pos"],
        how="left",
    )

    tie_rows = []
    for (brand, pos), grp in ranking_df.groupby(["brand_name", "pos"]):
        if len(grp) < top_n:
            continue
        boundary = grp[grp["rank"] == top_n].iloc[0]
        selected_keywords = set(grp["keyword"])
        pool = merged[(merged["brand_name"] == brand) & (merged["pos"] == pos)]

        if boundary["sort_basis"] == "P":
            candidates = pool[pool["L_gate"] & ~pool["keyword"].isin(selected_keywords)]
        else:
            candidates = pool[pool["F_gate"] & ~pool["L_gate"] & ~pool["keyword"].isin(selected_keywords)]

        tied = candidates[
            (candidates["raw_P"] == boundary["raw_P"]) & (candidates["L"] == boundary["L"])
        ]
        for _, row in tied.iterrows():
            tie_rows.append({
                "brand_name": brand,
                "pos": pos,
                "boundary_rank": top_n,
                "boundary_keyword": boundary["keyword"],
                "sort_basis": boundary["sort_basis"],
                "tied_keyword": row["keyword"],
                "F": row["F"],
                "L": row["L"],
                "raw_P": row["raw_P"],
            })

    tie_df = pd.DataFrame(
        tie_rows,
        columns=[
            "brand_name", "pos", "boundary_rank", "boundary_keyword", "sort_basis",
            "tied_keyword", "F", "L", "raw_P",
        ],
    )

    for (brand, pos), grp in tie_df.groupby(["brand_name", "pos"]):
        boundary_rank = grp["boundary_rank"].iloc[0]
        tied_keywords = grp["tied_keyword"].tolist()
        print(
            f"[{brand}/{pos}]: {boundary_rank}위 키워드와 동일한 F/L/P 값을 가진 키워드가 "
            f"순위 밖에 {len(tied_keywords)}개 더 있습니다. 확인하려면 --show-ties 옵션으로 "
            f"다시 실행하세요. (해당 키워드: {', '.join(tied_keywords)})"
        )
        if show_ties:
            detail = grp[["tied_keyword", "F", "L", "raw_P"]].rename(columns={"tied_keyword": "keyword"})
            print(detail.to_string(index=False))

    return tie_df


def summarize_boundary_ties(
    tie_df: pd.DataFrame, ranking_df: pd.DataFrame, keyword_pos_df: pd.DataFrame
) -> pd.DataFrame:
    """
    find_top_n_boundary_ties()의 tie_df를 결과 파일에 넣을 문장형 안내로 요약한다.

    브랜드x POS 조합별로 (경계 키워드 + 동점 키워드) 전체의 등장 content_id를
    비교해, 전부 완전히 동일한 콘텐츠 집합을 공유하면 "구조적 동점"(반복되는
    정형 문구 등에서 나왔을 가능성), 하나라도 다르면 "독립적 동점"(서로 다른
    콘텐츠에서 우연히 같은 F/L/raw_P가 나온 경우)으로 판별해 문장에 반영한다.

    Returns
    -------
    DataFrame: brand_name, pos, message, is_structural, content_ids
      (tie_df가 비어 있으면 빈 DataFrame)
    """
    if tie_df.empty:
        return pd.DataFrame(columns=["brand_name", "pos", "message", "is_structural", "content_ids"])

    rows = []
    for (brand, pos), grp in tie_df.groupby(["brand_name", "pos"]):
        boundary_rank = grp["boundary_rank"].iloc[0]
        boundary_keyword = grp["boundary_keyword"].iloc[0]
        boundary_row = ranking_df[
            (ranking_df["brand_name"] == brand)
            & (ranking_df["pos"] == pos)
            & (ranking_df["rank"] == boundary_rank)
        ].iloc[0]
        tied_keywords = grp["tied_keyword"].tolist()

        def _content_ids(keyword: str) -> frozenset:
            return frozenset(
                keyword_pos_df[
                    (keyword_pos_df["brand_name"] == brand)
                    & (keyword_pos_df["pos"] == pos)
                    & (keyword_pos_df["keyword"] == keyword)
                ]["content_id"]
            )

        kw_cid_sets = {kw: _content_ids(kw) for kw in [boundary_keyword] + tied_keywords}
        is_structural = len(set(kw_cid_sets.values())) == 1
        union_cids = sorted(set().union(*kw_cid_sets.values()))

        note = (
            "(참고: 이 키워드들은 동일한 콘텐츠에서 함께 등장 — 반복되는 정형 문구일 "
            "가능성이 있으니 확인 권장)"
            if is_structural
            else "(참고: 서로 다른 콘텐츠에서 독립적으로 등장한 동점입니다)"
        )

        message = (
            f"[{brand}/{pos}] {boundary_rank}위 '{boundary_keyword}'와 "
            f"F={boundary_row['F']}·L={boundary_row['L']}·raw_P={boundary_row['raw_P']:.6f}가 동일한 "
            f"키워드가 {len(tied_keywords)}개 더 있습니다: {', '.join(tied_keywords)} "
            f"(등장 콘텐츠: {union_cids}) {note}"
        )

        rows.append({
            "brand_name": brand,
            "pos": pos,
            "message": message,
            "is_structural": is_structural,
            "content_ids": union_cids,
        })

    return pd.DataFrame(rows, columns=["brand_name", "pos", "message", "is_structural", "content_ids"])


def format_ranking_for_notion(ranking_df: pd.DataFrame, pos: str) -> pd.DataFrame:
    """ranking_df를 지정 POS('noun' 또는 'verb_adj')만 골라 Notion 붙여넣기용 표로 만든다."""
    sub = ranking_df[ranking_df["pos"] == pos].copy()
    sub["표시"] = sub["is_backfill"].map({True: "*", False: ""})
    sub = sub.rename(columns={
        "brand_name": "브랜드", "rank": "순위", "keyword": "키워드", "pos": "POS",
    })
    return sub[[
        "브랜드", "순위", "키워드", "POS", "sort_basis", "표시",
        "F", "L", "raw_P", "SUM_follows", "SUM_reach", "ad_id_list",
    ]]


def show_brand_ranking(ranking_df: pd.DataFrame, brand_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    지정한 브랜드 하나의 최종 순위표를 noun/verb_adj로 나눠서 각각 DataFrame으로 반환한다.

    Returns
    -------
    (noun_df, verb_adj_df) — 둘 다 순위(rank) 오름차순 정렬.
    """
    sub = ranking_df[ranking_df["brand_name"] == brand_name].copy()
    sub["표시"] = sub["is_backfill"].map({True: "*", False: ""})
    sub = sub.rename(columns={"rank": "순위", "keyword": "키워드"})
    cols = [
        "순위", "키워드", "sort_basis", "표시",
        "F", "L", "raw_P", "SUM_follows", "SUM_reach", "ad_id_list",
    ]
    noun_df = sub[sub["pos"] == "noun"][cols].sort_values("순위").reset_index(drop=True)
    verb_adj_df = sub[sub["pos"] == "verb_adj"][cols].sort_values("순위").reset_index(drop=True)
    return noun_df, verb_adj_df


# ══════════════════════════════════════════════════════════════════════════
# 파이프라인 오케스트레이션
# ══════════════════════════════════════════════════════════════════════════
def run_flp_pipeline(engine, config: dict) -> dict:
    """
    F/L/P 전체 파이프라인을 config 하나로 한 번에 실행한다.

    config['ANALYSIS_PERIOD_MONTHS'], config['PERIOD_FILTER_BASIS'], config['F_THRESHOLD'],
    config['L_TOP_N'], config['TOP_N_KEYWORDS'],
    config['EXCLUDED_BRANDS']만 바꿔서 호출하면 다른 기간/기준/임계값으로 재실행된다.

    Returns
    -------
    dict: content_df, keyword_pos_df, f_df_gated, organic_top_n_df, ad_top_n_df,
          content_score_df, keyword_scored_df, p_df, ranking_df, shortage_df
    """
    exclude = config.get("EXCLUDED_BRANDS", ())

    content_df = build_content_level_dataset(engine, config)
    content_df = apply_brand_exclusions(content_df, exclude)

    keyword_pos_df = build_content_keyword_pos_table(content_df)
    keyword_pos_df = apply_brand_exclusions(keyword_pos_df, exclude)

    f_df = compute_keyword_frequency(engine, keyword_pos_df, content_df)
    f_df_gated = apply_f_gate(f_df, config["F_THRESHOLD"])

    organic_top_n_df = compute_organic_top_n_flags(engine, content_df, config["L_TOP_N"])
    ad_top_n_df = compute_ad_top_n_flags(engine, content_df, config, config["L_TOP_N"])
    content_ad_top_n_df = compute_content_ad_top_n_flag(content_df, ad_top_n_df)
    content_score_df = compute_content_score(organic_top_n_df, content_ad_top_n_df)
    l_score_df = compute_keyword_l_score(keyword_pos_df, content_score_df)

    keyword_scored_df = f_df_gated.merge(
        l_score_df[["brand_name", "keyword", "pos", "L"]],
        on=["brand_name", "keyword", "pos"],
        how="left",
    )
    keyword_scored_df = apply_l_gate(keyword_scored_df)

    gated_keyword_pos_df = keyword_pos_df.merge(
        f_df_gated.loc[f_df_gated["F_gate"], ["brand_name", "keyword", "pos"]],
        on=["brand_name", "keyword", "pos"],
        how="inner",
    )
    p_df = compute_keyword_performance(gated_keyword_pos_df, content_df)

    ranking_df, shortage_df = build_final_ranking(keyword_scored_df, p_df, config["TOP_N_KEYWORDS"])

    return {
        "content_df": content_df,
        "keyword_pos_df": keyword_pos_df,
        "f_df_gated": f_df_gated,
        "organic_top_n_df": organic_top_n_df,
        "ad_top_n_df": ad_top_n_df,
        "content_score_df": content_score_df,
        "keyword_scored_df": keyword_scored_df,
        "p_df": p_df,
        "ranking_df": ranking_df,
        "shortage_df": shortage_df,
    }


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="F/L/P 키워드 순위 파이프라인 실행 (브랜드 x POS별 top-N 키워드 순위표 생성)",
    )
    parser.add_argument(
        "--months", type=int, default=CONFIG["ANALYSIS_PERIOD_MONTHS"],
        help=f"분석 기준 기간(개월), 기본값 {CONFIG['ANALYSIS_PERIOD_MONTHS']}",
    )
    parser.add_argument(
        "--basis", choices=PERIOD_FILTER_BASES, default=CONFIG["PERIOD_FILTER_BASIS"],
        help=f"기간 필터 기준, 기본값 {CONFIG['PERIOD_FILTER_BASIS']!r}",
    )
    parser.add_argument(
        "--top-n", type=int, default=CONFIG["TOP_N_KEYWORDS"], dest="top_n",
        help=f"최종 순위표에 노출할 키워드 개수(브랜드x POS별), 기본값 {CONFIG['TOP_N_KEYWORDS']}",
    )
    parser.add_argument(
        "--l-top-n", type=int, default=CONFIG["L_TOP_N"], dest="l_top_n",
        help=(
            "L 점수 계산 시 오가닉/광고 각각 상위 몇 개 콘텐츠·광고를 '상위권'으로 "
            f"볼지 결정하는 기준, 기본값 {CONFIG['L_TOP_N']}. --top-n(최종 순위표 "
            "노출 개수)과는 다른 파라미터이니 혼동하지 말 것 — 이 값은 L_gate의 "
            "median이 어떤 풀에서 계산되는지에 영향을 줄 뿐, 순위표 행 수와는 무관하다."
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help="결과를 저장할 CSV 경로 (생략하면 flp_ranking_{months}m.csv)",
    )
    parser.add_argument(
        "--show-ties", action="store_true", dest="show_ties",
        help="TOP_N 경계선 동점 키워드가 있으면 F/L/raw_P 상세까지 함께 출력",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = {
        **CONFIG,
        "ANALYSIS_PERIOD_MONTHS": args.months,
        "PERIOD_FILTER_BASIS": args.basis,
        "TOP_N_KEYWORDS": args.top_n,
        "L_TOP_N": args.l_top_n,
    }

    print(
        f"[flp_keyword_ranking] 기간={args.months}개월, 기준={args.basis!r}, "
        f"top_n={args.top_n}, l_top_n={args.l_top_n} 실행 중..."
    )
    engine = get_engine()
    result = run_flp_pipeline(engine, config)

    ranking_df = result["ranking_df"]
    shortage_df = result["shortage_df"]

    print(f"[flp_keyword_ranking] 콘텐츠 {len(result['content_df'])}건, "
          f"브랜드 {result['content_df']['brand_name'].nunique()}개")
    print(f"[flp_keyword_ranking] 최종 순위표 {len(ranking_df)}행 "
          f"(backfill 발동 {int(ranking_df['is_backfill'].sum())}건)")
    if not shortage_df.empty:
        print(f"[flp_keyword_ranking] {args.top_n}개 미달 조합 {len(shortage_df)}개:")
        for note in shortage_df["note"]:
            print(f"  - {note}")

    tie_df = find_top_n_boundary_ties(
        ranking_df, result["keyword_scored_df"], result["p_df"], args.top_n, show_ties=args.show_ties
    )
    tie_summary_df = summarize_boundary_ties(tie_df, ranking_df, result["keyword_pos_df"])

    output_path = args.output or f"flp_ranking_{args.months}m.csv"
    ranking_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[flp_keyword_ranking] 결과 저장: {output_path}")

    ties_path = Path(output_path).with_name(Path(output_path).stem + "_ties.txt")
    if tie_summary_df.empty:
        ties_path.write_text("TOP_N 경계선 동점 없음.\n", encoding="utf-8")
    else:
        ties_path.write_text("\n".join(tie_summary_df["message"]) + "\n", encoding="utf-8")
    print(f"[flp_keyword_ranking] 동점 안내 저장: {ties_path}")


if __name__ == "__main__":
    main()
