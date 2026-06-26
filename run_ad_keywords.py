import os
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / "scripts" / ".env")
except Exception:
    pass

import pg8000.native

import nltk
_nltk_data = os.environ.get("NLTK_DATA")
if _nltk_data:
    nltk.data.path.insert(0, str(Path(_nltk_data).expanduser()))
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from kiwipiepy import Kiwi


def _wn_pos(treebank_tag: str):
    """Penn Treebank POS 태그 → WordNet POS 변환"""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


_ENGLISH_FUNCTION_WORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "with", "without",
    "for", "from", "to", "of", "in", "on", "at", "by", "as", "into", "over", "under", "about",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "it", "its",
    "you", "your", "yours", "we", "our", "ours", "us",
    "they", "their", "theirs",
    "he", "she", "him", "her", "his",
    "me", "my",
    "do", "did", "does",
}
_KIWI_EXCLUDE_PREFIXES = ("J", "E")
# 인접 토큰 병합 시도 시 두 번째 토큰(t_next)이 이 태그 집합에 속해야만 병합 가능.
_MERGE_NEXT_ALLOWED_TAGS = {"NNG", "NNP", "NNB", "IC", "XR", "XSN", "SL"}
_KIWI_EXCLUDE_TAGS = {
    "SF", "SP", "SS", "SE", "SO", "SW",
    "EC", "EF", "EP", "ETM", "ETN",
    "IC",  # 감탄사 — '아우터'→'아우'(IC)+'터' 처럼 외래어가 부분 추출되는 오류 방지
    "SL",  # 외국어(영어) → _extract_english 단계에서 lemmatization 포함 처리
    "SN",  # 숫자
}


def env_bool(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "y", "on")


def normalize_keyword(keyword: str) -> str:
    if not keyword:
        return ""
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", str(keyword)).lower()


def to_int_or_none(x):
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def build_in_clause_params(values, prefix="p"):
    params = {}
    keys = []
    for i, v in enumerate(values):
        k = f"{prefix}{i}"
        params[k] = v
        keys.append(f":{k}")
    clause = "(" + ",".join(keys) + ")" if keys else "(NULL)"
    return clause, params


_HASHTAG_TOKEN_RE = re.compile(r"#([^\s#]+)")

def remove_or_keep_hashtags(text: str) -> str:
    if not text:
        return text

    out = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] != "#":
            out.append(text[i])
            i += 1
            continue

        j = i
        hashtags = []

        while j < n:
            while j < n and text[j].isspace():
                j += 1
            if j >= n or text[j] != "#":
                break

            m = _HASHTAG_TOKEN_RE.match(text, j)
            if not m:
                break
            token = m.group(1)
            hashtags.append(token)
            j = m.end()

        if len(hashtags) >= 2:
            i = j
            out.append(" ")
        elif len(hashtags) == 1:
            out.append(hashtags[0])
            i = j
        else:
            out.append("#")
            i += 1

    return re.sub(r"\s+", " ", "".join(out)).strip()


class AdNounExtractor:
    def __init__(self, custom_dict=None):
        t0 = time.time()
        print("🔧 Initializing Kiwi...")
        self.kiwi = Kiwi()
        self.custom_words_map = {}

        if custom_dict:
            for word in custom_dict:
                original = (word or "").strip()
                if not original:
                    continue

                key = normalize_keyword(original)
                if key:
                    self.custom_words_map[key] = original

                clean = re.sub(r"[^가-힣a-zA-Z0-9]", "", original)
                if clean:
                    try:
                        self.kiwi.add_user_word(clean, "NNP", 10.0)
                    except Exception as e:
                        print(f"⚠️ add_user_word skipped for '{original}' (clean='{clean}'): {e}")

        self.lemmatizer = WordNetLemmatizer()
        print(f"✅ Kiwi initialized with {len(self.custom_words_map)} custom words (took {time.time()-t0:.2f}s)")

    def _extract_english(self, processed: str, seen: set) -> list:
        """영어: regex → POS 태깅 → lemmatization"""
        candidates = re.findall(r"\b[a-zA-Z]{2,}\b", processed)
        result = []
        for w, pos in (pos_tag(candidates) if candidates else []):
            norm = re.sub(r"[^a-zA-Z0-9]", "", w).lower()
            if not norm or norm in _ENGLISH_FUNCTION_WORDS:
                continue
            # 대문자 약어(SNS 등)는 lemmatization 스킵 (lemmatize("sns", NOUN) → "sn" 방지)
            lemma = norm if w.isupper() else self.lemmatizer.lemmatize(norm, pos=_wn_pos(pos))
            if lemma not in seen:
                result.append(lemma)
                seen.add(lemma)
        return result

    def _extract_korean(self, processed: str, seen: set) -> list:
        """한국어: Kiwi 형태소 분석 (SL 제외로 영어 중복 방지)"""
        result = []
        tokens = list(self.kiwi.tokenize(processed))
        i = 0
        while i < len(tokens):
            t = tokens[i]

            # exclude 필터보다 먼저 인접 토큰 병합 시도
            # '아우터가' → '아우'(IC/NNG)+'터'(NNB/NNG) 조합이 단일 명사(NNG/NNP)이면 병합
            if i + 1 < len(tokens):
                t_next = tokens[i + 1]
                if t.start + t.len == t_next.start and str(t_next.tag) in _MERGE_NEXT_ALLOWED_TAGS:
                    combined = re.sub(r"[^가-힣a-zA-Z0-9]", "", t.form + t_next.form)
                    if combined and all('가' <= c <= '힣' for c in combined):
                        merged = False
                        for toks, _ in self.kiwi.analyze(combined, top_n=3):
                            if (len(toks) == 1
                                    and toks[0].form == combined
                                    and str(toks[0].tag) in ("NNG", "NNP")):
                                norm = combined.lower()
                                if len(combined) >= 2 and norm not in seen and not norm.isdigit():
                                    result.append(combined)
                                    seen.add(norm)
                                i += 2
                                merged = True
                                break
                        if merged:
                            continue

            # 병합 없음 → 단독 토큰 처리
            if t.tag in _KIWI_EXCLUDE_TAGS or t.tag.startswith(_KIWI_EXCLUDE_PREFIXES):
                i += 1
                continue
            cleaned = re.sub(r"[^가-힣a-zA-Z0-9]", "", t.form)
            norm = cleaned.lower()
            if len(cleaned) < 2 or not norm or norm in seen or norm.isdigit():
                i += 1
                continue
            result.append(cleaned)
            seen.add(norm)
            i += 1
        return result

    def extract_words(self, text: str, debug=False):
        if not text or not str(text).strip():
            return []

        text = str(text)
        # 하이픈 줄바꿈만 병합 ("brand-\ning" → "branding"), 나머지는 공백으로
        text = re.sub(r"([a-zA-Z가-힣])-\n([a-zA-Z가-힣])", r"\1\2", text)
        text = re.sub(r"([a-zA-Z가-힣])\n([a-zA-Z가-힣])", r"\1 \2", text)
        all_words = []
        seen = set()

        # custom_dict: 공백 제거 후 포함 여부 검사
        text_no_space = re.sub(r"[^가-힣a-zA-Z0-9]", "", text.lower())
        for key, original in self.custom_words_map.items():
            if key in text_no_space and key not in seen:
                all_words.append(original)
                seen.add(key)

        processed = text
        processed = re.sub(r"\([^)]*\)", " ", processed)
        processed = re.sub(r"\[[^\]]*\]", " ", processed)
        processed = re.sub(r"http[s]?://\S+", " ", processed)
        processed = re.sub(r"www\.\S+", " ", processed)
        processed = re.sub(r"[\U00010000-\U0010ffff]", " ", processed)
        processed = remove_or_keep_hashtags(processed)
        # 축약형 처리는 apostrophe 제거 전에 수행
        processed = re.sub(r"\b\w+n['\u2018\u2019\u201b]t\b", "", processed, flags=re.IGNORECASE)
        processed = re.sub(r"['\u2018\u2019\u201b](?:re|ll|ve|d|s|m)\b", "", processed, flags=re.IGNORECASE)
        processed = re.sub(r'["\'`]', " ", processed)
        processed = re.sub(r"\s+", " ", processed).strip()

        all_words.extend(self._extract_english(processed, seen))
        all_words.extend(self._extract_korean(processed, seen))

        if debug:
            print(f"\n🔹[DEBUG] 원본 body: {text[:200]}")
            print(f"🔹 전처리 후: {processed[:200]}")
            print(f"🔹 final_words: {all_words}")

        return all_words


def classify_keywords(extracted_words, init_map, brand_set=None):
    if not extracted_words:
        return [], []

    brand_set = brand_set or set()
    normalized_map = {}

    for w in extracted_words:
        n = normalize_keyword(w)
        if not n or n.isdigit():
            continue

        is_brand_part = False
        for brand in brand_set:
            if n == brand or (len(n) >= 2 and n in brand) or (brand in n):
                is_brand_part = True
                break
        if is_brand_part:
            continue

        normalized_map[n] = w

    if not init_map:
        return [], list(normalized_map.values())

    init_norms = list(init_map.keys())
    matched_init_norms = set()

    essential = []
    variable_norms = []

    for norm in normalized_map.keys():
        hit = None
        if norm in init_map:
            hit = norm
        else:
            if len(norm) >= 2 and re.search(r"[가-힣]", norm):
                for e_norm in init_norms:
                    if len(e_norm) >= 2 and (norm in e_norm or e_norm in norm):
                        hit = e_norm
                        break

        if hit:
            if hit not in matched_init_norms:
                essential.append(init_map[hit])
                matched_init_norms.add(hit)
            else:
                variable_norms.append(norm)
        else:
            variable_norms.append(norm)

    final_variable = []
    for norm in variable_norms:
        if norm in matched_init_norms:
            continue
        final_variable.append(normalized_map[norm])

    return essential, final_variable


def find_overlap_ad_ids(conn, limit=50):
    rows = conn.run(f"""
        WITH init AS (
          SELECT
            aa.id AS account_id,
            regexp_replace(lower(iw), '[^가-힣a-z0-9]', '', 'g') AS init_norm
          FROM ad_accounts aa
          JOIN business_portfolios bp ON aa.business_portfolio_id = bp.id
          JOIN clients cl ON bp.client_id = cl.id
          JOIN client_info ci ON ci.client_id = cl.id
          CROSS JOIN LATERAL unnest(ci.init_essential) AS iw
        ),
        var AS (
          SELECT
            a.id AS ad_id,
            a.account_id,
            regexp_replace(lower(vw), '[^가-힣a-z0-9]', '', 'g') AS var_norm
          FROM ads a
          JOIN ad_keywords ak ON ak.ad_id = a.id
          CROSS JOIN LATERAL unnest(ak.variable_keywords) AS vw
        )
        SELECT v.ad_id
        FROM var v
        JOIN init i
          ON i.account_id = v.account_id
         AND i.init_norm = v.var_norm
        WHERE v.var_norm <> ''
        GROUP BY v.ad_id
        ORDER BY v.ad_id
        LIMIT {int(limit)}
    """)
    return [int(r[0]) for r in rows] if rows else []


def overlap_still_exists(conn, ad_id: int) -> bool:
    rows = conn.run("""
        WITH init AS (
          SELECT
            aa.id AS account_id,
            regexp_replace(lower(iw), '[^가-힣a-z0-9]', '', 'g') AS init_norm
          FROM ad_accounts aa
          JOIN business_portfolios bp ON aa.business_portfolio_id = bp.id
          JOIN clients cl ON bp.client_id = cl.id
          JOIN client_info ci ON ci.client_id = cl.id
          CROSS JOIN LATERAL unnest(ci.init_essential) AS iw
        ),
        var AS (
          SELECT
            a.id AS ad_id,
            a.account_id,
            regexp_replace(lower(vw), '[^가-힣a-z0-9]', '', 'g') AS var_norm
          FROM ads a
          JOIN ad_keywords ak ON ak.ad_id = a.id
          CROSS JOIN LATERAL unnest(ak.variable_keywords) AS vw
          WHERE a.id = :ad_id
        )
        SELECT 1
        FROM var v
        JOIN init i
          ON i.account_id = v.account_id
         AND i.init_norm = v.var_norm
        WHERE v.var_norm <> ''
        LIMIT 1
    """, ad_id=ad_id)
    return bool(rows)


def main():
    start = time.time()

    db_url = os.environ.get("DB_URL", "").strip()
    if not db_url:
        raise ValueError("DB_URL 환경 변수를 설정해주세요")
    _parsed = urlparse(db_url)
    db_host     = _parsed.hostname
    db_user     = _parsed.username
    db_password = _parsed.password
    db_name     = _parsed.path.lstrip("/")
    db_port     = _parsed.port or 5432

    # Kiwi가 오분석하는 패션·뷰티 외래어 기본 사전 (CUSTOM_DICT 환경변수로 추가 가능)
    _BASE_FASHION_DICT = [
        "아우터", "이너웨어", "레이어드", "니트웨어", "캐주얼웨어",
        "스킨케어", "헤어케어", "바디케어", "선케어",
        "오버핏", "슬림핏", "레귤러핏",
    ]
    raw_custom_dict = os.environ.get("CUSTOM_DICT", "")
    custom_dict = _BASE_FASHION_DICT + [w.strip() for w in raw_custom_dict.split(",") if w.strip()]

    BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", 50))
    DEBUG        = env_bool("DEBUG", "false")
    TARGET_MODE  = os.environ.get("TARGET_MODE", "").strip().lower()
    TARGET_LIMIT = int(os.environ.get("TARGET_LIMIT", "50"))

    print(f"⚙️  settings: BATCH_SIZE={BATCH_SIZE}, DEBUG={DEBUG}, "
          f"TARGET_MODE={TARGET_MODE}, TARGET_LIMIT={TARGET_LIMIT}, "
          f"custom_dict_size={len(custom_dict)}")

    conn = pg8000.native.Connection(
        user=db_user, password=db_password,
        host=db_host, database=db_name,
        port=db_port, timeout=30
    )
    print("✅ DB connected")

    extractor = AdNounExtractor(custom_dict=custom_dict)

    total_processed = 0
    total_failed    = 0
    total_fixed     = 0
    total_still_bad = 0

    # overlap 모드: 한 번만 실행
    if TARGET_MODE == "overlap":
        t0 = time.time()
        target_ids = find_overlap_ad_ids(conn, limit=TARGET_LIMIT)
        print(f"🎯 overlap targets: {len(target_ids)} ids (took {time.time()-t0:.2f}s)")

        if not target_ids:
            print("overlap 대상 없음")
            conn.close()
            return

        in_clause, in_params = build_in_clause_params(target_ids, prefix="tid")
        ads = conn.run(
            f"""
            SELECT id AS ad_id, account_id, body
            FROM ads
            WHERE id IN {in_clause}
            ORDER BY id
            """,
            **in_params
        )
        print(f"📦 fetched target ads: {len(ads)}")
        batches = [ads]

    # 기본 모드: last_ad_id 기준으로 배치 반복
    else:
        batches = None  # 아래 루프에서 직접 처리

    if TARGET_MODE == "overlap":
        for ads in batches:
            _process_batch(
                conn, extractor, ads, DEBUG, TARGET_MODE
            )
    else:
        last_ad_id = 0
        batch_num  = 0

        while True:
            ads = conn.run("""
                SELECT id AS ad_id, account_id, body
                FROM ads
                WHERE body IS NOT NULL AND TRIM(body) <> ''
                  AND id > :last_ad_id
                ORDER BY id
                LIMIT :limit
            """, last_ad_id=last_ad_id, limit=BATCH_SIZE)

            if not ads:
                print("✅ 처리할 광고 없음. 종료.")
                break

            batch_num += 1
            print(f"\n📦 batch {batch_num}: {len(ads)}개 (last_ad_id={last_ad_id})")

            processed, failed, fixed, still_bad = _process_batch(
                conn, extractor, ads, DEBUG, TARGET_MODE
            )
            total_processed += processed
            total_failed    += failed

            last_ad_id = max(int(r[0]) for r in ads)

            if len(ads) < BATCH_SIZE:
                print("✅ 마지막 배치. 종료.")
                break

    conn.close()
    elapsed = round(time.time() - start, 3)
    print(f"\n🏁 완료 | processed={total_processed}, failed={total_failed}, "
          f"fixed={total_fixed}, still_overlap={total_still_bad}, elapsed={elapsed}s")


def _process_batch(conn, extractor, ads, debug, target_mode):
    account_ids = sorted(set(
        to_int_or_none(r[1]) for r in ads if to_int_or_none(r[1]) is not None
    ))
    in_clause, in_params = build_in_clause_params(account_ids, prefix="aid")
    init_rows = conn.run(
        f"""
        SELECT aa.id, ci.init_essential, ci.brand_name
        FROM ad_accounts aa
        JOIN business_portfolios bp ON aa.business_portfolio_id = bp.id
        JOIN clients cl ON bp.client_id = cl.id
        JOIN client_info ci ON ci.client_id = cl.id
        WHERE aa.id IN {in_clause}
        """,
        **in_params
    )

    account_keywords = {}
    account_brands   = {}

    for aid, kw_list, bn_list in init_rows:
        aid_int = to_int_or_none(aid)
        if aid_int is None:
            continue

        m = {}
        if isinstance(kw_list, (list, tuple)):
            for kw in kw_list:
                if isinstance(kw, str) and kw.strip() and kw.strip().lower() != "nan":
                    n = normalize_keyword(kw)
                    if n:
                        m[n] = kw.strip()
        account_keywords[aid_int] = m

        b_set = set()
        if isinstance(bn_list, (list, tuple)):
            for bn in bn_list:
                if isinstance(bn, str) and bn.strip():
                    norm_bn = normalize_keyword(bn)
                    if norm_bn:
                        b_set.add(norm_bn)
        account_brands[aid_int] = b_set

    processed = 0
    failed    = 0
    fixed     = 0
    still_bad = 0

    for ad_id, account_id, body in ads:
        account_id_int = to_int_or_none(account_id)
        extracted      = extractor.extract_words(body, debug=debug)
        init_map       = account_keywords.get(account_id_int, {})
        brand_set      = account_brands.get(account_id_int, set())
        essential, variable = classify_keywords(extracted, init_map, brand_set)

        try:
            conn.run("""
                INSERT INTO ad_keywords
                    (ad_id, essential_keywords, variable_keywords)
                VALUES (:ad_id, :essential_keywords, :variable_keywords)
                ON CONFLICT (ad_id)
                DO UPDATE SET
                    essential_keywords = EXCLUDED.essential_keywords,
                    variable_keywords  = EXCLUDED.variable_keywords,
                    updated_at         = now()
            """,
                ad_id=ad_id,
                essential_keywords=essential,
                variable_keywords=variable,
            )
            processed += 1

            if target_mode == "overlap":
                ad_id_int = int(ad_id)
                if overlap_still_exists(conn, ad_id_int):
                    still_bad += 1
                    print(f"🚨 still_overlap ad_id={ad_id_int}")
                else:
                    fixed += 1
                    print(f"✅ fixed ad_id={ad_id_int}")

        except Exception as e:
            failed += 1
            print(f"❌ upsert failed ad_id={ad_id}: {e}")

    return processed, failed, fixed, still_bad


if __name__ == "__main__":
    main()
