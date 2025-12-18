# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import io
import re

from supabase import create_client, Client

# requests는 requirements.txt에 반드시 포함
try:
    import requests
except Exception:
    requests = None


# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="세르지오타키니 판매 분석",
    page_icon="👔",
    layout="wide"
)

# =========================
# 템플릿 컬럼 정의 (최종)
# - 판매: 당시즌 기준 + 가격 포함
# - 소재: 두께 제거, 밀도 -> 조직
# =========================
SALES_COLS = ["품번", "컬러", "가격", "제조방식", "소재명", "핏", "기장", "당시즌판매수량", "당시즌판매액"]
MATERIAL_COLS = ["소재명", "소재업체", "혼용원단", "혼용율", "중량", "조직", "GU", "RA", "SA"]

# =========================
# Supabase 연결
# =========================
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]  # Streamlit에는 anon key
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Supabase 연결 실패: {e}")
        return None

supabase: Client = init_supabase()


# =========================
# 유틸: JSON-safe 변환 (NaN/Inf 제거)
# =========================
def make_json_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out = out.replace([np.nan, np.inf, -np.inf], None)
    return out

# =========================
# 유틸: 필수 텍스트 컬럼 기본값 보정 (NOT NULL 대비)
# =========================
def fill_required_text(df: pd.DataFrame, cols, default="UNKNOWN") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = default
        out[c] = out[c].astype(str).replace(["None", "nan"], "").fillna("")
        out[c] = out[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
        out[c] = out[c].replace("", default)
    return out

# =========================
# 유틸: 안전한 JSON 응답 처리
# =========================
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None

def show_api_error(out, fallback="AI 예측 실패(응답 형식 오류)"):
    if isinstance(out, dict):
        return out.get("error", fallback)
    return fallback

# =========================
# 유틸: 혼용율 파싱/파생 feature (예측 정확도용)
# - "COTTON / POLYESTER / ELASTANE" + "74 / 21 / 5"
# - 또는 "NYLON/VISCOSE" + "24/76"
# =========================
FIBER_ALIASES = {
    "COTTON": ["COTTON", "CO", "CT", "COTNA"],
    "POLYESTER": ["POLYESTER", "PES", "PE", "PL"],
    "NYLON": ["NYLON", "PA", "N", "NL"],
    "RAYON": ["RAYON", "VISCOSE", "VISC", "VI", "LYOCELL", "TENCEL", "MODAL"],
    "WOOL": ["WOOL", "WL"],
    "ACRYLIC": ["ACRYLIC", "AC"],
    "SPANDEX": ["SPANDEX", "ELASTANE", "ELASTIN", "PU", "SP", "LYCRA"],
    "POLYURETHANE": ["POLYURETHANE", "PU"],
}

SYNTHETIC_SET = {"POLYESTER", "NYLON", "ACRYLIC", "POLYURETHANE"}
NATURAL_SET = {"COTTON", "WOOL"}
REGENERATED_SET = {"RAYON"}

def _norm_fiber_name(name: str) -> str:
    if not name:
        return ""
    s = str(name).strip().upper()
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # alias mapping
    for canon, alist in FIBER_ALIASES.items():
        for a in alist:
            if s == a:
                return canon
    # 부분 일치도 처리
    for canon, alist in FIBER_ALIASES.items():
        for a in alist:
            if a in s:
                return canon
    return s

def parse_blend_components(blend_fibers: str, blend_ratio: str):
    """
    return: list of (fiber_canon, ratio_float)
    """
    if not blend_fibers or not blend_ratio:
        return []

    fibers = [x.strip() for x in re.split(r"[/,|]+", str(blend_fibers)) if x.strip()]
    ratios = [x.strip() for x in re.split(r"[/,|]+", str(blend_ratio)) if x.strip()]

    # 숫자만 추출
    ratios_num = []
    for r in ratios:
        m = re.findall(r"[-+]?\d*\.?\d+", r)
        if m:
            try:
                ratios_num.append(float(m[0]))
            except Exception:
                pass

    if not fibers or not ratios_num:
        return []

    # 길이 맞추기: 짧은 쪽 기준
    n = min(len(fibers), len(ratios_num))
    fibers = fibers[:n]
    ratios_num = ratios_num[:n]

    # 합이 0이면 무효
    s = sum(ratios_num)
    if s <= 0:
        return []

    # 보정(합이 100이 아니어도 상대비로)
    comps = []
    for f, r in zip(fibers, ratios_num):
        fn = _norm_fiber_name(f)
        comps.append((fn, float(r)))
    return comps

def derive_blend_features(혼용원단: str, 혼용율: str):
    """
    예측 feature:
    - pct_cotton, pct_synthetic, pct_regenerated, pct_spandex
    - n_fibers
    """
    comps = parse_blend_components(혼용원단, 혼용율)
    if not comps:
        return {
            "pct_cotton": None,
            "pct_synthetic": None,
            "pct_regenerated": None,
            "pct_spandex": None,
            "n_fibers": None,
        }

    total = sum(r for _, r in comps)
    if total <= 0:
        return {
            "pct_cotton": None,
            "pct_synthetic": None,
            "pct_regenerated": None,
            "pct_spandex": None,
            "n_fibers": len(comps),
        }

    pct_cotton = 0.0
    pct_synth = 0.0
    pct_regen = 0.0
    pct_spandex = 0.0

    for f, r in comps:
        p = r / total * 100.0
        if f in NATURAL_SET:
            if f == "COTTON":
                pct_cotton += p
        if f in SYNTHETIC_SET:
            pct_synth += p
        if f in REGENERATED_SET:
            pct_regen += p
        if f in {"SPANDEX", "POLYURETHANE"}:
            pct_spandex += p

    return {
        "pct_cotton": round(pct_cotton, 2),
        "pct_synthetic": round(pct_synth, 2),
        "pct_regenerated": round(pct_regen, 2),
        "pct_spandex": round(pct_spandex, 2),
        "n_fibers": len(comps),
    }

# =========================
# 유틸: 상관/구간 분석
# =========================
def _bin_series(s: pd.Series, method="equal_width", bins=4):
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    if method == "quantile":
        return pd.qcut(s, q=bins, duplicates="drop")
    return pd.cut(s, bins=bins)

def _safe_corr(a: pd.Series, b: pd.Series):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    df2 = pd.concat([a, b], axis=1).dropna()
    if len(df2) < 3:
        return np.nan
    return df2.iloc[:, 0].corr(df2.iloc[:, 1])


# =========================
# 데이터 로드
# =========================
@st.cache_data(ttl=600)
def load_sales_data():
    if supabase is None:
        return pd.DataFrame(columns=SALES_COLS)
    try:
        res = supabase.table("sales_data").select("*").execute()
        if res.data:
            df = pd.DataFrame(res.data)
            for c in SALES_COLS:
                if c not in df.columns:
                    df[c] = None
            df = df[SALES_COLS].copy()

            for col in ["가격", "당시즌판매수량", "당시즌판매액"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            df = fill_required_text(df, ["품번", "컬러", "제조방식", "소재명", "핏", "기장"])
            return df
        return pd.DataFrame(columns=SALES_COLS)
    except Exception as e:
        st.error(f"판매 데이터 로드 실패: {e}")
        return pd.DataFrame(columns=SALES_COLS)

@st.cache_data(ttl=600)
def load_material_data():
    """
    ✅ 혼용율은 숫자 캐스팅하지 않음 (예: '24 / 76' 보존)
    """
    if supabase is None:
        return pd.DataFrame(columns=MATERIAL_COLS)
    try:
        res = supabase.table("material_data").select("*").execute()
        if res.data:
            df = pd.DataFrame(res.data)
            for c in MATERIAL_COLS:
                if c not in df.columns:
                    df[c] = None
            df = df[MATERIAL_COLS].copy()

            # 숫자형: 중량/GU/RA/SA만
            for col in ["중량", "GU", "RA", "SA"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # 텍스트 정리
            for tcol in ["소재명", "소재업체", "혼용원단", "혼용율", "조직"]:
                if tcol in df.columns:
                    df[tcol] = (
                        df[tcol].astype(str)
                        .replace(["None", "nan"], "")
                        .fillna("")
                        .apply(lambda x: x.strip())
                    )
            return df
        return pd.DataFrame(columns=MATERIAL_COLS)
    except Exception as e:
        st.error(f"소재 데이터 로드 실패: {e}")
        return pd.DataFrame(columns=MATERIAL_COLS)


# =========================
# 데이터 저장/삭제
# =========================
def save_sales_data(new_df: pd.DataFrame) -> bool:
    if supabase is None:
        st.error("❌ Supabase 연결이 없습니다.")
        return False
    try:
        df = new_df.copy()
        for c in SALES_COLS:
            if c not in df.columns:
                df[c] = None

        df = fill_required_text(df, ["품번", "컬러", "제조방식", "소재명", "핏", "기장"])

        for col in ["가격", "당시즌판매수량", "당시즌판매액"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df = make_json_safe_df(df)

        records = df[SALES_COLS].to_dict("records")
        if not records:
            st.warning("저장할 데이터가 없습니다.")
            return False

        supabase.table("sales_data").insert(records).execute()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"판매 저장 실패: {e}")
        return False

def replace_sales_data(df_upload: pd.DataFrame) -> bool:
    """
    ✅ 업로드 중복 폭증 방지용: 전체 교체(삭제 후 insert)
    """
    if supabase is None:
        st.error("❌ Supabase 연결이 없습니다.")
        return False
    try:
        supabase.table("sales_data").delete().neq("id", 0).execute()

        df = df_upload.copy()
        for c in SALES_COLS:
            if c not in df.columns:
                df[c] = None

        df = fill_required_text(df, ["품번", "컬러", "제조방식", "소재명", "핏", "기장"])
        for col in ["가격", "당시즌판매수량", "당시즌판매액"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df = make_json_safe_df(df)
        records = df[SALES_COLS].to_dict("records")
        if not records:
            st.warning("저장할 데이터가 없습니다.")
            return False

        supabase.table("sales_data").insert(records).execute()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"판매 전체 교체 실패: {e}")
        return False

def save_material_data(new_df: pd.DataFrame) -> bool:
    if supabase is None:
        st.error("❌ Supabase 연결이 없습니다.")
        return False
    try:
        df = new_df.copy()
        for c in MATERIAL_COLS:
            if c not in df.columns:
                df[c] = None

        df = fill_required_text(df, ["소재명"], default="UNKNOWN_MATERIAL")

        # 숫자형: 중량/GU/RA/SA만
        for col in ["중량", "GU", "RA", "SA"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 텍스트 정리
        for tcol in ["소재업체", "혼용원단", "혼용율", "조직"]:
            if tcol in df.columns:
                df[tcol] = (
                    df[tcol].astype(str)
                    .replace(["None", "nan"], "")
                    .fillna("")
                    .apply(lambda x: x.strip())
                )

        df = make_json_safe_df(df)

        records = df[MATERIAL_COLS].to_dict("records")
        if not records:
            st.warning("저장할 데이터가 없습니다.")
            return False

        supabase.table("material_data").insert(records).execute()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"소재 저장 실패: {e}")
        return False

def replace_material_data(df_upload: pd.DataFrame) -> bool:
    if supabase is None:
        st.error("❌ Supabase 연결이 없습니다.")
        return False
    try:
        supabase.table("material_data").delete().neq("id", 0).execute()

        df = df_upload.copy()
        for c in MATERIAL_COLS:
            if c not in df.columns:
                df[c] = None

        df = fill_required_text(df, ["소재명"], default="UNKNOWN_MATERIAL")
        for col in ["중량", "GU", "RA", "SA"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for tcol in ["소재업체", "혼용원단", "혼용율", "조직"]:
            if tcol in df.columns:
                df[tcol] = (
                    df[tcol].astype(str)
                    .replace(["None", "nan"], "")
                    .fillna("")
                    .apply(lambda x: x.strip())
                )

        df = make_json_safe_df(df)
        records = df[MATERIAL_COLS].to_dict("records")
        if not records:
            st.warning("저장할 데이터가 없습니다.")
            return False

        supabase.table("material_data").insert(records).execute()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"소재 전체 교체 실패: {e}")
        return False

def delete_all_sales_data() -> bool:
    if supabase is None:
        return False
    try:
        supabase.table("sales_data").delete().neq("id", 0).execute()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"판매 데이터 삭제 실패: {e}")
        return False

def delete_all_material_data() -> bool:
    if supabase is None:
        return False
    try:
        supabase.table("material_data").delete().neq("id", 0).execute()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"소재 데이터 삭제 실패: {e}")
        return False


# =========================
# 품번 파싱/매핑
# =========================
def parse_item_code(code):
    if not code:
        return None
    code = str(code).strip()
    if len(code) < 8:
        return None
    try:
        return {
            "brand": code[0] if len(code) > 0 else None,
            "gender": code[1] if len(code) > 1 else None,
            "item_code": code[2:4] if len(code) >= 4 else None,
            "sequence": code[4:7] if len(code) >= 7 else None,
            "year": code[7] if len(code) >= 8 else None,
            "season": code[8] if len(code) >= 9 else None,
        }
    except Exception:
        return None

ITEM_MAPPING = {
    "DJ": "다운점퍼", "DV": "다운베스트", "JK": "자켓", "JP": "점퍼",
    "KC": "니트가디건", "PD": "패딩", "VT": "베스트", "WJ": "윈드브레이커", "WT": "우븐티셔츠",
    "HD": "후드티", "KP": "스웨터풀오버", "KV": "스웨터베스트", "KU": "반팔스웨터",
    "MT": "맨투맨", "OP": "원피스", "PQ": "폴로티셔츠", "RL": "긴팔티셔츠",
    "RS": "반팔티셔츠", "TR": "트레이닝상의", "WS": "우븐셔츠",
    "LG": "레깅스", "PT": "팬츠", "SK": "스커트", "SP": "반바지",
    "SR": "여성하의스코트", "TB": "트레이닝숏팬츠", "TP": "트레이닝하의",
    "BR": "브라", "SL": "슬리브리스",
}

CATEGORY_MAPPING = {
    "DJ": "아우터", "DV": "아우터", "JK": "아우터", "JP": "아우터", "KC": "아우터",
    "PD": "아우터", "VT": "아우터", "WJ": "아우터", "WT": "아우터",
    "HD": "이너", "KP": "이너", "KV": "이너", "KU": "이너", "MT": "이너",
    "OP": "이너", "PQ": "이너", "RL": "이너", "RS": "이너", "TR": "이너", "WS": "이너",
    "LG": "하의", "PT": "하의", "SK": "하의", "SP": "하의", "SR": "하의",
    "TB": "하의", "TP": "하의",
    "BR": "기타", "SL": "기타",
}

GENDER_MAPPING = {"M": "남성", "W": "여성", "U": "공용"}
SEASON_MAPPING = {"1": "봄", "3": "여름", "4": "가을", "6": "겨울"}
YEAR_MAPPING = {"3": "2023", "4": "2024", "5": "2025", "6": "2026"}

FIT_OPTIONS = ["slim", "regular", "semi-over", "over", "SLIM", "REGULAR", "SEMI-OVER", "OVER"]
LENGTH_OPTIONS = ["Crop", "Mid", "Long", "Regular", "Semi-Crop", "Short", "CROP", "MID", "LONG", "REGULAR", "SEMI-CROP", "SHORT"]
MANUFACTURING_OPTIONS = ["컷앤소", "우븐", "스웨터", "KNIT", "WOVEN", "CUT&SEW", "CUT-SEW"]

def enrich_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    parsed_data = []
    for _, row in enriched.iterrows():
        parsed = parse_item_code(row.get("품번"))
        if parsed:
            parsed_data.append({
                "성별": GENDER_MAPPING.get(parsed["gender"], "알수없음"),
                "아이템명": ITEM_MAPPING.get(parsed["item_code"], "알수없음"),
                "카테고리": CATEGORY_MAPPING.get(parsed["item_code"], "기타"),
                "연도": YEAR_MAPPING.get(parsed["year"], "알수없음"),
                "시즌": SEASON_MAPPING.get(parsed["season"], "알수없음"),
            })
        else:
            parsed_data.append({
                "성별": "알수없음", "아이템명": "알수없음",
                "카테고리": "기타", "연도": "알수없음", "시즌": "알수없음"
            })
    return pd.concat([enriched.reset_index(drop=True), pd.DataFrame(parsed_data)], axis=1)


# =========================
# 소재 매핑 유틸 (예측 입력 feature 강화)
# =========================
def get_material_row(material_name: str, mat_df: pd.DataFrame):
    if mat_df is None or mat_df.empty or not material_name:
        return None
    # 소재명 exact 우선, 없으면 trim 비교
    m = mat_df[mat_df["소재명"].astype(str).str.strip() == str(material_name).strip()]
    if not m.empty:
        return m.iloc[0].to_dict()
    m2 = mat_df[mat_df["소재명"].astype(str).str.contains(str(material_name).strip(), na=False)]
    if not m2.empty:
        return m2.iloc[0].to_dict()
    return None


# =========================
# Session State 초기화
# =========================
if "sales_data" not in st.session_state:
    st.session_state.sales_data = load_sales_data()
if "material_data" not in st.session_state:
    st.session_state.material_data = load_material_data()
if "ai_session_id" not in st.session_state:
    st.session_state.ai_session_id = None
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []


# =========================
# Sidebar / Menu
# =========================
st.sidebar.title("👔 세르지오타키니")
st.sidebar.markdown("### 판매 분석 시스템")

menu = st.sidebar.radio(
    "메뉴",
    ["🎯 조합 예측(AI)", "📥 데이터 입력", "📊 대시보드", "🏆 랭킹", "🧵 소재 분석", "🤖 AI 인사이트/챗봇", "💾 데이터 관리"]
)


# =========================
# 1) 조합 예측 (AI 기반)
# =========================
if menu == "🎯 조합 예측(AI)":
    st.title("🎯 조합 예측 시뮬레이터 (AI 기반)")

    if requests is None:
        st.error("requests 패키지가 없습니다. requirements.txt에 requests를 추가하세요.")
    elif st.session_state.sales_data.empty:
        st.warning("⚠️ 데이터가 없습니다. '데이터 입력'에서 먼저 입력해주세요.")
    else:
        df_enriched = enrich_sales_data(st.session_state.sales_data)
        materials = sorted(df_enriched["소재명"].dropna().unique().tolist())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 조합 입력 (예측 Feature 강화)")
            gender = st.selectbox("성별", ["남성", "여성", "공용"])
            item_name = st.selectbox("아이템", sorted(set(ITEM_MAPPING.values())))
            manufacturing = st.selectbox("제조방식", MANUFACTURING_OPTIONS)

            material = st.selectbox("소재", materials) if materials else st.text_input("소재명 입력")
            fit = st.selectbox("핏", FIT_OPTIONS)
            length = st.selectbox("기장", LENGTH_OPTIONS)

            # ✅ 예측 정확도 핵심 feature: 가격
            # 최근 데이터의 중앙값을 기본값으로 제안
            default_price = int(df_enriched["가격"].median()) if "가격" in df_enriched.columns and len(df_enriched) else 0
            price = st.number_input("가격(예측 입력)", min_value=0, step=1000, value=default_price)

            # ✅ 소재마스터 자동 매핑(조직/GU/RA/SA/중량/혼용정보)
            mat_row = get_material_row(material, st.session_state.material_data)
            with st.expander("🧵 소재 마스터 매핑(자동 입력) 확인", expanded=True):
                if mat_row:
                    st.write({
                        "조직": mat_row.get("조직"),
                        "GU": mat_row.get("GU"),
                        "RA": mat_row.get("RA"),
                        "SA": mat_row.get("SA"),
                        "중량": mat_row.get("중량"),
                        "혼용원단": mat_row.get("혼용원단"),
                        "혼용율": mat_row.get("혼용율"),
                        "소재업체": mat_row.get("소재업체"),
                    })
                else:
                    st.info("소재 마스터에서 해당 소재를 찾지 못했습니다. (예측은 가능하지만 정확도는 낮아질 수 있음)")

            predict_btn = st.button("🔮 AI 예측 결과 보기", type="primary", use_container_width=True)

        with col2:
            st.subheader("📊 예측 결과")
            if predict_btn:
                fn_predict = st.secrets.get("SUPABASE_FUNCTION_PREDICT_URL", "")
                if not fn_predict:
                    st.error("SUPABASE_FUNCTION_PREDICT_URL이 설정되지 않았습니다.")
                else:
                    # 파생 feature 생성
                    blend_feats = {}
                    if mat_row:
                        blend_feats = derive_blend_features(mat_row.get("혼용원단"), mat_row.get("혼용율"))
                    else:
                        blend_feats = {
                            "pct_cotton": None, "pct_synthetic": None, "pct_regenerated": None, "pct_spandex": None, "n_fibers": None
                        }

                    payload = {
                        # 기본 조합 feature
                        "gender": gender,
                        "item_name": item_name,
                        "manufacturing": manufacturing,
                        "material": material,
                        "fit": fit,
                        "length": length,

                        # ✅ 예측 정확도 핵심: 가격
                        "price": float(price),

                        # ✅ 소재 마스터 feature(있으면 전달)
                        "material_meta": {
                            "org": (mat_row.get("조직") if mat_row else None),
                            "gu": (mat_row.get("GU") if mat_row else None),
                            "ra": (mat_row.get("RA") if mat_row else None),
                            "sa": (mat_row.get("SA") if mat_row else None),
                            "weight": (mat_row.get("중량") if mat_row else None),
                            "blend_fibers": (mat_row.get("혼용원단") if mat_row else None),
                            "blend_ratio": (mat_row.get("혼용율") if mat_row else None),
                            "supplier": (mat_row.get("소재업체") if mat_row else None),
                        },

                        # ✅ 혼용 파생 feature(모델 성능에 유의미)
                        "blend_features": blend_feats,
                    }

                    # 마지막 안전처리(Inf/NaN 제거)
                    def _clean(x):
                        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
                            return None
                        return x

                    def _deep_clean(obj):
                        if isinstance(obj, dict):
                            return {k: _deep_clean(v) for k, v in obj.items()}
                        if isinstance(obj, list):
                            return [_deep_clean(v) for v in obj]
                        return _clean(obj)

                    payload = _deep_clean(payload)

                    out = None
                    try:
                        r = requests.post(fn_predict, json=payload, timeout=120)
                        out = safe_json(r)
                    except Exception as e:
                        st.error(f"AI 예측 호출 실패: {e}")

                    if not isinstance(out, dict) or not out.get("ok"):
                        st.error(show_api_error(out))
                        st.caption("※ Edge Function이 추가 feature를 아직 처리하지 않아도, 기본 feature로 예측은 가능해야 합니다.")
                    else:
                        res = out.get("result", {}) or {}
                        st.success("✅ AI 예측 완료")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("예상 당시즌판매수량", f"{float(res.get('pred_qty', 0)):.0f}개")
                        c2.metric("예상 당시즌판매액", f"{float(res.get('pred_amt', 0)):,.0f}원")
                        c3.metric("신뢰도", f"{float(res.get('confidence', 0)):.0f}%")

                        st.markdown("#### 근거")
                        st.write(res.get("rationale", ""))

                        warnings = res.get("warnings", [])
                        if warnings:
                            st.warning(" / ".join(warnings))

                        # 디버그: 실제로 어떤 feature를 보냈는지 확인(운영 안정화용)
                        with st.expander("🛠️ (디버그) 예측 입력 payload 보기", expanded=False):
                            st.json(payload)


# =========================
# 2) 데이터 입력
# =========================
elif menu == "📥 데이터 입력":
    st.title("📥 데이터 입력")

    tab1, tab2, tab3 = st.tabs(["📝 수동 입력", "📂 Excel 업로드", "🧵 소재 관리"])

    with tab1:
        st.subheader("판매 데이터 수동 입력")
        st.info("💡 템플릿: 품번/컬러/가격/제조방식/소재명/핏/기장/당시즌판매수량/당시즌판매액")

        col1, col2 = st.columns(2)
        with col1:
            input_code = st.text_input("품번", placeholder="TXHD6054")
            input_color = st.text_input("컬러", placeholder="BKS")
            input_price_unit = st.number_input("가격", min_value=0, step=1000, value=149000)
            input_manufacturing = st.text_input("제조방식", value="KNIT")
            input_material = st.text_input("소재명", placeholder="JZR3055 595쮸리")

        with col2:
            input_fit = st.text_input("핏", value="REGULAR")
            input_length = st.text_input("기장", value="REGULAR")
            input_qty = st.number_input("당시즌판매수량", min_value=0, step=1, value=15)

            auto_calc_amt = st.checkbox("당시즌판매액 자동 계산(가격×수량)", value=True)
            if auto_calc_amt:
                input_amt = int(input_price_unit * input_qty)
                st.number_input("당시즌판매액", min_value=0, step=1000, value=input_amt, disabled=True)
            else:
                input_amt = st.number_input("당시즌판매액", min_value=0, step=1000, value=2235000)

        if st.button("➕ 판매 데이터 추가", type="primary"):
            if input_code and input_color and input_material:
                new_row = pd.DataFrame([{
                    "품번": input_code,
                    "컬러": input_color,
                    "가격": int(input_price_unit),
                    "제조방식": input_manufacturing,
                    "소재명": input_material,
                    "핏": input_fit,
                    "기장": input_length,
                    "당시즌판매수량": int(input_qty),
                    "당시즌판매액": int(input_amt),
                }])
                if save_sales_data(new_row):
                    st.session_state.sales_data = load_sales_data()
                    st.success("✅ 추가 완료!")
                    st.rerun()
            else:
                st.error("❌ 품번, 컬러, 소재명은 필수입니다.")

    with tab2:
        st.subheader("Excel 업로드 (판매 데이터)")

        template = pd.DataFrame(columns=SALES_COLS)
        template.loc[0] = ["TXHD6054", "BKS", 149000, "KNIT", "JZR3055 595쮸리", "REGULAR", "REGULAR", 15, 2235000]

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            template.to_excel(writer, index=False, sheet_name="판매데이터")

        st.download_button(
            "📥 판매 템플릿 다운로드",
            buffer.getvalue(),
            "판매데이터_템플릿.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        uploaded = st.file_uploader("판매 Excel 파일 선택", type=["xlsx", "xls"])
        if uploaded:
            try:
                df_upload = pd.read_excel(uploaded)
                df_upload = make_json_safe_df(df_upload)

                missing = [c for c in SALES_COLS if c not in df_upload.columns]
                if missing:
                    st.error(f"❌ 업로드 파일 컬럼 누락: {missing}")
                else:
                    st.dataframe(df_upload.head(20), use_container_width=True)

                    colA, colB = st.columns(2)
                    with colA:
                        if st.button("✅ 판매 업로드 적용(추가 Insert)"):
                            if save_sales_data(df_upload):
                                st.session_state.sales_data = load_sales_data()
                                st.success(f"✅ {len(df_upload)}개 추가!")
                                st.rerun()

                    with colB:
                        if st.button("♻️ 판매 업로드 적용(전체 교체)", help="중복 폭증 방지: 기존 판매 데이터 삭제 후 업로드로 교체"):
                            if replace_sales_data(df_upload):
                                st.session_state.sales_data = load_sales_data()
                                st.success(f"✅ 전체 교체 완료! ({len(df_upload)}개)")
                                st.rerun()

            except Exception as e:
                st.error(f"❌ 오류: {e}")

    with tab3:
        st.subheader("소재 마스터 관리 (두께 제거 / 조직 사용)")
        st.caption("✅ 혼용율은 '24 / 76' 같은 문자열도 그대로 저장됩니다.")

        template_mat = pd.DataFrame(columns=MATERIAL_COLS)
        template_mat.loc[0] = ["BF-5933", "BF", "POLYESTER", "100", 300, "INTERLOCK", 2, 1, 3]

        buffer2 = io.BytesIO()
        with pd.ExcelWriter(buffer2, engine="openpyxl") as writer:
            template_mat.to_excel(writer, index=False, sheet_name="소재데이터")

        st.download_button(
            "📥 소재 템플릿 다운로드",
            buffer2.getvalue(),
            "소재템플릿.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        uploaded_mat = st.file_uploader("소재 Excel 파일 선택", type=["xlsx", "xls"])
        if uploaded_mat:
            try:
                df_mat = pd.read_excel(uploaded_mat)
                df_mat = make_json_safe_df(df_mat)

                missing = [c for c in MATERIAL_COLS if c not in df_mat.columns]
                if missing:
                    st.error(f"❌ 업로드 파일 컬럼 누락: {missing}")
                else:
                    st.dataframe(df_mat.head(30), use_container_width=True)

                    colA, colB = st.columns(2)
                    with colA:
                        if st.button("✅ 소재 업로드 적용(추가 Insert)"):
                            if save_material_data(df_mat):
                                st.session_state.material_data = load_material_data()
                                st.success("✅ 소재 추가 완료!")
                                st.rerun()
                    with colB:
                        if st.button("♻️ 소재 업로드 적용(전체 교체)", help="중복 폭증 방지: 기존 소재 데이터 삭제 후 업로드로 교체"):
                            if replace_material_data(df_mat):
                                st.session_state.material_data = load_material_data()
                                st.success("✅ 소재 전체 교체 완료!")
                                st.rerun()
            except Exception as e:
                st.error(f"❌ 오류: {e}")


# =========================
# 3) 대시보드
# =========================
elif menu == "📊 대시보드":
    st.title("📊 판매 분석 대시보드 (당시즌 기준)")

    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data.copy())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 당시즌판매수량", f"{df['당시즌판매수량'].sum():,}개")
        c2.metric("총 당시즌판매액", f"{df['당시즌판매액'].sum():,}원")

        total_qty = df["당시즌판매수량"].sum()
        total_amt = df["당시즌판매액"].sum()
        avg_price = (total_amt / total_qty) if total_qty > 0 else 0
        c3.metric("평균 판매단가(판매액/수량)", f"{avg_price:,.0f}원")
        c4.metric("총 SKU", f"{len(df):,}개")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("👥 성별 당시즌판매수량")
            gender_sales = df.groupby("성별")["당시즌판매수량"].sum()
            fig1 = px.pie(values=gender_sales.values, names=gender_sales.index, hole=0.4)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("🏭 제조방식별 당시즌판매수량")
            manu_sales = df.groupby("제조방식")["당시즌판매수량"].sum().sort_values(ascending=False)
            fig2 = px.bar(x=manu_sales.values, y=manu_sales.index, orientation="h")
            fig2.update_layout(showlegend=False, xaxis_title="당시즌판매수량", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)


# =========================
# 4) 랭킹
# =========================
elif menu == "🏆 랭킹":
    st.title("🏆 조합별 성과 랭킹 (당시즌 기준)")

    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data.copy())
        df["조합"] = df["성별"] + " / " + df["아이템명"] + " / " + df["제조방식"] + " / " + df["소재명"] + " / " + df["핏"] + " / " + df["기장"]

        combo_stats = df.groupby("조합").agg({
            "당시즌판매수량": ["sum", "mean", "count"],
            "당시즌판매액": ["sum", "mean"],
        }).round(0)

        combo_stats.columns = ["총당시즌판매수량", "평균당시즌판매수량", "데이터수", "총당시즌판매액", "평균당시즌판매액"]
        combo_stats = combo_stats.reset_index()

        metric = st.radio("분석 기준", ["총당시즌판매수량", "평균당시즌판매수량", "총당시즌판매액", "평균당시즌판매액"], horizontal=True)
        top_n = st.slider("표시할 조합 수", 5, 20, 10)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"🥇 Best {top_n}")
            top_combos = combo_stats.nlargest(top_n, metric)
            fig_top = px.bar(top_combos, x=metric, y="조합", orientation="h")
            fig_top.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_top, use_container_width=True)
            st.dataframe(top_combos, use_container_width=True, hide_index=True)

        with col2:
            st.subheader(f"🥉 Worst {top_n}")
            bottom_combos = combo_stats.nsmallest(top_n, metric)
            fig_bottom = px.bar(bottom_combos, x=metric, y="조합", orientation="h")
            fig_bottom.update_layout(showlegend=False, yaxis={"categoryorder": "total descending"})
            st.plotly_chart(fig_bottom, use_container_width=True)
            st.dataframe(bottom_combos, use_container_width=True, hide_index=True)


# =========================
# 5) 소재 분석 + 조직 매트릭스
# =========================
elif menu == "🧵 소재 분석":
    st.title("🧵 소재별 성과 분석 (당시즌 기준)")

    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data.copy())

        material_stats = df.groupby("소재명").agg({
            "당시즌판매수량": ["sum", "mean", "count"],
            "당시즌판매액": ["sum", "mean"],
            "품번": "nunique",
        }).round(0)

        material_stats.columns = ["총당시즌판매수량", "평균당시즌판매수량", "데이터수", "총당시즌판매액", "평균당시즌판매액", "SKU수"]
        material_stats = material_stats.reset_index().sort_values("총당시즌판매수량", ascending=False)

        st.subheader("📊 소재별 성과 요약")
        st.dataframe(material_stats, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🧬 조직 × GU/RA/SA × 판매 성과 매트릭스")

        if st.session_state.material_data.empty:
            st.warning("소재 마스터(material_data)가 비어 있어 조직 매트릭스를 생성할 수 없습니다.")
        else:
            sales_df = st.session_state.sales_data.copy()
            mat_df = st.session_state.material_data.copy()

            sales_df["당시즌판매수량"] = pd.to_numeric(sales_df["당시즌판매수량"], errors="coerce").fillna(0)
            sales_df["당시즌판매액"] = pd.to_numeric(sales_df["당시즌판매액"], errors="coerce").fillna(0)

            for c in ["GU", "RA", "SA"]:
                if c in mat_df.columns:
                    mat_df[c] = pd.to_numeric(mat_df[c], errors="coerce")

            mat_small = mat_df[["소재명", "조직", "GU", "RA", "SA"]].drop_duplicates(subset=["소재명"])
            merged = sales_df.merge(mat_small, on="소재명", how="left")

            miss_org = merged["조직"].isna().mean() if len(merged) else 1.0
            st.caption(f"조직 미매칭 비율: **{miss_org*100:.1f}%**")

            matrix = (
                merged.dropna(subset=["조직"])
                .groupby("조직")
                .agg(
                    평균_GU=("GU", "mean"),
                    평균_RA=("RA", "mean"),
                    평균_SA=("SA", "mean"),
                    총판매수량=("당시즌판매수량", "sum"),
                    평균판매수량=("당시즌판매수량", "mean"),
                    총판매액=("당시즌판매액", "sum"),
                    SKU수=("품번", "nunique"),
                    데이터수=("품번", "count"),
                )
                .reset_index()
            )

            for c in ["평균_GU", "평균_RA", "평균_SA", "평균판매수량"]:
                matrix[c] = matrix[c].round(2)
            matrix["총판매액"] = matrix["총판매액"].fillna(0).astype(int)

            st.dataframe(
                matrix.sort_values("총판매수량", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 📊 조직 포지셔닝 (SA × GU, 버블=총판매수량)")
            if not matrix.empty:
                fig = px.scatter(
                    matrix,
                    x="평균_SA",
                    y="평균_GU",
                    size="총판매수량",
                    color="조직",
                    hover_name="조직",
                    size_max=60,
                    labels={"평균_SA": "SA(매끈함)", "평균_GU": "GU(광택)"}
                )
                fig.update_layout(xaxis_title="SA (매끈함 ↑)", yaxis_title="GU (광택 ↑)")
                st.plotly_chart(fig, use_container_width=True)


# =========================
# 6) AI 인사이트/챗봇 (Edge Function 호출)
# =========================
elif menu == "🤖 AI 인사이트/챗봇":
    st.title("🤖 AI 인사이트 & Q&A 챗봇")

    if requests is None:
        st.error("requests 패키지가 없습니다. requirements.txt에 requests를 추가하세요.")
    elif supabase is None:
        st.error("Supabase 연결이 없습니다.")
    else:
        fn_url = st.secrets.get("SUPABASE_FUNCTION_INSIGHTS_URL", "")
        if not fn_url:
            st.warning("st.secrets에 SUPABASE_FUNCTION_INSIGHTS_URL을 설정해주세요.")
        else:
            st.caption("※ OpenAI Key는 Streamlit이 아니라 Supabase Edge Function에만 설정하세요(보안).")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📌 자동 인사이트 생성")
                scope = st.text_input("스코프(scope)", value="global", help="예: global / org:INTERLOCK")
                if st.button("🚀 인사이트 생성", type="primary", use_container_width=True):
                    try:
                        r = requests.post(fn_url, json={"mode": "insight", "scope": scope}, timeout=120)
                        out = safe_json(r)
                        if isinstance(out, dict) and out.get("ok"):
                            st.success("생성 완료")
                            st.markdown(out.get("insight", ""))
                        else:
                            st.error(show_api_error(out, "Unknown error"))
                    except Exception as e:
                        st.error(f"호출 실패: {e}")

            with col2:
                st.subheader("💬 추가 질의응답(챗봇)")
                q = st.text_area("질문", placeholder="예: INTERLOCK 조직은 SA가 높은데 판매가 왜 낮아? 다음 시즌 테스트는?")
                if st.button("질문하기", use_container_width=True):
                    if not q.strip():
                        st.warning("질문을 입력해주세요.")
                    else:
                        try:
                            # 최근 N턴만 유지(렉 방지)
                            st.session_state.chat_log = st.session_state.chat_log[-20:]

                            payload = {
                                "mode": "chat",
                                "scope": scope,
                                "session_id": st.session_state.ai_session_id,
                                "question": q.strip(),
                            }
                            r = requests.post(fn_url, json=payload, timeout=120)
                            out = safe_json(r)
                            if isinstance(out, dict) and out.get("ok"):
                                st.session_state.ai_session_id = out.get("session_id")
                                answer = out.get("answer", "")
                                st.session_state.chat_log.append(("Q", q.strip()))
                                st.session_state.chat_log.append(("A", answer))
                            else:
                                st.error(show_api_error(out, "Unknown error"))
                        except Exception as e:
                            st.error(f"호출 실패: {e}")

                # 로그 표시
                if st.session_state.chat_log:
                    st.divider()
                    for role, text in st.session_state.chat_log[-20:]:
                        if role == "Q":
                            st.markdown(f"**Q:** {text}")
                        else:
                            st.markdown(f"**A:** {text}")


# =========================
# 7) 데이터 관리
# =========================
elif menu == "💾 데이터 관리":
    st.title("💾 데이터 관리")

    tab1, tab2, tab3 = st.tabs(["📥 데이터 다운로드", "📊 데이터 확인/편집", "🗑️ 데이터 삭제"])

    with tab1:
        st.subheader("📥 데이터 다운로드")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 판매 데이터")
            if not st.session_state.sales_data.empty:
                buffer1 = io.BytesIO()
                with pd.ExcelWriter(buffer1, engine="openpyxl") as writer:
                    st.session_state.sales_data[SALES_COLS].to_excel(writer, index=False, sheet_name="판매데이터")
                st.download_button(
                    "📥 판매 데이터 Excel 다운로드",
                    buffer1.getvalue(),
                    f"판매데이터_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("다운로드할 판매 데이터가 없습니다.")

        with col2:
            st.markdown("#### 소재 데이터")
            if not st.session_state.material_data.empty:
                buffer2 = io.BytesIO()
                with pd.ExcelWriter(buffer2, engine="openpyxl") as writer:
                    st.session_state.material_data[MATERIAL_COLS].to_excel(writer, index=False, sheet_name="소재데이터")
                st.download_button(
                    "📥 소재 데이터 Excel 다운로드",
                    buffer2.getvalue(),
                    f"소재데이터_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("다운로드할 소재 데이터가 없습니다.")

    with tab2:
        st.subheader("📊 데이터 확인/편집 (간단 편집)")
        st.caption("⚠️ 편집 후 저장은 '전체 교체' 방식으로 반영됩니다(중복/불일치 방지).")

        st.markdown("#### 판매 데이터 편집")
        df_edit_sales = st.data_editor(
            st.session_state.sales_data[SALES_COLS].copy(),
            use_container_width=True,
            num_rows="dynamic",
            key="editor_sales"
        )
        if st.button("💾 판매 편집 내용 저장(전체 교체)", type="primary"):
            if replace_sales_data(df_edit_sales):
                st.session_state.sales_data = load_sales_data()
                st.success("✅ 판매 데이터 저장 완료")
                st.rerun()

        st.divider()

        st.markdown("#### 소재 데이터 편집")
        df_edit_mat = st.data_editor(
            st.session_state.material_data[MATERIAL_COLS].copy(),
            use_container_width=True,
            num_rows="dynamic",
            key="editor_mat"
        )
        if st.button("💾 소재 편집 내용 저장(전체 교체)", type="primary"):
            if replace_material_data(df_edit_mat):
                st.session_state.material_data = load_material_data()
                st.success("✅ 소재 데이터 저장 완료")
                st.rerun()

    with tab3:
        st.subheader("🗑️ 데이터 삭제")
        st.warning("⚠️ **주의**: 삭제된 데이터는 복구할 수 없습니다! 먼저 백업을 다운로드하세요.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ 판매 데이터 전체 삭제", type="secondary"):
                if delete_all_sales_data():
                    st.session_state.sales_data = load_sales_data()
                    st.success("✅ 판매 데이터가 삭제되었습니다.")
                    st.rerun()
        with c2:
            if st.button("🗑️ 소재 데이터 전체 삭제", type="secondary"):
                if delete_all_material_data():
                    st.session_state.material_data = load_material_data()
                    st.success("✅ 소재 데이터가 삭제되었습니다.")
                    st.rerun()


# =========================
# Footer
# =========================
st.sidebar.divider()
st.sidebar.info(f"""
📊 **현재 데이터 현황**
- 판매 데이터: {len(st.session_state.sales_data)}건
- 소재 데이터: {len(st.session_state.material_data)}건

🔄 **캐시**
- 캐시 시간: 10분
""")
st.sidebar.caption("© 2025 세르지오타키니 판매분석시스템 (Streamlit+Supabase)")
