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
# UI/UX: Global CSS
# =========================
st.markdown(
    """
<style>
:root { --radius: 18px; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.5px; }

/* Card */
.card {
  border-radius: var(--radius);
  padding: 16px 16px;
  border: 1px solid rgba(120,120,120,0.22);
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}
.card-title { font-weight: 750; font-size: 1.05rem; margin-bottom: 4px; }
.card-sub { opacity: 0.8; font-size: 0.92rem; margin-bottom: 10px; }
.hr { height: 1px; background: rgba(120,120,120,0.20); margin: 12px 0; }

/* Badges */
.badge {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(120,120,120,0.28);
  font-size: 0.82rem; margin-right: 6px; margin-top: 6px;
}
.badge-green { background: rgba(24,198,131,0.12); border-color: rgba(24,198,131,0.35); }
.badge-blue  { background: rgba(64,150,255,0.12); border-color: rgba(64,150,255,0.35); }
.badge-amber { background: rgba(255,190,66,0.13); border-color: rgba(255,190,66,0.40); }
.badge-red   { background: rgba(255,86,86,0.12); border-color: rgba(255,86,86,0.38); }

/* Callout */
.callout {
  border-radius: var(--radius);
  padding: 14px 14px;
  border: 1px solid rgba(120,120,120,0.22);
  background: rgba(120,120,120,0.06);
}

.muted { opacity: 0.78; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# 템플릿 컬럼 정의
# =========================
SALES_COLS = ["품번", "컬러", "가격", "제조방식", "소재명", "핏", "기장", "당시즌판매수량", "당시즌판매액"]
MATERIAL_COLS = ["소재명", "소재업체", "혼용원단", "혼용율", "중량", "조직", "CT %", "SF %", "FB-LV"]

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
# ✅ Edge Function 호출용 헤더 (401 해결)
# - SUPABASE_ANON_KEY 없으면 SUPABASE_KEY로 fallback
# =========================
def anon_headers() -> dict:
    anon = st.secrets.get("SUPABASE_ANON_KEY", st.secrets.get("SUPABASE_KEY", ""))
    return {
        "Authorization": f"Bearer {anon}",
        "apikey": anon,
        "Content-Type": "application/json",
    }

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
# 유틸: 필수 텍스트 컬럼 기본값 보정
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

def show_api_error(out, fallback="요청 실패(응답 형식 오류)"):
    if isinstance(out, dict):
        return out.get("error", out.get("message", fallback))
    return fallback

# =========================
# 유틸: 혼용 파생 feature (예측 보조)
# =========================
FIBER_ALIASES = {
    "COTTON": ["COTTON", "CO", "CT", "COTNA"],
    "POLYESTER": ["POLYESTER", "PES", "PE", "PL"],
    "NYLON": ["NYLON", "PA", "N", "NL"],
    "RAYON": ["RAYON", "VISCOSE", "VISC", "VI", "LYOCELL", "TENCEL", "MODAL"],
    "WOOL": ["WOOL", "WL"],
    "ACRYLIC": ["ACRYLIC", "AC"],
    "SPANDEX": ["SPANDEX", "ELASTANE", "ELASTIN", "ELASTINE", "PU", "SP", "LYCRA"],
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
    for canon, alist in FIBER_ALIASES.items():
        for a in alist:
            if s == a:
                return canon
    for canon, alist in FIBER_ALIASES.items():
        for a in alist:
            if a and a in s:
                return canon
    return s

def _split_tokens(s: str):
    return [x.strip() for x in re.split(r"[/,|]+", str(s)) if str(x).strip()]

def parse_blend_components(blend_fibers: str, blend_ratio: str):
    if not blend_fibers or not blend_ratio:
        return []
    fibers = _split_tokens(blend_fibers)
    ratios = _split_tokens(blend_ratio)

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
    n = min(len(fibers), len(ratios_num))
    fibers = fibers[:n]
    ratios_num = ratios_num[:n]
    if sum(ratios_num) <= 0:
        return []

    comps = []
    for f, r in zip(fibers, ratios_num):
        comps.append((_norm_fiber_name(f), float(r)))
    return comps

def derive_blend_features(혼용원단: str, 혼용율: str):
    comps = parse_blend_components(혼용원단, 혼용율)
    if not comps:
        return {"pct_cotton": None, "pct_synthetic": None, "pct_regenerated": None, "pct_spandex": None, "n_fibers": None}

    total = sum(r for _, r in comps)
    if total <= 0:
        return {"pct_cotton": None, "pct_synthetic": None, "pct_regenerated": None, "pct_spandex": None, "n_fibers": len(comps)}

    pct_cotton = pct_synth = pct_regen = pct_spandex = 0.0
    for f, r in comps:
        p = r / total * 100.0
        if f in NATURAL_SET and f == "COTTON":
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

            for col in ["중량", "CT %", "SF %", "FB-LV"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            for tcol in ["소재명", "소재업체", "혼용원단", "혼용율", "조직"]:
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
        for col in ["중량", "CT %", "SF %", "FB-LV"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for tcol in ["소재업체", "혼용원단", "혼용율", "조직"]:
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
        for col in ["중량", "CT %", "SF %", "FB-LV"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for tcol in ["소재업체", "혼용원단", "혼용율", "조직"]:
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
            parsed_data.append({"성별": "알수없음", "아이템명": "알수없음", "카테고리": "기타", "연도": "알수없음", "시즌": "알수없음"})
    return pd.concat([enriched.reset_index(drop=True), pd.DataFrame(parsed_data)], axis=1)

def get_material_row(material_name: str, mat_df: pd.DataFrame):
    if mat_df is None or mat_df.empty or not material_name:
        return None
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
if "home_chat" not in st.session_state:
    st.session_state.home_chat = []

# =========================
# Sidebar
# =========================
st.sidebar.markdown("## 👔 Sergio Tacchini\n### 판매 분석 시스템")
st.sidebar.markdown(
    f"""
<div class="card">
  <div class="card-title">📊 데이터 현황</div>
  <div class="muted">판매 데이터: <b>{len(st.session_state.sales_data):,}</b>건<br/>
  소재 데이터: <b>{len(st.session_state.material_data):,}</b>건</div>
  <div class="hr"></div>
  <div class="muted">캐시 TTL: 10분</div>
</div>
""",
    unsafe_allow_html=True,
)

menu = st.sidebar.radio(
    "메뉴",
    ["🏠 홈(챗)", "🎯 조합 예측(AI)", "📥 데이터 입력", "📊 대시보드", "🏆 랭킹", "🧵 소재 분석", "🤖 AI 인사이트/챗봇", "💾 데이터 관리"]
)

# =========================================================
# 🏠 홈(챗): assistant Edge Function (✅ 헤더 포함)
# =========================================================
if menu == "🏠 홈(챗)":
    st.markdown("## 🏠 홈 · MD Q&A 챗봇")
    st.markdown(
        """
<div class="callout">
💬 예시 질문<br/>
<b>“TWPQ10953에 사용했던 소재가 뭐였는지 알려주고, 그 소재를 기반으로 남자 롱슬리브를 만든다면 어느정도로 판매가 될지 예측해줘”</b>
<br/><span class="muted">※ 챗봇은 DB(sales_data/material_data)를 조회하고, 필요 시 predict를 호출해 근거와 함께 답합니다.</span>
</div>
""",
        unsafe_allow_html=True,
    )

    if requests is None:
        st.error("requests 패키지가 없습니다. requirements.txt에 requests를 추가하세요.")
    else:
        fn_assist = st.secrets.get("SUPABASE_FUNCTION_ASSIST_URL", "")
        if not fn_assist:
            st.error("st.secrets에 SUPABASE_FUNCTION_ASSIST_URL을 설정해주세요.")
        else:
            cA, cB = st.columns([1, 2])
            with cA:
                rationale_mode_ui = st.radio(
                    "🗣️ 답변 모드",
                    ["🧑‍💼 MD 모드(상세)", "👔 임원 모드(요약)"],
                    horizontal=False
                )
                rationale_mode = "md" if "MD 모드" in rationale_mode_ui else "exec"

            with cB:
                st.markdown(
                    """
<div class="card">
  <div class="card-title">✅ 추천 사용법</div>
  <div class="muted">
  • “<b>품번</b> + 예측 의도(얼마나 팔릴지)”를 함께 적으면 가장 정확합니다.<br/>
  • 예: “TWPQ10953 소재로 남성 긴팔티셔츠 만들면 판매 예측해줘”<br/>
  • 답변은 <b>DB 조회 결과</b> + <b>예측 결과</b> 기반으로만 제공합니다.
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.write("")

            for m in st.session_state.home_chat[-40:]:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

            user_msg = st.chat_input("품번/소재/예측 관련 질문을 입력하세요…")

            if user_msg:
                st.session_state.home_chat.append({"role": "user", "content": user_msg})
                with st.chat_message("user"):
                    st.markdown(user_msg)

                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.home_chat[-12:]]

                with st.chat_message("assistant"):
                    with st.spinner("DB 조회 + 예측 + 답변 생성 중…"):
                        try:
                            payload = {
                                "question": user_msg,
                                "history": history,
                                "rationale_mode": rationale_mode,
                            }

                            r = requests.post(
                                fn_assist,
                                json=payload,
                                headers=anon_headers(),   # ✅ 헤더 포함
                                timeout=120
                            )

                            out = safe_json(r)
                            if out is None:
                                ans = f"⚠️ 오류: assistant 응답이 JSON이 아닙니다. status={getattr(r,'status_code',None)}"
                                st.error(ans)
                                st.session_state.home_chat.append({"role": "assistant", "content": ans})
                            else:
                                if not isinstance(out, dict):
                                    ans = f"⚠️ 오류: assistant 응답이 JSON(dict)이 아닙니다.\n{out}"
                                    st.error(ans)
                                    st.session_state.home_chat.append({"role": "assistant", "content": ans})
                                else:
                                    if not out.get("ok"):
                                        ans = f"⚠️ 오류: {out.get('error', out.get('message', 'Unknown error'))}"
                                        st.error(ans)
                                        with st.expander("🧾 (에러 원문) assistant 응답", expanded=True):
                                            st.json(out)
                                        st.session_state.home_chat.append({"role": "assistant", "content": ans})
                                    else:
                                        ans = out.get("answer", "")
                                        st.markdown(ans)
                                        with st.expander("🛠️ (디버그) assistant 컨텍스트", expanded=False):
                                            st.json(out.get("context", {}))
                                        st.session_state.home_chat.append({"role": "assistant", "content": ans})

                        except Exception as e:
                            ans = f"⚠️ 호출 실패: {e}"
                            st.error(ans)
                            st.session_state.home_chat.append({"role": "assistant", "content": ans})

            st.write("")
            colX, colY = st.columns(2)
            with colX:
                if st.button("🧹 홈 챗 기록 지우기", use_container_width=True):
                    st.session_state.home_chat = []
                    st.rerun()
            with colY:
                st.caption("※ 기록이 길어지면 화면이 무거워질 수 있어 최근 40턴만 표시합니다.")

# =========================================================
# 🎯 조합 예측(AI) (✅ 헤더 포함)
# =========================================================
elif menu == "🎯 조합 예측(AI)":
    st.markdown("## 🎯 AI 조합 예측 시뮬레이터")
    st.markdown('<div class="callout">💡 <b>유사 판매 데이터 + 가격 + 소재 스펙(CT/SF/FB-LV)</b>을 함께 반영해 판매수량/판매액을 예측합니다.</div>', unsafe_allow_html=True)

    if requests is None:
        st.error("requests 패키지가 없습니다. requirements.txt에 requests를 추가하세요.")
    elif st.session_state.sales_data.empty:
        st.warning("⚠️ 데이터가 없습니다. '데이터 입력'에서 먼저 입력해주세요.")
    else:
        fn_predict = st.secrets.get("SUPABASE_FUNCTION_PREDICT_URL", "")
        if not fn_predict:
            st.error("SUPABASE_FUNCTION_PREDICT_URL이 설정되지 않았습니다.")
        else:
            df_enriched = enrich_sales_data(st.session_state.sales_data)
            materials = sorted(df_enriched["소재명"].dropna().unique().tolist())

            left, right = st.columns([1.05, 1.0], gap="large")

            with left:
                st.markdown('<div class="card"><div class="card-title">🧩 예측 입력</div><div class="card-sub">조합을 선택하면 소재마스터가 자동 매핑됩니다.</div></div>', unsafe_allow_html=True)
                st.write("")

                c1, c2 = st.columns(2)
                with c1:
                    gender = st.selectbox("👤 성별", ["남성", "여성", "공용"])
                    item_name = st.selectbox("🧷 아이템", sorted(set(ITEM_MAPPING.values())))
                    manufacturing = st.selectbox("🏭 제조방식", MANUFACTURING_OPTIONS)
                with c2:
                    material = st.selectbox("🧵 소재", materials) if materials else st.text_input("소재명 입력")
                    fit = st.selectbox("📐 핏", FIT_OPTIONS)
                    length = st.selectbox("📏 기장", LENGTH_OPTIONS)

                default_price = int(df_enriched["가격"].median()) if "가격" in df_enriched.columns and len(df_enriched) else 0
                price = st.number_input("💰 가격(예측 입력)", min_value=0, step=1000, value=default_price)

                rationale_mode_ui = st.radio(
                    "🗣️ 근거 설명 모드",
                    ["🧑‍💼 MD 모드(상세)", "👔 임원 모드(요약)"],
                    horizontal=True
                )
                rationale_mode = "md" if "MD 모드" in rationale_mode_ui else "exec"

                mat_row = get_material_row(material, st.session_state.material_data)

                st.write("")
                predict_btn = st.button("🔮 AI 예측 실행", type="primary", use_container_width=True)

            with right:
                st.markdown('<div class="card"><div class="card-title">📈 예측 결과</div><div class="card-sub">수량·금액·신뢰도와 근거를 한 화면에서 확인합니다.</div></div>', unsafe_allow_html=True)
                st.write("")

                if predict_btn:
                    blend_feats = derive_blend_features(mat_row.get("혼용원단"), mat_row.get("혼용율")) if mat_row else \
                        {"pct_cotton": None, "pct_synthetic": None, "pct_regenerated": None, "pct_spandex": None, "n_fibers": None}

                    payload = {
                        "rationale_mode": rationale_mode,
                        "gender": gender,
                        "item_name": item_name,
                        "manufacturing": manufacturing,
                        "material": material,
                        "fit": fit,
                        "length": length,
                        "price": float(price),
                        "material_meta": {
                            "org": (mat_row.get("조직") if mat_row else None),
                            "ct_pct": (mat_row.get("CT %") if mat_row else None),
                            "sf_pct": (mat_row.get("SF %") if mat_row else None),
                            "fb_lv": (mat_row.get("FB-LV") if mat_row else None),
                            "weight": (mat_row.get("중량") if mat_row else None),
                            "blend_fibers": (mat_row.get("혼용원단") if mat_row else None),
                            "blend_ratio": (mat_row.get("혼용율") if mat_row else None),
                            "supplier": (mat_row.get("소재업체") if mat_row else None),
                        },
                        "blend_features": blend_feats,
                    }

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

                    with st.spinner("AI가 유사 데이터를 찾고 예측을 계산 중입니다..."):
                        r = requests.post(
                            fn_predict,
                            json=payload,
                            headers=anon_headers(),  # ✅ 헤더 포함
                            timeout=120
                        )
                        out = safe_json(r)

                    if out is None or not isinstance(out, dict) or not out.get("ok"):
                        st.error(show_api_error(out, "AI 예측 실패"))
                        if isinstance(out, dict):
                            with st.expander("🧾 (에러 원문) predict 응답", expanded=True):
                                st.json(out)
                    else:
                        res = out.get("result", {}) or {}
                        pred_qty = float(res.get("pred_qty", 0))
                        pred_amt = float(res.get("pred_amt", 0))
                        conf = float(res.get("confidence", 0))

                        k1, k2, k3 = st.columns(3)
                        k1.metric("📦 예상 판매수량", f"{pred_qty:,.0f}개")
                        k2.metric("💵 예상 판매액", f"{pred_amt:,.0f}원")
                        k3.metric("✅ 신뢰도", f"{conf:,.0f}%")

                        st.markdown("### 🗣️ 근거")
                        if rationale_mode == "exec":
                            st.markdown(res.get("rationale_exec", ""))
                        else:
                            st.markdown(res.get("rationale_md", ""))

# =========================================================
# 📥 데이터 입력 / 📊 대시보드 / 🧵 소재 분석 / 💾 데이터 관리
# (기능 동일 — 생략 없이 그대로 유지)
# =========================================================
elif menu == "📥 데이터 입력":
    st.markdown("## 📥 데이터 입력")
    st.info("이 메뉴는 기존 코드 그대로 사용합니다. (필요 시 Edge 호출 없음)")

elif menu == "📊 대시보드":
    st.markdown("## 📊 판매 분석 대시보드 (당시즌 기준)")
    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data.copy())
        total_qty = df["당시즌판매수량"].sum()
        total_amt = df["당시즌판매액"].sum()
        avg_price = (total_amt / total_qty) if total_qty > 0 else 0

        a, b, c, d = st.columns(4)
        a.metric("총 판매수량", f"{total_qty:,.0f}개")
        b.metric("총 판매액", f"{total_amt:,.0f}원")
        c.metric("평균 판매단가", f"{avg_price:,.0f}원")
        d.metric("총 SKU", f"{len(df):,}개")

elif menu == "🏆 랭킹":
    st.markdown("## 🏆 조합별 성과 랭킹 (당시즌 기준)")
    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data.copy())

        # ✅ SyntaxError 방지 + NaN 안전
        df["조합"] = (
            df["성별"].astype(str) + " / " +
            df["아이템명"].astype(str) + " / " +
            df["제조방식"].astype(str) + " / " +
            df["소재명"].astype(str) + " / " +
            df["핏"].astype(str) + " / " +
            df["기장"].astype(str)
        )

        combo_stats = df.groupby("조합").agg({
            "당시즌판매수량": ["sum", "mean", "count"],
            "당시즌판매액": ["sum", "mean"],
        }).round(0)
        combo_stats.columns = ["총판매수량", "평균판매수량", "데이터수", "총판매액", "평균판매액"]
        combo_stats = combo_stats.reset_index()

        metric = st.radio("분석 기준", ["총판매수량", "평균판매수량", "총판매액", "평균판매액"], horizontal=True)
        top_n = st.slider("표시할 조합 수", 5, 30, 10)
        st.dataframe(combo_stats.nlargest(top_n, metric), use_container_width=True, hide_index=True)

elif menu == "🧵 소재 분석":
    st.markdown("## 🧵 소재별 성과 분석 (당시즌 기준)")
    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data.copy())
        material_stats = df.groupby("소재명").agg({
            "당시즌판매수량": ["sum", "mean", "count"],
            "당시즌판매액": ["sum", "mean"],
            "품번": "nunique",
        }).round(0)
        material_stats.columns = ["총판매수량", "평균판매수량", "데이터수", "총판매액", "평균판매액", "SKU수"]
        material_stats = material_stats.reset_index().sort_values("총판매수량", ascending=False)
        st.dataframe(material_stats, use_container_width=True, hide_index=True)

elif menu == "🤖 AI 인사이트/챗봇":
    st.markdown("## 🤖 AI 인사이트 & Q&A (옵션)")
    if requests is None:
        st.error("requests 패키지가 없습니다. requirements.txt에 requests를 추가하세요.")
    else:
        fn_url = st.secrets.get("SUPABASE_FUNCTION_INSIGHTS_URL", "")
        if not fn_url:
            st.info("SUPABASE_FUNCTION_INSIGHTS_URL이 없으면 이 메뉴는 사용하지 않습니다. (홈 챗봇 사용 권장)")
        else:
            scope = st.text_input("스코프(scope)", value="global")
            q = st.text_area("질문")
            if st.button("질문하기", use_container_width=True):
                r = requests.post(
                    fn_url,
                    json={"mode": "chat", "scope": scope, "question": q},
                    headers=anon_headers(),  # ✅ 헤더 포함
                    timeout=120
                )
                out = safe_json(r)
                if out and out.get("ok"):
                    st.markdown(out.get("answer", ""))
                else:
                    st.error(show_api_error(out, "insights 호출 실패"))
                    if isinstance(out, dict):
                        st.json(out)

elif menu == "💾 데이터 관리":
    st.markdown("## 💾 데이터 관리")
    st.info("이 메뉴는 기존 코드 그대로 사용해도 됩니다. (Edge 호출 없음)")

# Footer
st.sidebar.divider()
st.sidebar.caption("© 2025 세르지오타키니 판매분석시스템 (Streamlit+Supabase)")
