# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import io
import re

from supabase import create_client, Client

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

.badge {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(120,120,120,0.28);
  font-size: 0.82rem; margin-right: 6px; margin-top: 6px;
}
.badge-green { background: rgba(24,198,131,0.12); border-color: rgba(24,198,131,0.35); }
.badge-blue  { background: rgba(64,150,255,0.12); border-color: rgba(64,150,255,0.35); }
.badge-amber { background: rgba(255,190,66,0.13); border-color: rgba(255,190,66,0.40); }
.badge-red   { background: rgba(255,86,86,0.12); border-color: rgba(255,86,86,0.38); }

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
# ✅ Edge Function 호출용 헤더 (401 방지)
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

def is_quota_error(out: dict) -> bool:
    """assistant에서 OpenAI quota 초과가 날 때 error 문자열에 포함"""
    if not isinstance(out, dict):
        return False
    err = str(out.get("error", "")).lower()
    return ("exceeded your current quota" in err) or ("check your plan and billing" in err)

# =========================
# 유틸: 질문에서 품번 추출 / 의도 파악 (Fallback 용)
# =========================
def extract_style_codes(text: str):
    if not text:
        return []
    s = text.upper()
    # 대문자+숫자 조합 8~14자 정도 (회사 코드 다양성 고려)
    matches = re.findall(r"\b[A-Z]{1,4}[A-Z0-9]{6,12}\b", s)
    # 중복 제거, 상위 3개
    out = []
    for m in matches:
        if m not in out:
            out.append(m)
    return out[:3]

def wants_prediction(text: str) -> bool:
    if not text:
        return False
    keys = ["예측", "얼마나", "판매", "수량", "판매액", "팔릴", "보수", "공격", "베이스", "base", "low", "high"]
    return any(k in text for k in keys)

def infer_gender(text: str) -> str:
    if not text:
        return "남성"
    if "여성" in text or "여자" in text:
        return "여성"
    if "공용" in text or "유니" in text or "unisex" in text.lower():
        return "공용"
    return "남성"

def infer_item_name(text: str) -> str:
    if not text:
        return "긴팔티셔츠"
    t = text.lower()
    if "롱슬리브" in t or "long sleeve" in t or "긴팔" in t or "롱 슬리브" in t:
        return "긴팔티셔츠"
    if "반팔" in t or "숏슬리브" in t or "short sleeve" in t:
        return "반팔티셔츠"
    if "맨투맨" in t:
        return "맨투맨"
    if "후드" in t:
        return "후드티"
    if "폴로" in t:
        return "폴로티셔츠"
    return "긴팔티셔츠"

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
# 🏠 홈(챗): assistant Edge Function + Fallback
# =========================================================
if menu == "🏠 홈(챗)":
    st.markdown("## 🏠 홈 · MD Q&A 챗봇")
    st.markdown(
        """
<div class="callout">
💬 예시 질문<br/>
<b>“TWPQ10953에 사용했던 소재가 뭐였는지 알려주고, 그 소재를 기반으로 남자 롱슬리브를 만든다면 어느정도로 판매가 될지 예측해줘”</b>
<br/><span class="muted">※ 기본은 assistant(LLM) 답변. 단, LLM 쿼터 초과 시에도 DB+예측 기반 Fallback 답변을 제공합니다.</span>
</div>
""",
        unsafe_allow_html=True,
    )

    if requests is None:
        st.error("requests 패키지가 없습니다. requirements.txt에 requests를 추가하세요.")
    else:
        fn_assist = st.secrets.get("SUPABASE_FUNCTION_ASSIST_URL", "")
        fn_predict = st.secrets.get("SUPABASE_FUNCTION_PREDICT_URL", "")
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
  • 예: “TWPQ10953 소재로 남성 롱슬리브 만들면 판매 예측해줘”
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
                    with st.spinner("답변 생성 중… (LLM 실패 시 자동 Fallback)"):
                        ans = ""
                        try:
                            # 1) assistant 호출 (헤더 포함)
                            payload = {"question": user_msg, "history": history, "rationale_mode": rationale_mode}
                            r = requests.post(fn_assist, json=payload, headers=anon_headers(), timeout=120)
                            out = safe_json(r)

                            if isinstance(out, dict) and out.get("ok"):
                                ans = out.get("answer", "")
                                st.markdown(ans)
                                with st.expander("🛠️ (디버그) assistant 컨텍스트", expanded=False):
                                    st.json(out.get("context", {}))
                                st.session_state.home_chat.append({"role": "assistant", "content": ans})
                                raise SystemExit  # 정상 종료(아래 fallback 스킵)

                            # 2) assistant 실패 → quota 초과면 Fallback
                            if isinstance(out, dict) and is_quota_error(out):
                                st.warning("⚠️ OpenAI API 쿼터/과금 제한으로 LLM 답변이 불가하여, DB+예측 기반으로 대체 답변을 생성합니다.")
                            else:
                                # quota가 아니면 원문 보여주고 fallback도 시도(가능하면)
                                st.warning("⚠️ assistant 응답이 실패하여, DB+예측 기반으로 대체 답변을 생성합니다.")
                                with st.expander("🧾 (assistant 에러 원문)", expanded=False):
                                    st.json(out if isinstance(out, dict) else {"raw": str(out)})

                            # 3) Fallback: DB 조회 + (필요 시) predict 호출
                            codes = extract_style_codes(user_msg)
                            style = codes[0] if codes else None

                            # 3-1) sales_data에서 품번 찾기
                            sales_row = None
                            if style:
                                sdf = st.session_state.sales_data
                                hit = sdf[sdf["품번"].astype(str).str.strip().str.upper() == style.upper()]
                                if not hit.empty:
                                    sales_row = hit.iloc[0].to_dict()

                            # 3-2) material_data에서 소재 찾기
                            mat_row = None
                            material_name = None
                            if sales_row:
                                material_name = str(sales_row.get("소재명", "")).strip()
                            if material_name:
                                mdf = st.session_state.material_data
                                hitm = mdf[mdf["소재명"].astype(str).str.strip() == material_name]
                                if not hitm.empty:
                                    mat_row = hitm.iloc[0].to_dict()

                            # 3-3) 예측 필요하면 predict 호출
                            pred_block = ""
                            if wants_prediction(user_msg) and fn_predict:
                                g = infer_gender(user_msg)
                                item = infer_item_name(user_msg)

                                manu = str((sales_row or {}).get("제조방식", "KNIT")).strip() or "KNIT"
                                fit = str((sales_row or {}).get("핏", "REGULAR")).strip() or "REGULAR"
                                length = str((sales_row or {}).get("기장", "REGULAR")).strip() or "REGULAR"
                                price = float((sales_row or {}).get("가격", 0) or 0)

                                pred_payload = {
                                    "rationale_mode": rationale_mode,
                                    "gender": g,
                                    "item_name": item,
                                    "manufacturing": manu,
                                    "material": material_name or "UNKNOWN",
                                    "fit": fit,
                                    "length": length,
                                    "price": price,
                                    "material_meta": {
                                        "org": (mat_row or {}).get("조직"),
                                        "ct_pct": (mat_row or {}).get("CT %"),
                                        "sf_pct": (mat_row or {}).get("SF %"),
                                        "fb_lv": (mat_row or {}).get("FB-LV"),
                                        "weight": (mat_row or {}).get("중량"),
                                        "blend_fibers": (mat_row or {}).get("혼용원단"),
                                        "blend_ratio": (mat_row or {}).get("혼용율"),
                                        "supplier": (mat_row or {}).get("소재업체"),
                                    }
                                }

                                rr = requests.post(fn_predict, json=pred_payload, headers=anon_headers(), timeout=120)
                                pout = safe_json(rr)

                                if isinstance(pout, dict) and pout.get("ok"):
                                    pr = pout.get("result", {}) or {}
                                    pred_block = f"""
### 🔮 판매 예측 결과
- 예상 판매수량: **{int(pr.get('pred_qty', 0))}개**
- 예상 판매액: **{int(pr.get('pred_amt', 0)):,}원**
- 신뢰도: **{int(pr.get('confidence', 0))}%**

#### 보수/기본/공격
- 보수(LOW): **{(pr.get('positioning', {}) or {}).get('low', 'N/A')}**
- 기본(BASE): **{(pr.get('positioning', {}) or {}).get('base', 'N/A')}**
- 공격(HIGH): **{(pr.get('positioning', {}) or {}).get('high', 'N/A')}**

#### 근거
{pr.get('rationale_exec') if rationale_mode=='exec' else pr.get('rationale_md')}
"""
                                else:
                                    pred_block = f"### 🔮 판매 예측\n⚠️ predict 호출 실패: {show_api_error(pout, 'Unknown error')}"

                            # 3-4) Fallback 답변 템플릿 구성
                            facts = []
                            if style:
                                facts.append(f"- 요청 품번: **{style}**")
                            if sales_row:
                                facts.append(f"- DB 판매데이터 확인: 소재명 **{sales_row.get('소재명','N/A')}**, 제조방식 **{sales_row.get('제조방식','N/A')}**, 핏 **{sales_row.get('핏','N/A')}**, 기장 **{sales_row.get('기장','N/A')}**, 가격 **{int(float(sales_row.get('가격',0) or 0)):,}원**")
                            else:
                                facts.append("- DB 판매데이터에서 해당 품번을 찾지 못했습니다. (품번이 DB에 없거나 오타일 수 있습니다.)")

                            if mat_row:
                                facts.append(f"- DB 소재마스터 확인: 조직 **{mat_row.get('조직','N/A')}**, CT% **{mat_row.get('CT %','N/A')}**, SF% **{mat_row.get('SF %','N/A')}**, FB-LV **{mat_row.get('FB-LV','N/A')}**")
                                facts.append(f"- 혼용: **{mat_row.get('혼용원단','N/A')}** / 혼용율: **{mat_row.get('혼용율','N/A')}**")
                            else:
                                facts.append("- DB 소재마스터에서 해당 소재를 찾지 못했습니다. (소재명 매칭 필요)")

                            ans = f"""
### ✅ 요청 요약
- {user_msg}

### 📌 데이터에서 확인된 사실
{chr(10).join(facts)}

{pred_block}

### ➕ 다음 액션(추천)
- 1) 품번이 DB에 없으면 **판매 데이터에 해당 품번을 1줄만이라도 등록**해주세요.
- 2) 소재명 매칭이 안 되면 **material_data에 동일한 소재명으로 마스터를 추가**하면 예측 정확도가 올라갑니다.
"""
                            st.markdown(ans)
                            st.session_state.home_chat.append({"role": "assistant", "content": ans})

                        except SystemExit:
                            pass
                        except Exception as e:
                            ans = f"⚠️ 호출 실패(예외): {e}"
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
# 🎯 조합 예측(AI)
# =========================================================
elif menu == "🎯 조합 예측(AI)":
    st.markdown("## 🎯 AI 조합 예측 시뮬레이터")

    if requests is None:
        st.error("requests 패키지가 없습니다. requirements.txt에 requests를 추가하세요.")
    else:
        fn_predict = st.secrets.get("SUPABASE_FUNCTION_PREDICT_URL", "")
        if not fn_predict:
            st.error("SUPABASE_FUNCTION_PREDICT_URL이 설정되지 않았습니다.")
        elif st.session_state.sales_data.empty:
            st.warning("⚠️ 데이터가 없습니다. '데이터 입력'에서 먼저 입력해주세요.")
        else:
            df_enriched = enrich_sales_data(st.session_state.sales_data)
            materials = sorted(df_enriched["소재명"].dropna().unique().tolist())

            col1, col2 = st.columns(2, gap="large")

            with col1:
                gender = st.selectbox("성별", ["남성", "여성", "공용"])
                item_name = st.selectbox("아이템", sorted(set(ITEM_MAPPING.values())))
                manufacturing = st.selectbox("제조방식", MANUFACTURING_OPTIONS)
                material = st.selectbox("소재", materials) if materials else st.text_input("소재명 입력")
                fit = st.selectbox("핏", FIT_OPTIONS)
                length = st.selectbox("기장", LENGTH_OPTIONS)
                price = st.number_input("가격", min_value=0, step=1000, value=int(df_enriched["가격"].median()) if len(df_enriched) else 0)

                rationale_mode_ui = st.radio("근거 모드", ["MD(상세)", "임원(요약)"], horizontal=True)
                rationale_mode = "md" if "MD" in rationale_mode_ui else "exec"

                run = st.button("🔮 예측 실행", type="primary", use_container_width=True)

            with col2:
                if run:
                    mat_row = get_material_row(material, st.session_state.material_data)
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
                            "org": (mat_row or {}).get("조직"),
                            "ct_pct": (mat_row or {}).get("CT %"),
                            "sf_pct": (mat_row or {}).get("SF %"),
                            "fb_lv": (mat_row or {}).get("FB-LV"),
                            "weight": (mat_row or {}).get("중량"),
                            "blend_fibers": (mat_row or {}).get("혼용원단"),
                            "blend_ratio": (mat_row or {}).get("혼용율"),
                            "supplier": (mat_row or {}).get("소재업체"),
                        }
                    }

                    with st.spinner("예측 중…"):
                        r = requests.post(fn_predict, json=payload, headers=anon_headers(), timeout=120)
                        out = safe_json(r)

                    if not isinstance(out, dict) or not out.get("ok"):
                        st.error(show_api_error(out, "예측 실패"))
                        if isinstance(out, dict):
                            st.json(out)
                    else:
                        res = out.get("result", {}) or {}
                        st.metric("예상 판매수량", f"{int(res.get('pred_qty', 0)):,}개")
                        st.metric("예상 판매액", f"{int(res.get('pred_amt', 0)):,}원")
                        st.metric("신뢰도", f"{int(res.get('confidence', 0))}%")
                        st.markdown("### 근거")
                        st.markdown(res.get("rationale_exec") if rationale_mode == "exec" else res.get("rationale_md"))

# =========================================================
# 📥 데이터 입력
# =========================================================
elif menu == "📥 데이터 입력":
    st.title("📥 데이터 입력")
    tab1, tab2, tab3 = st.tabs(["📝 수동 입력", "📂 Excel 업로드", "🧵 소재 관리"])

    with tab1:
        st.subheader("판매 데이터 수동 입력")
        col1, col2 = st.columns(2)
        with col1:
            input_code = st.text_input("품번", placeholder="TWPQ10953")
            input_color = st.text_input("컬러", placeholder="BKS")
            input_price_unit = st.number_input("가격", min_value=0, step=1000, value=149000)
            input_manufacturing = st.text_input("제조방식", value="KNIT")
            input_material = st.text_input("소재명", placeholder="HS-17")
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
        template.loc[0] = ["TWPQ10953", "BKS", 149000, "KNIT", "HS-17", "REGULAR", "REGULAR", 15, 2235000]

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
            df_upload = pd.read_excel(uploaded)
            df_upload = make_json_safe_df(df_upload)
            missing = [c for c in SALES_COLS if c not in df_upload.columns]
            if missing:
                st.error(f"❌ 업로드 파일 컬럼 누락: {missing}")
            else:
                st.dataframe(df_upload.head(20), use_container_width=True)
                if st.button("✅ 판매 업로드 적용(추가 Insert)"):
                    if save_sales_data(df_upload):
                        st.session_state.sales_data = load_sales_data()
                        st.success(f"✅ {len(df_upload)}개 추가!")
                        st.rerun()

    with tab3:
        st.subheader("소재 마스터 관리 (CT% / SF% / FB-LV)")
        template_mat = pd.DataFrame(columns=MATERIAL_COLS)
        template_mat.loc[0] = ["HS-17", "BF", "POLYESTER / ELASTINE / MODAL", "50 / 30 / 20", 300, "PLAIN", 0, 100, 3]

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
            df_mat = pd.read_excel(uploaded_mat)
            df_mat = make_json_safe_df(df_mat)

            missing = [c for c in MATERIAL_COLS if c not in df_mat.columns]
            if missing:
                st.error(f"❌ 업로드 파일 컬럼 누락: {missing}")
            else:
                st.dataframe(df_mat.head(30), use_container_width=True)
                if st.button("✅ 소재 업로드 적용(추가 Insert)"):
                    if save_material_data(df_mat):
                        st.session_state.material_data = load_material_data()
                        st.success("✅ 소재 추가 완료!")
                        st.rerun()

# =========================================================
# 📊 대시보드
# =========================================================
elif menu == "📊 대시보드":
    st.title("📊 판매 분석 대시보드 (당시즌 기준)")
    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data.copy())
        total_qty = df["당시즌판매수량"].sum()
        total_amt = df["당시즌판매액"].sum()
        avg_price = (total_amt / total_qty) if total_qty > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 판매수량", f"{total_qty:,.0f}개")
        c2.metric("총 판매액", f"{total_amt:,.0f}원")
        c3.metric("평균 판매단가", f"{avg_price:,.0f}원")
        c4.metric("총 SKU", f"{len(df):,}개")

# =========================================================
# 🏆 랭킹
# =========================================================
elif menu == "🏆 랭킹":
    st.title("🏆 조합별 성과 랭킹 (당시즌 기준)")
    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data.copy())
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
        st.dataframe(combo_stats.sort_values("총판매수량", ascending=False), use_container_width=True, hide_index=True)

# =========================================================
# 🧵 소재 분석
# =========================================================
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
        material_stats.columns = ["총판매수량", "평균판매수량", "데이터수", "총판매액", "평균판매액", "SKU수"]
        material_stats = material_stats.reset_index().sort_values("총판매수량", ascending=False)
        st.dataframe(material_stats, use_container_width=True, hide_index=True)

# =========================================================
# 🤖 AI 인사이트/챗봇 (옵션) - 헤더 포함
# =========================================================
elif menu == "🤖 AI 인사이트/챗봇":
    st.title("🤖 AI 인사이트 & Q&A (옵션)")
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
                    headers=anon_headers(),
                    timeout=120
                )
                out = safe_json(r)
                if out and out.get("ok"):
                    st.markdown(out.get("answer", ""))
                else:
                    st.error(show_api_error(out, "insights 호출 실패"))
                    if isinstance(out, dict):
                        st.json(out)

# =========================================================
# 💾 데이터 관리
# =========================================================
elif menu == "💾 데이터 관리":
    st.title("💾 데이터 관리")
    tab1, tab2, tab3 = st.tabs(["📥 데이터 다운로드", "📊 데이터 확인/편집", "🗑️ 데이터 삭제"])

    with tab1:
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

    with tab2:
        st.caption("편집 후 저장은 '전체 교체'로 반영됩니다(중복/불일치 방지).")
        df_edit_sales = st.data_editor(st.session_state.sales_data[SALES_COLS].copy(), use_container_width=True, num_rows="dynamic")
        if st.button("💾 판매 편집 내용 저장(전체 교체)", type="primary", use_container_width=True):
            if replace_sales_data(df_edit_sales):
                st.session_state.sales_data = load_sales_data()
                st.success("✅ 저장 완료")
                st.rerun()

        st.divider()

        df_edit_mat = st.data_editor(st.session_state.material_data[MATERIAL_COLS].copy(), use_container_width=True, num_rows="dynamic")
        if st.button("💾 소재 편집 내용 저장(전체 교체)", type="primary", use_container_width=True):
            if replace_material_data(df_edit_mat):
                st.session_state.material_data = load_material_data()
                st.success("✅ 저장 완료")
                st.rerun()

    with tab3:
        st.warning("⚠️ **주의**: 삭제된 데이터는 복구할 수 없습니다! 먼저 백업을 다운로드하세요.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ 판매 데이터 전체 삭제", type="secondary"):
                if delete_all_sales_data():
                    st.session_state.sales_data = load_sales_data()
                    st.success("✅ 삭제 완료")
                    st.rerun()
        with c2:
            if st.button("🗑️ 소재 데이터 전체 삭제", type="secondary"):
                if delete_all_material_data():
                    st.session_state.material_data = load_material_data()
                    st.success("✅ 삭제 완료")
                    st.rerun()

# Footer
st.sidebar.divider()
st.sidebar.caption("© 2025 세르지오타키니 판매분석시스템 (Streamlit+Supabase)")
