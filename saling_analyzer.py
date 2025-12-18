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
# ✅ 해결책 A: DB 직접 조회 함수(핵심)
# =========================
def db_get_sales_row_by_code(item_code: str):
    """sales_data에서 품번 기준으로 1건 조회 (DB 직접 조회)"""
    if supabase is None or not item_code:
        return None
    code = str(item_code).strip()
    try:
        res = supabase.table("sales_data").select("*").eq("품번", code).limit(1).execute()
        if res.data:
            return res.data[0]
        # 대소문자/공백 이슈 대비(일부 케이스)
        res2 = supabase.table("sales_data").select("*").ilike("품번", code).limit(1).execute()
        if res2.data:
            return res2.data[0]
    except Exception:
        return None
    return None


def db_get_material_row_by_name(material_name: str):
    """material_data에서 소재명 기준으로 1건 조회 (DB 직접 조회)"""
    if supabase is None or not material_name:
        return None
    name = str(material_name).strip()
    try:
        res = supabase.table("material_data").select("*").eq("소재명", name).limit(1).execute()
        if res.data:
            return res.data[0]
        # 소재명에 추가 문자열/코드가 섞일 때 대비(부분일치)
        res2 = supabase.table("material_data").select("*").ilike("소재명", f"%{name}%").limit(1).execute()
        if res2.data:
            return res2.data[0]
    except Exception:
        return None
    return None

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
    if not isinstance(out, dict):
        return False
    err = str(out.get("error", "")).lower()
    return ("exceeded your current quota" in err) or ("check your plan and billing" in err)

# =========================
# 유틸: 질문 파싱(품번/의도/성별/아이템)
# =========================
def extract_style_codes(text: str):
    if not text:
        return []
    s = text.upper()
    matches = re.findall(r"\b[A-Z]{1,4}[A-Z0-9]{6,12}\b", s)
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
# 데이터 로드 (캐시)
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
    except Exception:
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
    except Exception:
        return pd.DataFrame(columns=MATERIAL_COLS)

# =========================
# 데이터 저장/삭제 (DB insert/replace)
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
            return False
        supabase.table("sales_data").insert(records).execute()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"판매 저장 실패: {e}")
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
            return False
        supabase.table("material_data").insert(records).execute()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"소재 저장 실패: {e}")
        return False

def delete_all_sales_data() -> bool:
    if supabase is None:
        return False
    try:
        supabase.table("sales_data").delete().neq("id", 0).execute()
        st.cache_data.clear()
        return True
    except Exception:
        return False

def delete_all_material_data() -> bool:
    if supabase is None:
        return False
    try:
        supabase.table("material_data").delete().neq("id", 0).execute()
        st.cache_data.clear()
        return True
    except Exception:
        return False

# =========================
# 품번 파싱/매핑 (대시보드용)
# =========================
def parse_item_code(code):
    if not code:
        return None
    code = str(code).strip()
    if len(code) < 8:
        return None
    try:
        return {
            "gender": code[1] if len(code) > 1 else None,
            "item_code": code[2:4] if len(code) >= 4 else None,
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

def enrich_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    parsed_data = []
    for _, row in enriched.iterrows():
        parsed = parse_item_code(row.get("품번"))
        if parsed:
            gender = GENDER_MAPPING.get(parsed.get("gender"), "알수없음")
            item = ITEM_MAPPING.get(parsed.get("item_code"), "알수없음")
            cat = CATEGORY_MAPPING.get(parsed.get("item_code"), "기타")
        else:
            gender, item, cat = "알수없음", "알수없음", "기타"
        parsed_data.append({"성별": gender, "아이템명": item, "카테고리": cat})
    return pd.concat([enriched.reset_index(drop=True), pd.DataFrame(parsed_data)], axis=1)

# =========================
# Session State 초기화
# =========================
if "sales_data" not in st.session_state:
    st.session_state.sales_data = load_sales_data()
if "material_data" not in st.session_state:
    st.session_state.material_data = load_material_data()
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
# 🏠 홈(챗): assistant + DB 직접조회 fallback
# =========================================================
if menu == "🏠 홈(챗)":
    st.markdown("## 🏠 홈 · MD Q&A 챗봇")
    st.markdown(
        """
<div class="callout">
💬 예시 질문<br/>
<b>“TWPQ10953에 사용했던 소재가 뭐였는지 알려주고, 그 소재를 기반으로 남자 롱슬리브를 만든다면 어느정도로 판매가 될지 예측해줘”</b>
<br/><span class="muted">※ LLM 실패(쿼터/과금 제한) 시에도 DB 직접 조회 + 예측 기반으로 답변합니다.</span>
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
                rationale_mode_ui = st.radio("🗣️ 답변 모드", ["🧑‍💼 MD 모드(상세)", "👔 임원 모드(요약)"], horizontal=False)
                rationale_mode = "md" if "MD 모드" in rationale_mode_ui else "exec"
            with cB:
                st.markdown(
                    """
<div class="card">
  <div class="card-title">✅ 추천 사용법</div>
  <div class="muted">• 품번 + “예측” 의도까지 같이 적으면 정확합니다.<br/>• 예: “TWPQ10953 소재로 남성 긴팔티셔츠 만들면 예측해줘”</div>
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
                        # 1) 먼저 assistant 호출 시도
                        ans = ""
                        out = None
                        try:
                            payload = {"question": user_msg, "history": history, "rationale_mode": rationale_mode}
                            r = requests.post(fn_assist, json=payload, headers=anon_headers(), timeout=120)
                            out = safe_json(r)
                        except Exception as e:
                            out = {"ok": False, "error": str(e)}

                        if isinstance(out, dict) and out.get("ok"):
                            ans = out.get("answer", "")
                            st.markdown(ans)
                            st.session_state.home_chat.append({"role": "assistant", "content": ans})
                        else:
                            # 2) 실패 → quota 포함 여부 안내
                            if isinstance(out, dict) and is_quota_error(out):
                                st.warning("⚠️ OpenAI API 쿼터/과금 제한으로 LLM 답변이 불가하여, DB+예측 기반으로 대체 답변을 생성합니다.")
                            else:
                                st.warning("⚠️ assistant 응답이 실패하여, DB+예측 기반으로 대체 답변을 생성합니다.")
                                with st.expander("🧾 (assistant 에러 원문)", expanded=False):
                                    st.json(out if isinstance(out, dict) else {"raw": str(out)})

                            # 3) ✅ 해결책 A: DB 직접 조회로 사실 확인
                            codes = extract_style_codes(user_msg)
                            style = codes[0] if codes else None

                            sales_row = db_get_sales_row_by_code(style) if style else None
                            material_name = str(sales_row.get("소재명", "")).strip() if sales_row else None
                            mat_row = db_get_material_row_by_name(material_name) if material_name else None

                            facts = []
                            if style:
                                facts.append(f"- 요청 품번: **{style}**")
                            if sales_row:
                                facts.append(
                                    f"- 판매데이터: 소재명 **{sales_row.get('소재명','N/A')}**, "
                                    f"제조방식 **{sales_row.get('제조방식','N/A')}**, "
                                    f"핏 **{sales_row.get('핏','N/A')}**, 기장 **{sales_row.get('기장','N/A')}**, "
                                    f"가격 **{int(float(sales_row.get('가격',0) or 0)):,}원**"
                                )
                            else:
                                facts.append("- 판매데이터에서 해당 품번을 찾지 못했습니다. (품번 오타/미등록 가능)")

                            if mat_row:
                                facts.append(
                                    f"- 소재마스터: 조직 **{mat_row.get('조직','N/A')}**, "
                                    f"CT% **{mat_row.get('CT %','N/A')}**, SF% **{mat_row.get('SF %','N/A')}**, FB-LV **{mat_row.get('FB-LV','N/A')}**"
                                )
                                facts.append(f"- 혼용: **{mat_row.get('혼용원단','N/A')}** / 혼용율: **{mat_row.get('혼용율','N/A')}**")
                            else:
                                facts.append("- 소재마스터에서 해당 소재를 찾지 못했습니다. (소재명 매칭 필요)")

                            # 4) 예측이 필요하면 predict 호출 (헤더 포함)
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

                                try:
                                    rr = requests.post(fn_predict, json=pred_payload, headers=anon_headers(), timeout=120)
                                    pout = safe_json(rr)
                                except Exception as e:
                                    pout = {"ok": False, "error": str(e)}

                                if isinstance(pout, dict) and pout.get("ok"):
                                    pr = pout.get("result", {}) or {}
                                    pos = pr.get("positioning", {}) or {}
                                    pred_block = f"""
### 🔮 판매 예측 결과
- 예상 판매수량: **{int(pr.get('pred_qty', 0))}개**
- 예상 판매액: **{int(pr.get('pred_amt', 0)):,}원**
- 신뢰도: **{int(pr.get('confidence', 0))}%**

#### 보수/기본/공격
- 보수(LOW): **{pos.get('low','N/A')}**
- 기본(BASE): **{pos.get('base','N/A')}**
- 공격(HIGH): **{pos.get('high','N/A')}**

#### 근거
{pr.get('rationale_exec') if rationale_mode=='exec' else pr.get('rationale_md')}
"""
                                else:
                                    pred_block = f"### 🔮 판매 예측\n⚠️ predict 호출 실패: {show_api_error(pout, 'Unknown error')}"

                            ans = f"""
### ✅ 요청 요약
- {user_msg}

### 📌 데이터에서 확인된 사실(DB 직접 조회)
{chr(10).join(facts)}

{pred_block}

### ➕ 다음 액션(추천)
- 품번이 미등록이면 **판매 데이터에 해당 품번 1줄을 추가**해주세요.
- 소재마스터 매칭이 안 되면 **material_data에 동일 소재명으로 마스터를 추가**해주세요.
"""
                            st.markdown(ans)
                            st.session_state.home_chat.append({"role": "assistant", "content": ans})

            st.write("")
            if st.button("🧹 홈 챗 기록 지우기", use_container_width=True):
                st.session_state.home_chat = []
                st.rerun()

# =========================================================
# 🎯 조합 예측(AI)
# =========================================================
elif menu == "🎯 조합 예측(AI)":
    st.title("🎯 조합 예측(AI)")
    if requests is None:
        st.error("requests 패키지가 없습니다.")
    else:
        fn_predict = st.secrets.get("SUPABASE_FUNCTION_PREDICT_URL", "")
        if not fn_predict:
            st.error("SUPABASE_FUNCTION_PREDICT_URL이 설정되지 않았습니다.")
        elif st.session_state.sales_data.empty:
            st.warning("판매 데이터가 없습니다.")
        else:
            df_enriched = enrich_sales_data(st.session_state.sales_data.copy())
            materials = sorted(df_enriched["소재명"].dropna().unique().tolist())

            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("성별", ["남성", "여성", "공용"])
                item_name = st.selectbox("아이템", sorted(set(ITEM_MAPPING.values())))
                manufacturing = st.selectbox("제조방식", ["KNIT", "WOVEN", "CUT&SEW"])
                material = st.selectbox("소재", materials) if materials else st.text_input("소재명 입력")
                fit = st.selectbox("핏", ["REGULAR", "SEMI-OVER", "OVER"])
                length = st.selectbox("기장", ["REGULAR", "LONG", "CROP"])
                price = st.number_input("가격", min_value=0, step=1000, value=int(df_enriched["가격"].median()) if len(df_enriched) else 0)
                rationale_mode = st.radio("근거 모드", ["md", "exec"], horizontal=True)
                run = st.button("🔮 예측 실행", type="primary", use_container_width=True)

            with col2:
                if run:
                    mat_row = db_get_material_row_by_name(material)  # DB 직접조회(정확도↑)
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
                        st.markdown(res.get("rationale_exec") if rationale_mode == "exec" else res.get("rationale_md"))

# =========================================================
# 📥 데이터 입력
# =========================================================
elif menu == "📥 데이터 입력":
    st.title("📥 데이터 입력")
    tab1, tab2 = st.tabs(["판매 입력", "소재 입력"])

    with tab1:
        st.subheader("판매 데이터 업로드/추가")
        uploaded = st.file_uploader("판매 Excel 파일 선택", type=["xlsx", "xls"])
        if uploaded:
            df_upload = pd.read_excel(uploaded)
            df_upload = make_json_safe_df(df_upload)
            missing = [c for c in SALES_COLS if c not in df_upload.columns]
            if missing:
                st.error(f"컬럼 누락: {missing}")
            else:
                st.dataframe(df_upload.head(30), use_container_width=True)
                if st.button("✅ 판매 업로드(추가 Insert)"):
                    if save_sales_data(df_upload):
                        st.session_state.sales_data = load_sales_data()
                        st.success("완료")
                        st.rerun()

    with tab2:
        st.subheader("소재 데이터 업로드/추가")
        uploaded = st.file_uploader("소재 Excel 파일 선택", type=["xlsx", "xls"], key="mat_uploader")
        if uploaded:
            df_mat = pd.read_excel(uploaded)
            df_mat = make_json_safe_df(df_mat)
            missing = [c for c in MATERIAL_COLS if c not in df_mat.columns]
            if missing:
                st.error(f"컬럼 누락: {missing}")
            else:
                st.dataframe(df_mat.head(30), use_container_width=True)
                if st.button("✅ 소재 업로드(추가 Insert)"):
                    if save_material_data(df_mat):
                        st.session_state.material_data = load_material_data()
                        st.success("완료")
                        st.rerun()

# =========================================================
# 📊 대시보드
# =========================================================
elif menu == "📊 대시보드":
    st.title("📊 대시보드")
    df = enrich_sales_data(st.session_state.sales_data.copy())
    if df.empty:
        st.info("데이터 없음")
    else:
        st.metric("총 판매수량", f"{df['당시즌판매수량'].sum():,.0f}개")
        st.metric("총 판매액", f"{df['당시즌판매액'].sum():,.0f}원")

# =========================================================
# 🏆 랭킹
# =========================================================
elif menu == "🏆 랭킹":
    st.title("🏆 랭킹")
    df = enrich_sales_data(st.session_state.sales_data.copy())
    if df.empty:
        st.info("데이터 없음")
    else:
        df["조합"] = (
            df["성별"].astype(str) + " / " +
            df["아이템명"].astype(str) + " / " +
            df["제조방식"].astype(str) + " / " +
            df["소재명"].astype(str) + " / " +
            df["핏"].astype(str) + " / " +
            df["기장"].astype(str)
        )
        combo_stats = df.groupby("조합").agg(
            총판매수량=("당시즌판매수량", "sum"),
            평균판매수량=("당시즌판매수량", "mean"),
            데이터수=("당시즌판매수량", "count"),
            총판매액=("당시즌판매액", "sum"),
            평균판매액=("당시즌판매액", "mean"),
        ).round(0).reset_index()
        st.dataframe(combo_stats.sort_values("총판매수량", ascending=False).head(50), use_container_width=True, hide_index=True)

# =========================================================
# 🧵 소재 분석
# =========================================================
elif menu == "🧵 소재 분석":
    st.title("🧵 소재 분석")
    df = enrich_sales_data(st.session_state.sales_data.copy())
    if df.empty:
        st.info("데이터 없음")
    else:
        material_stats = df.groupby("소재명").agg(
            총판매수량=("당시즌판매수량", "sum"),
            평균판매수량=("당시즌판매수량", "mean"),
            데이터수=("당시즌판매수량", "count"),
            총판매액=("당시즌판매액", "sum"),
        ).round(0).reset_index().sort_values("총판매수량", ascending=False)
        st.dataframe(material_stats, use_container_width=True, hide_index=True)

# =========================================================
# 🤖 AI 인사이트/챗봇 (옵션)
# =========================================================
elif menu == "🤖 AI 인사이트/챗봇":
    st.title("🤖 AI 인사이트/챗봇 (옵션)")
    if requests is None:
        st.error("requests 필요")
    else:
        fn_url = st.secrets.get("SUPABASE_FUNCTION_INSIGHTS_URL", "")
        if not fn_url:
            st.info("INSIGHTS URL이 없으면 사용하지 않습니다.")
        else:
            q = st.text_area("질문")
            if st.button("질문하기"):
                r = requests.post(fn_url, json={"mode": "chat", "scope": "global", "question": q}, headers=anon_headers(), timeout=120)
                out = safe_json(r)
                if isinstance(out, dict) and out.get("ok"):
                    st.markdown(out.get("answer", ""))
                else:
                    st.error(show_api_error(out, "실패"))
                    if isinstance(out, dict):
                        st.json(out)

# =========================================================
# 💾 데이터 관리
# =========================================================
elif menu == "💾 데이터 관리":
    st.title("💾 데이터 관리")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 판매 전체 삭제"):
            if delete_all_sales_data():
                st.session_state.sales_data = load_sales_data()
                st.success("완료")
                st.rerun()
    with col2:
        if st.button("🗑️ 소재 전체 삭제"):
            if delete_all_material_data():
                st.session_state.material_data = load_material_data()
                st.success("완료")
                st.rerun()

st.sidebar.divider()
st.sidebar.caption("© 2025 세르지오타키니 판매분석시스템 (Streamlit+Supabase)")
