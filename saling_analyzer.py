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
# 컬럼 정의
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
        key = st.secrets["SUPABASE_KEY"]  # anon key
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Supabase 연결 실패: {e}")
        return None

supabase: Client = init_supabase()

# =========================
# Edge Function 호출용 헤더 (401 방지)
# =========================
def anon_headers() -> dict:
    anon = st.secrets.get("SUPABASE_ANON_KEY", st.secrets.get("SUPABASE_KEY", ""))
    return {
        "Authorization": f"Bearer {anon}",
        "apikey": anon,
        "Content-Type": "application/json",
    }

# =========================
# 유틸
# =========================
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None

def show_api_error(out, fallback="요청 실패"):
    if isinstance(out, dict):
        return out.get("error", out.get("message", fallback))
    return fallback

def make_json_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.replace([np.nan, np.inf, -np.inf], None)

def fill_required_text(df: pd.DataFrame, cols, default="UNKNOWN") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = default
        out[c] = out[c].astype(str).replace(["None", "nan"], "").fillna("").apply(lambda x: x.strip())
        out[c] = out[c].replace("", default)
    return out

def is_quota_error(out: dict) -> bool:
    if not isinstance(out, dict):
        return False
    err = str(out.get("error", "")).lower()
    return ("exceeded your current quota" in err) or ("check your plan and billing" in err)

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
    if "롱슬리브" in t or "long sleeve" in t or "긴팔" in t:
        return "긴팔티셔츠"
    if "반팔" in t or "short sleeve" in t:
        return "반팔티셔츠"
    if "맨투맨" in t:
        return "맨투맨"
    if "후드" in t:
        return "후드티"
    if "폴로" in t:
        return "폴로티셔츠"
    return "긴팔티셔츠"

# =========================
# ✅ DB 직접 조회 (해결책 A 강화 버전)
# =========================
def db_get_sales_row_by_code(item_code: str):
    """품번을 최대한 넓게 매칭: eq → ilike → ilike %...%"""
    if supabase is None or not item_code:
        return None
    code = str(item_code).strip()
    try:
        # 1) 정확 일치
        r1 = supabase.table("sales_data").select("*").eq("품번", code).limit(1).execute()
        if r1.data:
            return r1.data[0]
        # 2) 대소문자 무시(패턴 없이도 동작하는 경우가 많지만 안전하게)
        r2 = supabase.table("sales_data").select("*").ilike("품번", code).limit(1).execute()
        if r2.data:
            return r2.data[0]
        # 3) 공백/추가문자 포함 대비
        r3 = supabase.table("sales_data").select("*").ilike("품번", f"%{code}%").limit(1).execute()
        if r3.data:
            return r3.data[0]
    except Exception:
        return None
    return None

def db_suggest_sales_codes(partial: str, limit: int = 10):
    """못 찾을 때 후보 품번 추천"""
    if supabase is None or not partial:
        return []
    q = str(partial).strip()
    try:
        r = supabase.table("sales_data").select("품번").ilike("품번", f"%{q}%").limit(limit).execute()
        return [x.get("품번") for x in (r.data or []) if x.get("품번")]
    except Exception:
        return []

def db_get_material_row_by_name(material_name: str):
    """소재명: eq → ilike → ilike %...%"""
    if supabase is None or not material_name:
        return None
    name = str(material_name).strip()
    try:
        r1 = supabase.table("material_data").select("*").eq("소재명", name).limit(1).execute()
        if r1.data:
            return r1.data[0]
        r2 = supabase.table("material_data").select("*").ilike("소재명", name).limit(1).execute()
        if r2.data:
            return r2.data[0]
        r3 = supabase.table("material_data").select("*").ilike("소재명", f"%{name}%").limit(1).execute()
        if r3.data:
            return r3.data[0]
    except Exception:
        return None
    return None

# =========================
# 데이터 로드(캐시)
# =========================
@st.cache_data(ttl=600)
def load_sales_data():
    if supabase is None:
        return pd.DataFrame(columns=SALES_COLS)
    try:
        res = supabase.table("sales_data").select("*").execute()
        df = pd.DataFrame(res.data or [])
        for c in SALES_COLS:
            if c not in df.columns:
                df[c] = None
        df = df[SALES_COLS].copy()
        for col in ["가격", "당시즌판매수량", "당시즌판매액"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df = fill_required_text(df, ["품번","컬러","제조방식","소재명","핏","기장"])
        return df
    except Exception:
        return pd.DataFrame(columns=SALES_COLS)

@st.cache_data(ttl=600)
def load_material_data():
    if supabase is None:
        return pd.DataFrame(columns=MATERIAL_COLS)
    try:
        res = supabase.table("material_data").select("*").execute()
        df = pd.DataFrame(res.data or [])
        for c in MATERIAL_COLS:
            if c not in df.columns:
                df[c] = None
        df = df[MATERIAL_COLS].copy()
        for col in ["중량","CT %","SF %","FB-LV"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = fill_required_text(df, ["소재명"], default="UNKNOWN_MATERIAL")
        return df
    except Exception:
        return pd.DataFrame(columns=MATERIAL_COLS)

# =========================
# 저장 함수(업로드용)
# =========================
def save_sales_data(new_df: pd.DataFrame) -> bool:
    if supabase is None:
        st.error("Supabase 연결 없음")
        return False
    df = new_df.copy()
    for c in SALES_COLS:
        if c not in df.columns:
            df[c] = None
    df = fill_required_text(df, ["품번","컬러","제조방식","소재명","핏","기장"])
    for col in ["가격", "당시즌판매수량", "당시즌판매액"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = make_json_safe_df(df)
    recs = df[SALES_COLS].to_dict("records")
    if not recs:
        return False
    supabase.table("sales_data").insert(recs).execute()
    st.cache_data.clear()
    return True

def save_material_data(new_df: pd.DataFrame) -> bool:
    if supabase is None:
        st.error("Supabase 연결 없음")
        return False
    df = new_df.copy()
    for c in MATERIAL_COLS:
        if c not in df.columns:
            df[c] = None
    df = fill_required_text(df, ["소재명"], default="UNKNOWN_MATERIAL")
    for col in ["중량","CT %","SF %","FB-LV"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = make_json_safe_df(df)
    recs = df[MATERIAL_COLS].to_dict("records")
    if not recs:
        return False
    supabase.table("material_data").insert(recs).execute()
    st.cache_data.clear()
    return True

def delete_all_sales_data() -> bool:
    if supabase is None:
        return False
    supabase.table("sales_data").delete().neq("id", 0).execute()
    st.cache_data.clear()
    return True

def delete_all_material_data() -> bool:
    if supabase is None:
        return False
    supabase.table("material_data").delete().neq("id", 0).execute()
    st.cache_data.clear()
    return True

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
</div>
""",
    unsafe_allow_html=True,
)

menu = st.sidebar.radio(
    "메뉴",
    ["🏠 홈(챗)", "🎯 조합 예측(AI)", "📥 데이터 입력", "📊 대시보드", "💾 데이터 관리"]
)

# =========================================================
# 🏠 홈(챗): assistant + DB직접조회 fallback
# =========================================================
if menu == "🏠 홈(챗)":
    st.markdown("## 🏠 홈 · MD Q&A 챗봇")
    st.markdown(
        """
<div class="callout">
<b>예시:</b> “TWPQ10953 소재가 뭐였고, 그 소재로 남자 롱슬리브 만들면 얼마나 팔릴지 예측해줘”
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
            st.error("SUPABASE_FUNCTION_ASSIST_URL이 설정되지 않았습니다.")
        else:
            rationale_mode = st.radio("답변 모드", ["md", "exec"], horizontal=True)
            for m in st.session_state.home_chat[-40:]:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

            user_msg = st.chat_input("질문을 입력하세요…")
            if user_msg:
                st.session_state.home_chat.append({"role": "user", "content": user_msg})
                with st.chat_message("user"):
                    st.markdown(user_msg)

                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.home_chat[-12:]]

                with st.chat_message("assistant"):
                    with st.spinner("답변 생성 중…"):
                        # 1) assistant 시도
                        out = None
                        try:
                            r = requests.post(fn_assist, json={"question": user_msg, "history": history, "rationale_mode": rationale_mode},
                                              headers=anon_headers(), timeout=120)
                            out = safe_json(r)
                        except Exception as e:
                            out = {"ok": False, "error": str(e)}

                        if isinstance(out, dict) and out.get("ok"):
                            ans = out.get("answer", "")
                            st.markdown(ans)
                            st.session_state.home_chat.append({"role": "assistant", "content": ans})
                        else:
                            if isinstance(out, dict) and is_quota_error(out):
                                st.warning("⚠️ OpenAI 쿼터/과금 제한 → DB+예측 기반으로 대체 답변을 생성합니다.")
                            else:
                                st.warning("⚠️ assistant 실패 → DB+예측 기반으로 대체 답변을 생성합니다.")
                                with st.expander("assistant 에러 원문", expanded=False):
                                    st.json(out if isinstance(out, dict) else {"raw": str(out)})

                            # 2) DB 직접 조회 (핵심)
                            codes = extract_style_codes(user_msg)
                            style = codes[0] if codes else None

                            sales_row = db_get_sales_row_by_code(style) if style else None
                            if not sales_row and style:
                                suggestions = db_suggest_sales_codes(style, limit=10)
                            else:
                                suggestions = []

                            material_name = str(sales_row.get("소재명", "")).strip() if sales_row else None
                            mat_row = db_get_material_row_by_name(material_name) if material_name else None

                            facts = []
                            facts.append(f"- 질문에서 추출된 품번: **{style or '없음'}**")

                            if sales_row:
                                facts.append(f"- 판매데이터에서 확인: 소재명 **{sales_row.get('소재명','N/A')}**, "
                                             f"제조방식 **{sales_row.get('제조방식','N/A')}**, "
                                             f"핏 **{sales_row.get('핏','N/A')}**, 기장 **{sales_row.get('기장','N/A')}**, "
                                             f"가격 **{int(float(sales_row.get('가격',0) or 0)):,}원**")
                            else:
                                facts.append("- 판매데이터에서 해당 품번을 찾지 못했습니다.")
                                if suggestions:
                                    facts.append(f"- 비슷한 품번 후보: {', '.join([f'`{x}`' for x in suggestions])}")

                            if mat_row:
                                facts.append(f"- 소재마스터에서 확인: 조직 **{mat_row.get('조직','N/A')}**, "
                                             f"CT% **{mat_row.get('CT %','N/A')}**, SF% **{mat_row.get('SF %','N/A')}**, FB-LV **{mat_row.get('FB-LV','N/A')}**")
                            else:
                                if material_name:
                                    facts.append(f"- 소재마스터에서 `{material_name}` 매칭 실패(소재명 불일치 가능)")
                                else:
                                    facts.append("- 소재명을 확인할 판매데이터가 없어 소재마스터 조회 불가")

                            pred_block = ""
                            if wants_prediction(user_msg) and fn_predict and material_name:
                                g = infer_gender(user_msg)
                                item = infer_item_name(user_msg)
                                manu = str(sales_row.get("제조방식", "KNIT")) if sales_row else "KNIT"
                                fit = str(sales_row.get("핏", "REGULAR")) if sales_row else "REGULAR"
                                length = str(sales_row.get("기장", "REGULAR")) if sales_row else "REGULAR"
                                price = float(sales_row.get("가격", 0) or 0) if sales_row else 0

                                pred_payload = {
                                    "rationale_mode": rationale_mode,
                                    "gender": g,
                                    "item_name": item,
                                    "manufacturing": manu,
                                    "material": material_name,
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
                                    pos = pr.get("positioning", {}) or {}
                                    pred_block = f"""
### 🔮 예측 결과
- 예상 판매수량: **{int(pr.get('pred_qty', 0))}개**
- 예상 판매액: **{int(pr.get('pred_amt', 0)):,}원**
- 신뢰도: **{int(pr.get('confidence', 0))}%**
- 보수/기본/공격: **{pos.get('low','N/A')} / {pos.get('base','N/A')} / {pos.get('high','N/A')}**

#### 근거
{pr.get('rationale_exec') if rationale_mode=='exec' else pr.get('rationale_md')}
"""
                                else:
                                    pred_block = f"### 🔮 예측\n⚠️ 실패: {show_api_error(pout, 'Unknown error')}"

                            ans = f"""
### ✅ 대체 답변(DB+예측 기반)
{chr(10).join(facts)}

{pred_block}

### ➕ 수정 포인트(가장 중요)
- `TWPQ10953`가 DB에 있는데 못 찾으면 **데이터 관리 → 판매 데이터 표에서 품번에 공백/소문자/오타가 있는지 확인**해주세요.
- 소재가 매칭 안 되면 **material_data의 소재명과 sales_data의 소재명이 정확히 동일**해야 합니다.
"""
                            st.markdown(ans)
                            st.session_state.home_chat.append({"role": "assistant", "content": ans})

            if st.button("🧹 홈 챗 기록 지우기", use_container_width=True):
                st.session_state.home_chat = []
                st.rerun()

# =========================================================
# 🎯 조합 예측(AI)
# =========================================================
elif menu == "🎯 조합 예측(AI)":
    st.title("🎯 조합 예측(AI)")
    if requests is None:
        st.error("requests 필요")
    else:
        fn_predict = st.secrets.get("SUPABASE_FUNCTION_PREDICT_URL", "")
        if not fn_predict:
            st.error("SUPABASE_FUNCTION_PREDICT_URL이 없습니다.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("성별", ["남성","여성","공용"])
                item_name = st.text_input("아이템명", value="긴팔티셔츠")
                manufacturing = st.selectbox("제조방식", ["KNIT","WOVEN","CUT&SEW"])
                material = st.text_input("소재명", value="")
                fit = st.selectbox("핏", ["REGULAR","SEMI-OVER","OVER"])
                length = st.selectbox("기장", ["REGULAR","LONG","CROP"])
                price = st.number_input("가격", min_value=0, step=1000, value=149000)
                mode = st.radio("근거 모드", ["md","exec"], horizontal=True)
                run = st.button("예측 실행", type="primary")

            with c2:
                if run:
                    mat_row = db_get_material_row_by_name(material)
                    payload = {
                        "rationale_mode": mode,
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
                    rr = requests.post(fn_predict, json=payload, headers=anon_headers(), timeout=120)
                    out = safe_json(rr)
                    if isinstance(out, dict) and out.get("ok"):
                        res = out.get("result", {}) or {}
                        st.metric("예상 판매수량", f"{int(res.get('pred_qty',0))}개")
                        st.metric("예상 판매액", f"{int(res.get('pred_amt',0)):,}원")
                        st.metric("신뢰도", f"{int(res.get('confidence',0))}%")
                        st.markdown(res.get("rationale_exec") if mode=="exec" else res.get("rationale_md"))
                    else:
                        st.error(show_api_error(out, "예측 실패"))
                        if isinstance(out, dict):
                            st.json(out)

# =========================================================
# 📥 데이터 입력
# =========================================================
elif menu == "📥 데이터 입력":
    st.title("📥 데이터 입력")
    tab1, tab2 = st.tabs(["판매 업로드", "소재 업로드"])

    with tab1:
        up = st.file_uploader("판매 Excel 업로드", type=["xlsx","xls"])
        if up:
            df = pd.read_excel(up)
            df = make_json_safe_df(df)
            miss = [c for c in SALES_COLS if c not in df.columns]
            if miss:
                st.error(f"컬럼 누락: {miss}")
            else:
                st.dataframe(df.head(30), use_container_width=True)
                if st.button("판매 저장(추가 Insert)"):
                    if save_sales_data(df):
                        st.session_state.sales_data = load_sales_data()
                        st.success("완료")
                        st.rerun()

    with tab2:
        up = st.file_uploader("소재 Excel 업로드", type=["xlsx","xls"], key="mat_up")
        if up:
            df = pd.read_excel(up)
            df = make_json_safe_df(df)
            miss = [c for c in MATERIAL_COLS if c not in df.columns]
            if miss:
                st.error(f"컬럼 누락: {miss}")
            else:
                st.dataframe(df.head(30), use_container_width=True)
                if st.button("소재 저장(추가 Insert)"):
                    if save_material_data(df):
                        st.session_state.material_data = load_material_data()
                        st.success("완료")
                        st.rerun()

# =========================================================
# 📊 대시보드
# =========================================================
elif menu == "📊 대시보드":
    st.title("📊 대시보드")
    df = st.session_state.sales_data.copy()
    if df.empty:
        st.info("데이터 없음")
    else:
        st.metric("총 판매수량", f"{df['당시즌판매수량'].sum():,.0f}개")
        st.metric("총 판매액", f"{df['당시즌판매액'].sum():,.0f}원")

# =========================================================
# 💾 데이터 관리 (✅ 복구: 조회/편집/다운로드/삭제)
# =========================================================
elif menu == "💾 데이터 관리":
    st.title("💾 데이터 관리")
    tab1, tab2, tab3 = st.tabs(["📊 데이터 확인/편집", "📥 다운로드", "🗑️ 삭제"])

    with tab1:
        st.subheader("판매 데이터")
        st.caption("현재 DB에 입력된 데이터를 확인/편집할 수 있습니다. (편집은 화면에서만, 저장 기능은 필요시 추가)")
        st.dataframe(st.session_state.sales_data, use_container_width=True, height=360)

        st.subheader("소재 데이터")
        st.dataframe(st.session_state.material_data, use_container_width=True, height=360)

        if st.button("🔄 DB 새로고침(캐시 초기화)", use_container_width=True):
            st.cache_data.clear()
            st.session_state.sales_data = load_sales_data()
            st.session_state.material_data = load_material_data()
            st.success("새로고침 완료")
            st.rerun()

    with tab2:
        st.subheader("판매 데이터 다운로드")
        if not st.session_state.sales_data.empty:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                st.session_state.sales_data.to_excel(writer, index=False, sheet_name="sales_data")
            st.download_button(
                "⬇️ 판매 데이터 Excel 다운로드",
                buf.getvalue(),
                f"sales_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("판매 데이터 없음")

        st.subheader("소재 데이터 다운로드")
        if not st.session_state.material_data.empty:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                st.session_state.material_data.to_excel(writer, index=False, sheet_name="material_data")
            st.download_button(
                "⬇️ 소재 데이터 Excel 다운로드",
                buf.getvalue(),
                f"material_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("소재 데이터 없음")

    with tab3:
        st.warning("⚠️ 주의: 삭제는 복구 불가")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ 판매 전체 삭제", use_container_width=True):
                if delete_all_sales_data():
                    st.cache_data.clear()
                    st.session_state.sales_data = load_sales_data()
                    st.success("삭제 완료")
                    st.rerun()
        with c2:
            if st.button("🗑️ 소재 전체 삭제", use_container_width=True):
                if delete_all_material_data():
                    st.cache_data.clear()
                    st.session_state.material_data = load_material_data()
                    st.success("삭제 완료")
                    st.rerun()

st.sidebar.divider()
st.sidebar.caption("© 2025 세르지오타키니 판매분석시스템 (Streamlit+Supabase)")
