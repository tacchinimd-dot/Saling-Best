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

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="세르지오 타키니 판매 분석",
    page_icon="👔",
    layout="wide"
)

# =========================================================
# Global CSS
# =========================================================
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.card { border-radius: 16px; padding: 16px; border: 1px solid #ddd; background: #fafafa; }
.card-title { font-weight: 700; margin-bottom: 6px; }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; background:#eee; margin-right:6px; font-size:12px; }
.callout { border-radius: 12px; padding: 14px; background:#f5f7fa; border:1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Constants
# =========================================================
SALES_COLS = ["품번","컬러","가격","제조방식","소재명","핏","기장","당시즌판매수량","당시즌판매액"]
MATERIAL_COLS = ["소재명","소재업체","혼용원단","혼용율","중량","조직","CT %","SF %","FB-LV"]

# =========================================================
# Supabase
# =========================================================
@st.cache_resource
def init_supabase():
    try:
        return create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_KEY"]   # anon key
        )
    except Exception as e:
        st.error(f"Supabase 연결 실패: {e}")
        return None

supabase: Client = init_supabase()

# =========================================================
# Helpers
# =========================================================
def anon_headers():
    anon = st.secrets.get("SUPABASE_ANON_KEY", st.secrets.get("SUPABASE_KEY", ""))
    return {
        "Authorization": f"Bearer {anon}",
        "apikey": anon,
        "Content-Type": "application/json",
    }

def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None

def fill_required_text(df, cols, default="UNKNOWN"):
    for c in cols:
        if c not in df.columns:
            df[c] = default
        df[c] = df[c].astype(str).fillna(default)
    return df

# =========================================================
# Load data
# =========================================================
@st.cache_data(ttl=600)
def load_sales_data():
    if supabase is None:
        return pd.DataFrame(columns=SALES_COLS)
    res = supabase.table("sales_data").select("*").execute()
    df = pd.DataFrame(res.data or [], columns=SALES_COLS)
    return fill_required_text(df, ["품번","컬러","제조방식","소재명","핏","기장"])

@st.cache_data(ttl=600)
def load_material_data():
    if supabase is None:
        return pd.DataFrame(columns=MATERIAL_COLS)
    res = supabase.table("material_data").select("*").execute()
    df = pd.DataFrame(res.data or [], columns=MATERIAL_COLS)
    return fill_required_text(df, ["소재명"])

# =========================================================
# Session state
# =========================================================
if "sales_data" not in st.session_state:
    st.session_state.sales_data = load_sales_data()
if "material_data" not in st.session_state:
    st.session_state.material_data = load_material_data()
if "home_chat" not in st.session_state:
    st.session_state.home_chat = []

# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown("## 👔 Sergio Tacchini")
menu = st.sidebar.radio(
    "메뉴",
    ["🏠 홈(챗)", "🎯 조합 예측(AI)", "📊 대시보드"]
)

# =========================================================
# 🏠 홈(챗) : assistant Edge Function
# =========================================================
if menu == "🏠 홈(챗)":
    st.markdown("## 🏠 MD Q&A 챗봇")
    st.markdown("""
<div class="callout">
예시 질문:<br/>
<b>“TWPQ10953에 사용한 소재가 뭐였고, 그 소재로 남자 롱슬리브 만들면 얼마나 팔릴지 예측해줘”</b>
</div>
""", unsafe_allow_html=True)

    fn_assist = st.secrets.get("SUPABASE_FUNCTION_ASSIST_URL", "")
    if not fn_assist:
        st.error("SUPABASE_FUNCTION_ASSIST_URL이 설정되지 않았습니다.")
    elif requests is None:
        st.error("requests 패키지가 필요합니다.")
    else:
        # 대화 표시
        for m in st.session_state.home_chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_msg = st.chat_input("질문을 입력하세요…")
        if user_msg:
            st.session_state.home_chat.append({"role":"user","content":user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            with st.chat_message("assistant"):
                with st.spinner("DB 조회 + 예측 + 답변 생성 중…"):
                    try:
                        payload = {
                            "question": user_msg,
                            "history": st.session_state.home_chat[-10:],
                            "rationale_mode": "md",
                        }
                        r = requests.post(
                            fn_assist,
                            json=payload,
                            headers=anon_headers(),
                            timeout=120
                        )
                        out = safe_json(r)

                        if not isinstance(out, dict):
                            ans = f"⚠️ assistant 응답이 JSON이 아닙니다: {out}"
                            st.error(ans)
                        elif not out.get("ok"):
                            ans = f"⚠️ 오류: {out.get('error','Unknown error')}"
                            st.error(ans)
                            with st.expander("에러 원문"):
                                st.json(out)
                        else:
                            ans = out.get("answer","")
                            st.markdown(ans)

                        st.session_state.home_chat.append(
                            {"role":"assistant","content":ans}
                        )

                    except Exception as e:
                        ans = f"⚠️ 호출 실패: {e}"
                        st.error(ans)
                        st.session_state.home_chat.append(
                            {"role":"assistant","content":ans}
                        )

        if st.button("🧹 대화 초기화"):
            st.session_state.home_chat = []
            st.rerun()

# =========================================================
# 🎯 조합 예측(AI) : predict Edge Function
# =========================================================
elif menu == "🎯 조합 예측(AI)":
    st.markdown("## 🎯 AI 조합 예측")

    fn_predict = st.secrets.get("SUPABASE_FUNCTION_PREDICT_URL", "")
    if not fn_predict:
        st.error("SUPABASE_FUNCTION_PREDICT_URL이 설정되지 않았습니다.")
    elif requests is None:
        st.error("requests 패키지가 필요합니다.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("성별", ["남성","여성","공용"])
            item_name = st.selectbox("아이템", ["긴팔티셔츠","반팔티셔츠","맨투맨"])
            manufacturing = st.selectbox("제조방식", ["KNIT","WOVEN","CUT&SEW"])
            material = st.selectbox(
                "소재",
                sorted(st.session_state.material_data["소재명"].unique())
                if not st.session_state.material_data.empty else []
            )
            fit = st.selectbox("핏", ["REGULAR","SEMI-OVER","OVER"])
            length = st.selectbox("기장", ["REGULAR","LONG","CROP"])
            price = st.number_input("가격", value=149000, step=1000)

            run = st.button("🔮 예측 실행")

        with col2:
            if run:
                payload = {
                    "gender": gender,
                    "item_name": item_name,
                    "manufacturing": manufacturing,
                    "material": material,
                    "fit": fit,
                    "length": length,
                    "price": price,
                    "rationale_mode": "md"
                }

                with st.spinner("예측 중…"):
                    r = requests.post(
                        fn_predict,
                        json=payload,
                        headers=anon_headers(),
                        timeout=120
                    )
                    out = safe_json(r)

                if not isinstance(out, dict) or not out.get("ok"):
                    st.error(out.get("error","예측 실패") if isinstance(out,dict) else out)
                    if isinstance(out,dict):
                        st.json(out)
                else:
                    res = out["result"]
                    st.metric("예상 판매수량", f"{res.get('pred_qty',0)} 개")
                    st.metric("예상 판매액", f"{res.get('pred_amt',0):,} 원")
                    st.metric("신뢰도", f"{res.get('confidence',0)} %")
                    st.markdown("### 근거")
                    st.markdown(res.get("rationale_md",""))

# =========================================================
# 📊 대시보드
# =========================================================
elif menu == "📊 대시보드":
    st.markdown("## 📊 판매 대시보드")
    if st.session_state.sales_data.empty:
        st.info("판매 데이터가 없습니다.")
    else:
        df = st.session_state.sales_data.copy()
        st.metric("총 판매수량", int(df["당시즌판매수량"].sum()))
        st.metric("총 판매액", int(df["당시즌판매액"].sum()))
        fig = px.bar(df, x="소재명", y="당시즌판매수량")
        st.plotly_chart(fig, use_container_width=True)
