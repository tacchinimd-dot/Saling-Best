# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import io
from supabase import create_client, Client

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="세르지오타키니 판매 분석",
    page_icon="👔",
    layout="wide"
)

# =========================
# 템플릿 컬럼 정의 (신규)
# =========================
SALES_COLS = ["품번", "컬러", "가격", "제조방식", "소재명", "핏", "기장", "당시즌판매수량", "당시즌판매액"]
MATERIAL_COLS = ["소재명", "소재업체", "혼용원단", "혼용율", "중량", "두께", "밀도", "GU", "RA", "SA"]

# =========================
# Supabase 연결
# =========================
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Supabase 연결 실패: {e}")
        return None

supabase: Client = init_supabase()

# =========================
# 유틸: JSON-safe 변환 (NaN/Inf 제거)
# =========================
def make_json_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Supabase insert(JSON)에서 터지는 NaN/Inf를 None으로 치환"""
    if df is None or df.empty:
        return df
    out = df.copy()
    out = out.replace([np.nan, np.inf, -np.inf], None)
    return out

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
            for col in ["혼용율", "중량", "밀도", "GU", "RA", "SA"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["두께"] = df["두께"].astype(str)
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

        # 컬럼 보정
        for c in SALES_COLS:
            if c not in df.columns:
                df[c] = None

        # 숫자형 보정
        for col in ["가격", "당시즌판매수량", "당시즌판매액"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # ✅ JSON-safe (NaN/Inf 제거)  ← (기존 코드에서 들여쓰기 깨져 SyntaxError 원인)
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

def save_material_data(new_df: pd.DataFrame) -> bool:
    if supabase is None:
        st.error("❌ Supabase 연결이 없습니다.")
        return False
    try:
        df = new_df.copy()

        for c in MATERIAL_COLS:
            if c not in df.columns:
                df[c] = None

        for col in ["혼용율", "중량", "밀도", "GU", "RA", "SA"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["두께"] = df["두께"].astype(str)

        # ✅ JSON-safe
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
# 예측 로직 (당시즌 기준)
# =========================
def predict_combination(gender, item_name, manufacturing, material, fit, length):
    if st.session_state.sales_data.empty:
        return None

    df = enrich_sales_data(st.session_state.sales_data)

    exact = df[
        (df["성별"] == gender) &
        (df["아이템명"] == item_name) &
        (df["제조방식"] == manufacturing) &
        (df["소재명"] == material) &
        (df["핏"] == fit) &
        (df["기장"] == length)
    ]
    if not exact.empty:
        return {
            "type": "exact",
            "avg_quantity": exact["당시즌판매수량"].mean(),
            "avg_price": exact["당시즌판매액"].mean(),
            "count": len(exact),
            "confidence": 95
        }

    similar = df[
        (df["성별"] == gender) &
        (df["아이템명"] == item_name) &
        (df["제조방식"] == manufacturing) &
        (df["소재명"] == material) &
        (df["핏"] == fit)
    ]
    if not similar.empty:
        return {"type": "similar_5",
                "avg_quantity": similar["당시즌판매수량"].mean(),
                "avg_price": similar["당시즌판매액"].mean(),
                "count": len(similar),
                "confidence": 80}

    similar = df[
        (df["성별"] == gender) &
        (df["아이템명"] == item_name) &
        (df["제조방식"] == manufacturing) &
        (df["소재명"] == material)
    ]
    if not similar.empty:
        return {"type": "similar_4",
                "avg_quantity": similar["당시즌판매수량"].mean(),
                "avg_price": similar["당시즌판매액"].mean(),
                "count": len(similar),
                "confidence": 65}

    similar = df[
        (df["성별"] == gender) &
        (df["아이템명"] == item_name) &
        (df["제조방식"] == manufacturing)
    ]
    if not similar.empty:
        return {"type": "similar_3",
                "avg_quantity": similar["당시즌판매수량"].mean(),
                "avg_price": similar["당시즌판매액"].mean(),
                "count": len(similar),
                "confidence": 45}

    return None

# =========================
# Session State 초기화
# =========================
if "sales_data" not in st.session_state:
    st.session_state.sales_data = load_sales_data()
if "material_data" not in st.session_state:
    st.session_state.material_data = load_material_data()

# =========================
# Sidebar / Menu
# =========================
st.sidebar.title("👔 세르지오타키니")
st.sidebar.markdown("### 판매 분석 시스템")

menu = st.sidebar.radio(
    "메뉴",
    ["🎯 조합 예측", "📥 데이터 입력", "📊 대시보드", "🏆 랭킹", "🧵 소재 분석", "💾 데이터 관리"]
)

# =========================
# 1) 조합 예측
# =========================
if menu == "🎯 조합 예측":
    st.title("🎯 조합 예측 시뮬레이터 (당시즌 기준)")

    if st.session_state.sales_data.empty:
        st.warning("⚠️ 데이터가 없습니다. '데이터 입력'에서 먼저 입력해주세요.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 조합 입력")
            gender = st.selectbox("성별", ["남성", "여성", "공용"])
            item_name = st.selectbox("아이템", sorted(set(ITEM_MAPPING.values())))
            manufacturing = st.selectbox("제조방식", MANUFACTURING_OPTIONS)

            df_enriched = enrich_sales_data(st.session_state.sales_data)
            materials = sorted(df_enriched["소재명"].dropna().unique().tolist())
            material = st.selectbox("소재", materials) if materials else st.text_input("소재명 입력")

            fit = st.selectbox("핏", FIT_OPTIONS)
            length = st.selectbox("기장", LENGTH_OPTIONS)

            predict_btn = st.button("🔮 예측 결과 보기", type="primary", use_container_width=True)

        with col2:
            st.subheader("📊 예측 결과")
            if predict_btn:
                result = predict_combination(gender, item_name, manufacturing, material, fit, length)
                if result:
                    st.success("✅ 예측 완료!")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("예상 당시즌판매수량", f"{result['avg_quantity']:.0f}개")
                    c2.metric("예상 당시즌판매액", f"{result['avg_price']:,.0f}원")
                    c3.metric("신뢰도", f"{result['confidence']}%")
                    st.divider()
                    label = {
                        "exact": "✨ 완전 일치",
                        "similar_5": "📌 유사 조합(5개 일치)",
                        "similar_4": "⚠️ 부분 일치(4개)",
                        "similar_3": "⚠️ 낮은 신뢰도(3개)"
                    }.get(result["type"], result["type"])
                    st.info(f"{label}: {result['count']}건 기반")
                else:
                    st.error("❌ 참고 데이터가 없습니다.")

# =========================
# 2) 데이터 입력
# =========================
elif menu == "📥 데이터 입력":
    st.title("📥 데이터 입력 (신규 템플릿)")

    tab1, tab2, tab3 = st.tabs(["📝 수동 입력", "📂 Excel 업로드", "🧵 소재 관리"])

    with tab1:
        st.subheader("판매 데이터 수동 입력")
        st.info("💡 신규 템플릿: 품번/컬러/가격/제조방식/소재명/핏/기장/당시즌판매수량/당시즌판매액")

        col1, col2 = st.columns(2)
        with col1:
            input_code = st.text_input("품번", placeholder="TXHD6054")
            if input_code:
                parsed = parse_item_code(input_code)
                if parsed:
                    gender_text = GENDER_MAPPING.get(parsed["gender"], "알수없음")
                    item_text = ITEM_MAPPING.get(parsed["item_code"], "알수없음")
                    st.success(f"✅ {gender_text} / {item_text}")
                else:
                    st.warning("⚠️ 품번 형식을 확인해주세요 (최소 8자리)")

            input_color = st.text_input("컬러", placeholder="BKS")
            input_price_unit = st.number_input("가격", min_value=0, step=1000, value=149000)
            input_manufacturing = st.text_input("제조방식", value="KNIT", help="예: KNIT / WOVEN / CUT&SEW")
            input_material = st.text_input("소재명", placeholder="JZR3055 595쮸리")

        with col2:
            input_fit = st.text_input("핏", value="SEMI-OVER")
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
        template.loc[0] = ["TXHD6054", "BKS", 149000, "KNIT", "JZR3055 595쮸리", "SEMI-OVER", "REGULAR", 15, 2235000]

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            template.to_excel(writer, index=False, sheet_name="판매데이터")

        st.download_button(
            "📥 판매 템플릿 다운로드",
            buffer.getvalue(),
            "판매데이터_템플릿_신규.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        uploaded = st.file_uploader("판매 Excel 파일 선택", type=["xlsx", "xls"])
        if uploaded:
            try:
                df_upload = pd.read_excel(uploaded)
                df_upload = make_json_safe_df(df_upload)
                st.dataframe(df_upload.head(10), use_container_width=True)

                missing = [c for c in SALES_COLS if c not in df_upload.columns]
                if missing:
                    st.error(f"❌ 업로드 파일 컬럼 누락: {missing}")
                else:
                    if st.button("✅ 판매 업로드 적용"):
                        if save_sales_data(df_upload):
                            st.session_state.sales_data = load_sales_data()
                            st.success(f"✅ {len(df_upload)}개 추가!")
                            st.rerun()
            except Exception as e:
                st.error(f"❌ 오류: {e}")

    with tab3:
        st.subheader("소재 마스터 관리 (신규 템플릿)")

        template_mat = pd.DataFrame(columns=MATERIAL_COLS)
        template_mat.loc[0] = ["BF-5933", "BF", "POLYESTER", 100, 30, "135X140", 275, 2, 1, 3]

        buffer2 = io.BytesIO()
        with pd.ExcelWriter(buffer2, engine="openpyxl") as writer:
            template_mat.to_excel(writer, index=False, sheet_name="소재데이터")

        st.download_button(
            "📥 소재 템플릿 다운로드",
            buffer2.getvalue(),
            "소재템플릿_신규.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        uploaded_mat = st.file_uploader("소재 Excel 파일 선택", type=["xlsx", "xls"])
        if uploaded_mat:
            try:
                df_mat = pd.read_excel(uploaded_mat)
                st.dataframe(df_mat.head(20), use_container_width=True)

                missing = [c for c in MATERIAL_COLS if c not in df_mat.columns]
                if missing:
                    st.error(f"❌ 업로드 파일 컬럼 누락: {missing}")
                else:
                    if st.button("✅ 소재 업로드 적용"):
                        if save_material_data(df_mat):
                            st.session_state.material_data = load_material_data()
                            st.success("✅ 소재 추가 완료!")
                            st.rerun()
            except Exception as e:
                st.error(f"❌ 오류: {e}")

# =========================
# 3) 대시보드 (당시즌 기준)
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

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("📁 카테고리별 당시즌판매수량")
            category_sales = df.groupby("카테고리")["당시즌판매수량"].sum().sort_values(ascending=False)
            fig3 = px.bar(x=category_sales.values, y=category_sales.index, orientation="h")
            fig3.update_layout(showlegend=False, xaxis_title="당시즌판매수량", yaxis_title="")
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.subheader("👕 아이템별 TOP 10 (당시즌판매수량)")
            item_sales = df.groupby("아이템명")["당시즌판매수량"].sum().sort_values(ascending=False).head(10)
            fig4 = px.bar(x=item_sales.values, y=item_sales.index, orientation="h")
            fig4.update_layout(showlegend=False, xaxis_title="당시즌판매수량", yaxis_title="")
            st.plotly_chart(fig4, use_container_width=True)

        st.divider()
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("🌈 컬러별 TOP 10 (당시즌판매수량)")
            color_sales = df.groupby("컬러")["당시즌판매수량"].sum().sort_values(ascending=False).head(10)
            fig5 = px.bar(x=color_sales.values, y=color_sales.index, orientation="h")
            fig5.update_layout(showlegend=False, xaxis_title="당시즌판매수량", yaxis_title="")
            st.plotly_chart(fig5, use_container_width=True)

        with col6:
            st.subheader("🌸 시즌별 당시즌판매수량")
            season_sales = df.groupby("시즌")["당시즌판매수량"].sum()
            fig6 = px.pie(values=season_sales.values, names=season_sales.index, hole=0.4)
            st.plotly_chart(fig6, use_container_width=True)

# =========================
# 4) 랭킹 (당시즌 기준)
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

        st.divider()
        st.subheader("🔥 조합 히트맵 (당시즌판매수량)")
        heatmap_x = st.selectbox("X축", ["아이템명", "제조방식", "소재명", "핏", "기장", "성별"])
        heatmap_y = st.selectbox("Y축", ["제조방식", "소재명", "핏", "기장", "아이템명", "성별"],
                                 index=1 if heatmap_x == "아이템명" else 0)
        if heatmap_x != heatmap_y:
            pivot = df.pivot_table(values="당시즌판매수량", index=heatmap_y, columns=heatmap_x, aggfunc="sum", fill_value=0)
            fig_heat = px.imshow(pivot, color_continuous_scale="RdYlGn", aspect="auto")
            fig_heat.update_layout(xaxis_title=heatmap_x, yaxis_title=heatmap_y)
            st.plotly_chart(fig_heat, use_container_width=True)

# =========================
# 5) 소재 분석 (당시즌 기준 + GU/RA/SA 상관/구간)
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
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🧵 소재별 총 당시즌판매수량 TOP 10")
            fig1 = px.bar(material_stats.head(10), x="총당시즌판매수량", y="소재명", orientation="h")
            fig1.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("💰 소재별 총 당시즌판매액 TOP 10")
            fig2 = px.bar(material_stats.head(10), x="총당시즌판매액", y="소재명", orientation="h")
            fig2.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("🔍 소재별 상세 분석")
        selected_material = st.selectbox("소재 선택", material_stats["소재명"].tolist())

        if selected_material:
            mdf = df[df["소재명"] == selected_material]

            c1, c2, c3 = st.columns(3)
            c1.metric("총 당시즌판매수량", f"{mdf['당시즌판매수량'].sum():,}개")
            c2.metric("평균 당시즌판매수량", f"{mdf['당시즌판매수량'].mean():.0f}개")
            c3.metric("사용 SKU", f"{mdf['품번'].nunique():,}개")

            st.markdown(f"#### {selected_material} 아이템별 성과(당시즌판매수량)")
            item_perf = mdf.groupby("아이템명")["당시즌판매수량"].sum().sort_values(ascending=False)
            fig3 = px.bar(x=item_perf.values, y=item_perf.index, orientation="h")
            fig3.update_layout(showlegend=False, xaxis_title="당시즌판매수량", yaxis_title="")
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown(f"#### {selected_material} 제조방식별 성과(당시즌판매수량)")
            manu_perf = mdf.groupby("제조방식")["당시즌판매수량"].sum().sort_values(ascending=False)
            fig4 = px.pie(values=manu_perf.values, names=manu_perf.index, hole=0.4)
            st.plotly_chart(fig4, use_container_width=True)

            if not st.session_state.material_data.empty:
                info = st.session_state.material_data[st.session_state.material_data["소재명"] == selected_material]
                if not info.empty:
                    st.markdown("#### 📋 소재 마스터(물성/광택/거칠기)")
                    st.dataframe(info[MATERIAL_COLS], use_container_width=True, hide_index=True)
                else:
                    st.info("소재 마스터에 해당 소재명이 없습니다(소재명 불일치/미등록).")

        st.divider()
        st.subheader("📈 GU/RA/SA ↔ 판매 상관/구간 분석 (소재 마스터 조인)")

        if st.session_state.material_data.empty:
            st.warning("소재 마스터(material_data)가 비어 있어 GU/RA/SA 분석을 진행할 수 없습니다.")
        else:
            sales_df = st.session_state.sales_data.copy()
            mat_df = st.session_state.material_data.copy()

            sales_df["당시즌판매수량"] = pd.to_numeric(sales_df["당시즌판매수량"], errors="coerce").fillna(0)
            sales_df["당시즌판매액"] = pd.to_numeric(sales_df["당시즌판매액"], errors="coerce").fillna(0)

            for c in ["GU", "RA", "SA"]:
                mat_df[c] = pd.to_numeric(mat_df[c], errors="coerce")

            mat_small = mat_df[["소재명", "GU", "RA", "SA"]].drop_duplicates(subset=["소재명"])
            merged = sales_df.merge(mat_small, on="소재명", how="left")

            missing_prop = merged["GU"].isna().mean() if len(merged) else 1.0
            st.caption(f"소재 마스터(GU/RA/SA) 미매칭 비율: **{missing_prop*100:.1f}%** (소재명 불일치/미등록 가능)")

            scope = st.radio("분석 범위", ["전체 소재", "선택한 소재만"], horizontal=True)
            if scope == "선택한 소재만":
                if "selected_material" in locals() and selected_material:
                    merged_scope = merged[merged["소재명"] == selected_material].copy()
                else:
                    merged_scope = merged.iloc[0:0].copy()
            else:
                merged_scope = merged.copy()

            if merged_scope.empty:
                st.warning("선택 범위에 분석할 데이터가 없습니다.")
            else:
                st.markdown("### 1) 상관분석 (Pearson)")
                target_metric = st.selectbox("판매 지표", ["당시즌판매수량", "당시즌판매액"], index=0)

                corr_rows = []
                for c in ["GU", "RA", "SA"]:
                    corr = _safe_corr(merged_scope[c], merged_scope[target_metric])
                    n = int(pd.concat([merged_scope[c], merged_scope[target_metric]], axis=1).dropna().shape[0])
                    corr_rows.append({"지표": c, "corr": corr, "샘플수": n})

                st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

                st.markdown("### 2) 구간(빈) 비교 분석")
                bin_metric = st.selectbox("구간화할 물성 지표", ["GU", "RA", "SA"], index=0)
                bin_method = st.radio("구간화 방식", ["equal_width", "quantile"], horizontal=True)
                bin_count = st.slider("구간 수", 2, 6, 4)

                tmp = merged_scope.dropna(subset=[bin_metric]).copy()
                if tmp.empty:
                    st.warning("구간 분석할 데이터가 없습니다. (해당 지표가 비어있음)")
                else:
                    bins = _bin_series(tmp[bin_metric], method=bin_method, bins=bin_count)
                    if bins is None:
                        st.warning("구간 생성 실패 (데이터 부족)")
                    else:
                        tmp["구간"] = bins.astype(str)
                        agg = tmp.groupby("구간").agg(
                            표본수=("품번", "count"),
                            판매수량_합=("당시즌판매수량", "sum"),
                            판매수량_평균=("당시즌판매수량", "mean"),
                            판매액_합=("당시즌판매액", "sum"),
                            판매액_평균=("당시즌판매액", "mean"),
                        ).reset_index()

                        for c in ["판매수량_평균", "판매액_평균"]:
                            agg[c] = agg[c].round(2)

                        st.dataframe(agg, use_container_width=True, hide_index=True)

                        colA, colB = st.columns(2)
                        with colA:
                            fig_q = px.bar(agg, x="구간", y="판매수량_평균",
                                           title=f"{bin_metric} 구간별 평균 당시즌판매수량")
                            st.plotly_chart(fig_q, use_container_width=True)
                        with colB:
                            fig_a = px.bar(agg, x="구간", y="판매액_평균",
                                           title=f"{bin_metric} 구간별 평균 당시즌판매액")
                            st.plotly_chart(fig_a, use_container_width=True)

                st.markdown("### 3) (옵션) 소재 단위 요약")
                if st.checkbox("소재명 단위로 요약 보기", value=False):
                    mat_level = merged_scope.groupby("소재명").agg(
                        GU=("GU", "mean"),
                        RA=("RA", "mean"),
                        SA=("SA", "mean"),
                        판매수량=("당시즌판매수량", "sum"),
                        판매액=("당시즌판매액", "sum"),
                        SKU수=("품번", "nunique"),
                    ).reset_index()
                    mat_level = mat_level.dropna(subset=["GU", "RA", "SA"], how="all")
                    st.dataframe(mat_level.sort_values("판매수량", ascending=False).head(30),
                                 use_container_width=True, hide_index=True)

# =========================
# 6) 데이터 관리
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
                csv1 = st.session_state.sales_data[SALES_COLS].to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "📥 판매 데이터 CSV 다운로드",
                    csv1,
                    f"판매데이터_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
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
        st.subheader("📊 데이터 확인 및 편집 (로컬)")
        st.caption("⚠️ data_editor에서 수정한 값은 DB 업데이트로 반영되지 않습니다(현재 버전은 insert/delete 중심).")
        data_type = st.radio("데이터 유형", ["판매 데이터", "소재 데이터"], horizontal=True)

        if data_type == "판매 데이터":
            if not st.session_state.sales_data.empty:
                edited_sales = st.data_editor(
                    st.session_state.sales_data[SALES_COLS],
                    use_container_width=True,
                    num_rows="dynamic"
                )
                if st.button("💾 (로컬) 판매 변경사항 반영"):
                    st.session_state.sales_data = edited_sales
                    st.success("✅ 로컬 변경사항 반영 완료 (DB 업데이트는 아님)")
                    st.rerun()
            else:
                st.warning("편집할 판매 데이터가 없습니다.")
        else:
            if not st.session_state.material_data.empty:
                edited_mat = st.data_editor(
                    st.session_state.material_data[MATERIAL_COLS],
                    use_container_width=True,
                    num_rows="dynamic"
                )
                if st.button("💾 (로컬) 소재 변경사항 반영"):
                    st.session_state.material_data = edited_mat
                    st.success("✅ 로컬 변경사항 반영 완료 (DB 업데이트는 아님)")
                    st.rerun()
            else:
                st.warning("편집할 소재 데이터가 없습니다.")

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

💡 **데이터 저장**
- Supabase 클라우드에 영구 저장
- 브라우저 종료해도 데이터 유지

🔄 **새로고침**
- 캐시 시간: 10분
""")
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 세르지오타키니 판매분석시스템 (Streamlit+Supabase)")
