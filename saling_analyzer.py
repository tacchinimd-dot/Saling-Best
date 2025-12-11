import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import numpy as np
import re

# ========================================
# 페이지 설정
# ========================================
st.set_page_config(
    page_title="세르지오타키니 판매 분석",
    page_icon="👔",
    layout="wide"
)

# ========================================
# 주차 관련 함수
# ========================================
def validate_week_format(week_str):
    """주차 형식 검증 (예: 24'W1 ~ 25'W52)"""
    pattern = r"^\d{2}'W\d{1,2}$"
    if re.match(pattern, week_str):
        parts = week_str.split("'W")
        year_code = int(parts[0])
        week_num = int(parts[1])
        
        # 24'W1 ~ 25'W52 범위 검증
        if year_code == 24 and 1 <= week_num <= 52:
            return True
        elif year_code == 25 and 1 <= week_num <= 52:
            return True
        else:
            return False
    return False

def parse_week(week_str):
    """주차 문자열을 연도와 주차로 분리"""
    if validate_week_format(week_str):
        parts = week_str.split("'W")
        year = int("20" + parts[0])
        week = int(parts[1])
        return year, week
    return None, None

def get_week_options():
    """주차 선택 옵션 생성 (24'W1 ~ 25'W52)"""
    weeks = []
    # 2024년 1주차 ~ 52주차
    for week in range(1, 53):
        weeks.append(f"24'W{week}")
    # 2025년 1주차 ~ 52주차
    for week in range(1, 53):
        weeks.append(f"25'W{week}")
    return weeks

# ========================================
# 품번 파싱 함수
# ========================================
def parse_item_code(code):
    """품번을 파싱하여 정보 추출"""
    if not code or len(code) < 8:
        return None
    
    try:
        brand = code[0]
        gender = code[1]
        item_code = code[2:4]
        sequence = code[4:7]
        year = code[7]
        season = code[8] if len(code) > 8 else None
        
        return {
            'brand': brand,
            'gender': gender,
            'item_code': item_code,
            'sequence': sequence,
            'year': year,
            'season': season
        }
    except:
        return None

# ========================================
# 아이템 코드 매핑
# ========================================
ITEM_MAPPING = {
    'DJ': '다운점퍼', 'DV': '다운베스트', 'JK': '자켓', 'JP': '점퍼',
    'KC': '니트가디건', 'PD': '패딩', 'VT': '베스트', 'WJ': '윈드브레이커', 'WT': '우븐티셔츠',
    'HD': '후드티', 'KP': '스웨터풀오버', 'KV': '스웨터베스트', 'KU': '반팔스웨터',
    'MT': '맨투맨', 'OP': '원피스', 'PQ': '폴로티셔츠', 'RL': '긴팔티셔츠',
    'RS': '반팔티셔츠', 'TR': '트레이닝상의', 'WS': '우븐셔츠',
    'LG': '레깅스', 'PT': '팬츠', 'SK': '스커트', 'SP': '반바지',
    'SR': '여성하의스코트', 'TB': '트레이닝숏팬츠', 'TP': '트레이닝하의',
    'BR': '브라', 'SL': '슬리브리스'
}

CATEGORY_MAPPING = {
    'DJ': '아우터', 'DV': '아우터', 'JK': '아우터', 'JP': '아우터', 'KC': '아우터',
    'PD': '아우터', 'VT': '아우터', 'WJ': '아우터', 'WT': '아우터',
    'HD': '이너', 'KP': '이너', 'KV': '이너', 'KU': '이너', 'MT': '이너',
    'OP': '이너', 'PQ': '이너', 'RL': '이너', 'RS': '이너', 'TR': '이너', 'WS': '이너',
    'LG': '하의', 'PT': '하의', 'SK': '하의', 'SP': '하의', 'SR': '하의',
    'TB': '하의', 'TP': '하의',
    'BR': '기타', 'SL': '기타'
}

GENDER_MAPPING = {'M': '남성', 'W': '여성', 'U': '공용'}
SEASON_MAPPING = {'1': '봄', '3': '여름', '4': '가을', '6': '겨울'}
YEAR_MAPPING = {'3': '2023', '4': '2024', '5': '2025', '6': '2026'}

FIT_OPTIONS = ['slim', 'regular', 'semi-over', 'over']
LENGTH_OPTIONS = ['Crop', 'Mid', 'Long', 'Regular', 'Semi-Crop', 'Short']

# ========================================
# Session State 초기화
# ========================================
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = pd.DataFrame(columns=[
        '주차', '품번', '컬러', '소재명', '핏', '기장', '판매수량', '판매금액'
    ])

if 'material_data' not in st.session_state:
    st.session_state.material_data = pd.DataFrame(columns=[
        '소재명', '소재업체', '혼용율', '중량', '두께', '밀도'
    ])

# ========================================
# 데이터 처리 함수
# ========================================
def enrich_sales_data(df):
    """판매 데이터에 품번 파싱 정보 및 주차 정보 추가"""
    enriched = df.copy()
    
    parsed_data = []
    week_data = []
    
    for idx, row in enriched.iterrows():
        code = row['품번']
        parsed = parse_item_code(code)
        if parsed:
            parsed_data.append({
                '성별': GENDER_MAPPING.get(parsed['gender'], '알수없음'),
                '아이템코드': parsed['item_code'],
                '아이템명': ITEM_MAPPING.get(parsed['item_code'], '알수없음'),
                '카테고리': CATEGORY_MAPPING.get(parsed['item_code'], '기타'),
                '연도': YEAR_MAPPING.get(parsed['year'], '알수없음'),
                '시즌': SEASON_MAPPING.get(parsed['season'], '알수없음')
            })
        else:
            parsed_data.append({
                '성별': '알수없음', '아이템코드': '', '아이템명': '알수없음',
                '카테고리': '기타', '연도': '알수없음', '시즌': '알수없음'
            })
        
        week_str = row['주차']
        year, week = parse_week(week_str)
        week_data.append({
            '주차_연도': year if year else 0,
            '주차_번호': week if week else 0
        })
    
    parsed_df = pd.DataFrame(parsed_data)
    week_df = pd.DataFrame(week_data)
    result = pd.concat([enriched.reset_index(drop=True), parsed_df, week_df], axis=1)
    return result

def predict_combination(gender, item_name, material, fit, length):
    """조합 예측 함수 (통계 기반)"""
    if st.session_state.sales_data.empty:
        return None
    
    df = enrich_sales_data(st.session_state.sales_data)
    
    exact_match = df[
        (df['성별'] == gender) &
        (df['아이템명'] == item_name) &
        (df['소재명'] == material) &
        (df['핏'] == fit) &
        (df['기장'] == length)
    ]
    
    if not exact_match.empty:
        return {
            'type': 'exact',
            'avg_quantity': exact_match['판매수량'].mean(),
            'avg_price': exact_match['판매금액'].mean(),
            'count': len(exact_match),
            'confidence': 95
        }
    
    similar = df[
        (df['성별'] == gender) &
        (df['아이템명'] == item_name) &
        (df['소재명'] == material) &
        (df['핏'] == fit)
    ]
    
    if not similar.empty:
        return {
            'type': 'similar_4',
            'avg_quantity': similar['판매수량'].mean(),
            'avg_price': similar['판매금액'].mean(),
            'count': len(similar),
            'confidence': 75
        }
    
    similar = df[
        (df['성별'] == gender) &
        (df['아이템명'] == item_name) &
        (df['소재명'] == material)
    ]
    
    if not similar.empty:
        return {
            'type': 'similar_3',
            'avg_quantity': similar['판매수량'].mean(),
            'avg_price': similar['판매금액'].mean(),
            'count': len(similar),
            'confidence': 60
        }
    
    similar = df[
        (df['성별'] == gender) &
        (df['아이템명'] == item_name)
    ]
    
    if not similar.empty:
        return {
            'type': 'similar_2',
            'avg_quantity': similar['판매수량'].mean(),
            'avg_price': similar['판매금액'].mean(),
            'count': len(similar),
            'confidence': 40
        }
    
    return None

# ========================================
# 사이드바
# ========================================
st.sidebar.title("👔 세르지오타키니")
st.sidebar.markdown("### 판매 분석 시스템")

menu = st.sidebar.radio(
    "메뉴",
    ["🎯 조합 예측 시뮬레이터", "📥 데이터 입력", "📊 대시보드", "🏆 조합 성과 랭킹", "🧵 소재 분석", "💾 데이터 관리"]
)

# ========================================
# 1. 조합 예측 시뮬레이터
# ========================================
if menu == "🎯 조합 예측 시뮬레이터":
    st.title("🎯 조합 예측 시뮬레이터")
    st.markdown("#### 특정 조합의 판매량을 예측합니다")
    
    if st.session_state.sales_data.empty:
        st.warning("⚠️ 예측을 위한 판매 데이터가 없습니다. '데이터 입력' 메뉴에서 데이터를 먼저 입력해주세요.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 조합 입력")
            
            gender = st.selectbox("성별", ["남성", "여성", "공용"])
            item_list = sorted(set(ITEM_MAPPING.values()))
            item_name = st.selectbox("아이템", item_list)
            
            df_enriched = enrich_sales_data(st.session_state.sales_data)
            materials = sorted(df_enriched['소재명'].unique().tolist())
            if materials:
                material = st.selectbox("소재", materials)
            else:
                material = st.text_input("소재명 입력")
            
            fit = st.selectbox("핏", FIT_OPTIONS)
            length = st.selectbox("기장", LENGTH_OPTIONS)
            
            predict_btn = st.button("🔮 예측 결과 보기", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("📊 예측 결과")
            
            if predict_btn:
                result = predict_combination(gender, item_name, material, fit, length)
                
                if result:
                    st.success("✅ 예측이 완료되었습니다!")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("예상 판매량", f"{result['avg_quantity']:.0f}개")
                    with metric_col2:
                        st.metric("예상 판매금액", f"{result['avg_price']:,.0f}원")
                    with metric_col3:
                        st.metric("신뢰도", f"{result['confidence']}%")
                    
                    st.divider()
                    
                    if result['type'] == 'exact':
                        st.info(f"✨ **완전 일치**: 동일한 조합의 과거 데이터 {result['count']}건을 기반으로 예측했습니다.")
                    elif result['type'] == 'similar_4':
                        st.info(f"📌 **유사 조합**: 성별, 아이템, 소재, 핏이 일치하는 {result['count']}건을 기반으로 예측했습니다.")
                    elif result['type'] == 'similar_3':
                        st.warning(f"⚠️ **부분 일치**: 성별, 아이템, 소재가 일치하는 {result['count']}건을 기반으로 예측했습니다.")
                    else:
                        st.warning(f"⚠️ **낮은 신뢰도**: 성별과 아이템만 일치하는 {result['count']}건을 기반으로 예측했습니다.")
                    
                    st.divider()
                    st.markdown("#### 🔍 유사 조합 참고")
                    
                    similar_combos = df_enriched[
                        (df_enriched['성별'] == gender) &
                        (df_enriched['아이템명'] == item_name)
                    ].groupby(['소재명', '핏', '기장']).agg({
                        '판매수량': 'mean',
                        '판매금액': 'mean'
                    }).round(0).reset_index()
                    
                    if not similar_combos.empty:
                        similar_combos = similar_combos.sort_values('판매수량', ascending=False).head(5)
                        st.dataframe(similar_combos, use_container_width=True, hide_index=True)
                    
                else:
                    st.error("❌ 해당 조합에 대한 참고 데이터가 없습니다.")

# ========================================
# 2. 데이터 입력
# ========================================
elif menu == "📥 데이터 입력":
    st.title("📥 데이터 입력 및 업로드")
    
    tab1, tab2, tab3 = st.tabs(["📝 판매 데이터 입력", "📂 판매 데이터 업로드", "🧵 소재 데이터 관리"])
    
    with tab1:
        st.subheader("판매 데이터 수동 입력")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 주차 형식: `24'W1` (2024년 1주차) ~ `25'W52` (2025년 52주차)")
            
            # 드롭다운으로 주차 선택
            week_options = get_week_options()
            input_week = st.selectbox(
                "주차 선택",
                options=week_options,
                index=len([w for w in week_options if w.startswith("24'")]),  # 25'W1을 기본값으로
                help="24'W1 (2024년 1주차) ~ 25'W52 (2025년 52주차)"
            )
            
            # 선택된 주차 정보 표시
            if input_week:
                year, week = parse_week(input_week)
                st.success(f"✅ {year}년 {week}주차 선택됨")
            
            input_code = st.text_input("품번", placeholder="예: TWSK10954")
            
            if input_code:
                parsed = parse_item_code(input_code)
                if parsed:
                    st.success(f"✅ {GENDER_MAPPING.get(parsed['gender'])} / {ITEM_MAPPING.get(parsed['item_code'])} / {YEAR_MAPPING.get(parsed['year'])} / {SEASON_MAPPING.get(parsed['season'])}")
                else:
                    st.warning("⚠️ 올바른 품번 형식이 아닙니다.")
            
            input_color = st.text_input("컬러", placeholder="예: 블랙")
            input_material = st.text_input("소재명", placeholder="예: 면100%")
        
        with col2:
            input_fit = st.selectbox("핏", FIT_OPTIONS, key="input_fit")
            input_length = st.selectbox("기장", LENGTH_OPTIONS, key="input_length")
            input_quantity = st.number_input("판매 수량", min_value=0, step=1)
            input_price = st.number_input("판매 금액 (원)", min_value=0, step=1000)
        
        if st.button("➕ 판매 데이터 추가", type="primary"):
            if input_week and validate_week_format(input_week) and input_code and input_color and input_material:
                new_row = pd.DataFrame([{
                    '주차': input_week,
                    '품번': input_code,
                    '컬러': input_color,
                    '소재명': input_material,
                    '핏': input_fit,
                    '기장': input_length,
                    '판매수량': input_quantity,
                    '판매금액': input_price
                }])
                st.session_state.sales_data = pd.concat([st.session_state.sales_data, new_row], ignore_index=True)
                st.success("✅ 판매 데이터가 추가되었습니다!")
                st.rerun()
            else:
                st.error("❌ 주차, 품번, 컬러, 소재명은 필수 입력 항목입니다.")
    
    with tab2:
        st.subheader("판매 데이터 Excel 업로드")
        
        template_sales = pd.DataFrame(columns=['주차', '품번', '컬러', '소재명', '핏', '기장', '판매수량', '판매금액'])
        template_sales.loc[0] = ["25'W1", 'TWRS10954', '블랙', '면100%', 'slim', 'crop', 120, 1200000]
        template_sales.loc[1] = ["25'W2", 'TMPO10953', '네이비', '폴리80%', 'regular', 'mid', 85, 850000]
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            template_sales.to_excel(writer, index=False, sheet_name='판매데이터')
        
        st.download_button(
            label="📥 판매 데이터 템플릿 다운로드",
            data=buffer.getvalue(),
            file_name="판매데이터_템플릿.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.info("📋 필수 열: 주차(형식: 25'W1), 품번, 컬러, 소재명, 핏, 기장, 판매수량, 판매금액")
        
        uploaded_sales = st.file_uploader("Excel 파일 선택", type=['xlsx', 'xls'], key="sales_upload")
        
        if uploaded_sales:
            try:
                df_upload = pd.read_excel(uploaded_sales)
                
                invalid_weeks = []
                for idx, week in enumerate(df_upload['주차']):
                    if not validate_week_format(str(week)):
                        invalid_weeks.append(f"행 {idx+2}: {week}")
                
                if invalid_weeks:
                    st.error(f"❌ 올바르지 않은 주차 형식:\n" + "\n".join(invalid_weeks[:5]))
                else:
                    st.write("📊 업로드된 데이터 미리보기:")
                    st.dataframe(df_upload.head(10), use_container_width=True)
                    
                    if st.button("✅ 판매 데이터 적용", type="primary"):
                        st.session_state.sales_data = pd.concat([st.session_state.sales_data, df_upload], ignore_index=True)
                        st.success(f"✅ {len(df_upload)}개의 판매 데이터가 추가되었습니다!")
                        st.rerun()
            except Exception as e:
                st.error(f"❌ 파일 읽기 오류: {e}")
    
    with tab3:
        st.subheader("소재 마스터 데이터 관리")
        
        template_material = pd.DataFrame(columns=['소재명', '소재업체', '혼용율', '중량', '두께', '밀도'])
        template_material.loc[0] = ['면100%', '태광섬유', '면100%', 180, 0.6, '고밀도']
        
        buffer2 = io.BytesIO()
        with pd.ExcelWriter(buffer2, engine='openpyxl') as writer:
            template_material.to_excel(writer, index=False, sheet_name='소재데이터')
        
        st.download_button(
            label="📥 소재 데이터 템플릿 다운로드",
            data=buffer2.getvalue(),
            file_name="소재데이터_템플릿.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        uploaded_material = st.file_uploader("Excel 파일 선택", type=['xlsx', 'xls'], key="material_upload")
        
        if uploaded_material:
            try:
                df_material = pd.read_excel(uploaded_material)
                st.write("📊 업로드된 소재 데이터:")
                st.dataframe(df_material, use_container_width=True)
                
                if st.button("✅ 소재 데이터 적용", type="primary"):
                    st.session_state.material_data = pd.concat([st.session_state.material_data, df_material], ignore_index=True)
                    st.success(f"✅ {len(df_material)}개의 소재 데이터가 추가되었습니다!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ 파일 읽기 오류: {e}")
        
        if not st.session_state.material_data.empty:
            st.divider()
            st.markdown("#### 📋 현재 소재 마스터 데이터")
            st.dataframe(st.session_state.material_data, use_container_width=True, hide_index=True)

# ========================================
# 3. 대시보드
# ========================================
elif menu == "📊 대시보드":
    st.title("📊 판매 분석 대시보드")
    
    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 판매수량", f"{df['판매수량'].sum():,}개")
        with col2:
            st.metric("총 판매금액", f"{df['판매금액'].sum():,}원")
        with col3:
            st.metric("평균 판매가", f"{df['판매금액'].mean():,.0f}원")
        with col4:
            st.metric("총 SKU", f"{len(df)}개")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 성별 판매 분포")
            gender_sales = df.groupby('성별')['판매수량'].sum().sort_values(ascending=False)
            fig1 = px.pie(values=gender_sales.values, names=gender_sales.index, hole=0.4)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("📁 카테고리별 판매")
            category_sales = df.groupby('카테고리')['판매수량'].sum().sort_values(ascending=False)
            fig2 = px.bar(x=category_sales.values, y=category_sales.index, orientation='h',
                         color=category_sales.values, color_continuous_scale='Blues')
            fig2.update_layout(showlegend=False, xaxis_title="판매수량", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("👕 아이템별 판매 TOP 10")
            item_sales = df.groupby('아이템명')['판매수량'].sum().sort_values(ascending=False).head(10)
            fig3 = px.bar(x=item_sales.values, y=item_sales.index, orientation='h',
                         color=item_sales.values, color_continuous_scale='Greens')
            fig3.update_layout(showlegend=False, xaxis_title="판매수량", yaxis_title="")
            st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            st.subheader("🌈 컬러별 판매")
            color_sales = df.groupby('컬러')['판매수량'].sum().sort_values(ascending=False).head(10)
            fig4 = px.bar(x=color_sales.values, y=color_sales.index, orientation='h',
                         color=color_sales.values, color_continuous_scale='Rainbow')
            fig4.update_layout(showlegend=False, xaxis_title="판매수량", yaxis_title="")
            st.plotly_chart(fig4, use_container_width=True)
        
        st.divider()
        st.subheader("📅 주차별 판매 추이")
        
        df_sorted = df.sort_values(['주차_연도', '주차_번호'])
        weekly_sales = df_sorted.groupby('주차').agg({
            '판매수량': 'sum',
            '판매금액': 'sum'
        }).reset_index()
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=weekly_sales['주차'], y=weekly_sales['판매수량'],
                                 mode='lines+markers', name='판매수량'))
        fig5.update_layout(xaxis_title="주차", yaxis_title="판매수량", hovermode='x unified')
        st.plotly_chart(fig5, use_container_width=True)
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("🌸 시즌별 판매")
            season_sales = df.groupby('시즌')['판매수량'].sum()
            fig6 = px.pie(values=season_sales.values, names=season_sales.index, hole=0.4)
            st.plotly_chart(fig6, use_container_width=True)
        
        with col6:
            st.subheader("📅 연도별 판매")
            year_sales = df.groupby('연도')['판매수량'].sum().sort_index()
            fig7 = px.line(x=year_sales.index, y=year_sales.values, markers=True)
            fig7.update_layout(xaxis_title="연도", yaxis_title="판매수량")
            st.plotly_chart(fig7, use_container_width=True)

# ========================================
# 4. 조합 성과 랭킹
# ========================================
elif menu == "🏆 조합 성과 랭킹":
    st.title("🏆 조합별 성과 랭킹")
    
    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data)
        
        df['조합'] = df['성별'] + ' / ' + df['아이템명'] + ' / ' + df['소재명'] + ' / ' + df['핏'] + ' / ' + df['기장']
        
        combo_stats = df.groupby('조합').agg({
            '판매수량': ['sum', 'mean', 'count'],
            '판매금액': ['sum', 'mean']
        }).round(0)
        
        combo_stats.columns = ['총판매수량', '평균판매수량', '판매횟수', '총판매금액', '평균판매금액']
        combo_stats = combo_stats.reset_index()
        
        metric = st.radio("분석 기준", ["총판매수량", "평균판매수량", "총판매금액", "평균판매금액"], horizontal=True)
        top_n = st.slider("표시할 조합 수", 5, 20, 10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🥇 Best {top_n} 조합")
            top_combos = combo_stats.nlargest(top_n, metric)
            
            fig_top = px.bar(top_combos, x=metric, y='조합', orientation='h',
                            color=metric, color_continuous_scale='Greens')
            fig_top.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)
            
            st.dataframe(top_combos, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader(f"🥉 Worst {top_n} 조합")
            bottom_combos = combo_stats.nsmallest(top_n, metric)
            
            fig_bottom = px.bar(bottom_combos, x=metric, y='조합', orientation='h',
                               color=metric, color_continuous_scale='Reds')
            fig_bottom.update_layout(showlegend=False, yaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_bottom, use_container_width=True)
            
            st.dataframe(bottom_combos, use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("🔥 조합 히트맵 분석")
        
        heatmap_x = st.selectbox("X축", ["아이템명", "소재명", "핏", "기장", "성별"])
        heatmap_y = st.selectbox("Y축", ["소재명", "핏", "기장", "아이템명", "성별"],
                                index=1 if heatmap_x == "아이템명" else 0)
        
        if heatmap_x != heatmap_y:
            pivot = df.pivot_table(values='판매수량', index=heatmap_y, columns=heatmap_x,
                                  aggfunc='sum', fill_value=0)
            fig_heat = px.imshow(pivot, color_continuous_scale='RdYlGn', aspect='auto')
            fig_heat.update_layout(xaxis_title=heatmap_x, yaxis_title=heatmap_y)
            st.plotly_chart(fig_heat, use_container_width=True)

# ========================================
# 5. 소재 분석
# ========================================
elif menu == "🧵 소재 분석":
    st.title("🧵 소재별 성과 분석")
    
    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data)
        
        material_stats = df.groupby('소재명').agg({
            '판매수량': ['sum', 'mean', 'count'],
            '판매금액': ['sum', 'mean'],
            '품번': 'nunique'
        }).round(0)
        
        material_stats.columns = ['총판매수량', '평균판매수량', '판매횟수', '총판매금액', '평균판매금액', 'SKU수']
        material_stats = material_stats.reset_index().sort_values('총판매수량', ascending=False)
        
        st.subheader("📊 소재별 성과 요약")
        st.dataframe(material_stats, use_container_width=True, hide_index=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧵 소재별 총 판매수량")
            fig1 = px.bar(material_stats.head(10), x='총판매수량', y='소재명', orientation='h',
                         color='총판매수량', color_continuous_scale='Blues')
            fig1.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("💰 소재별 총 판매금액")
            fig2 = px.bar(material_stats.head(10), x='총판매금액', y='소재명', orientation='h',
                         color='총판매금액', color_continuous_scale='Greens')
            fig2.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()
        st.subheader("🔍 소재별 상세 분석")
        
        selected_material = st.selectbox("소재 선택", material_stats['소재명'].tolist())
        
        if selected_material:
            material_df = df[df['소재명'] == selected_material]
            
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("총 판매수량", f"{material_df['판매수량'].sum():,}개")
            with col4:
                st.metric("평균 판매수량", f"{material_df['판매수량'].mean():.0f}개")
            with col5:
                st.metric("사용 SKU", f"{material_df['품번'].nunique()}개")
            
            st.markdown(f"#### {selected_material} 사용 아이템별 성과")
            item_perf = material_df.groupby('아이템명')['판매수량'].sum().sort_values(ascending=False)
            fig3 = px.bar(x=item_perf.values, y=item_perf.index, orientation='h',
                         color=item_perf.values, color_continuous_scale='Oranges')
            fig3.update_layout(showlegend=False, xaxis_title="판매수량", yaxis_title="")
            st.plotly_chart(fig3, use_container_width=True)
            
            if not st.session_state.material_data.empty:
                material_info = st.session_state.material_data[
                    st.session_state.material_data['소재명'] == selected_material
                ]
                if not material_info.empty:
                    st.markdown("#### 📋 소재 상세 정보")
                    st.dataframe(material_info, use_container_width=True, hide_index=True)

# ========================================
# 6. 데이터 관리
# ========================================
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
                with pd.ExcelWriter(buffer1, engine='openpyxl') as writer:
                    st.session_state.sales_data.to_excel(writer, index=False, sheet_name='판매데이터')
                
                st.download_button(
                    label="📥 판매 데이터 Excel 다운로드",
                    data=buffer1.getvalue(),
                    file_name=f"판매데이터_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                csv1 = st.session_state.sales_data.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 판매 데이터 CSV 다운로드",
                    data=csv1,
                    file_name=f"판매데이터_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("다운로드할 판매 데이터가 없습니다.")
        
        with col2:
            st.markdown("#### 소재 데이터")
            if not st.session_state.material_data.empty:
                buffer2 = io.BytesIO()
                with pd.ExcelWriter(buffer2, engine='openpyxl') as writer:
                    st.session_state.material_data.to_excel(writer, index=False, sheet_name='소재데이터')
                
                st.download_button(
                    label="📥 소재 데이터 Excel 다운로드",
                    data=buffer2.getvalue(),
                    file_name=f"소재데이터_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("다운로드할 소재 데이터가 없습니다.")
    
    with tab2:
        st.subheader("📊 데이터 확인 및 편집")
        
        data_type = st.radio("데이터 유형", ["판매 데이터", "소재 데이터"], horizontal=True)
        
        if data_type == "판매 데이터":
            if not st.session_state.sales_data.empty:
                st.info(f"총 {len(st.session_state.sales_data)}개의 판매 데이터")
                
                edited_sales = st.data_editor(
                    st.session_state.sales_data,
                    use_container_width=True,
                    num_rows="dynamic"
                )
                
                if st.button("💾 판매 데이터 변경사항 저장"):
                    st.session_state.sales_data = edited_sales
                    st.success("✅ 변경사항이 저장되었습니다!")
                    st.rerun()
            else:
                st.warning("편집할 판매 데이터가 없습니다.")
        
        else:
            if not st.session_state.material_data.empty:
                st.info(f"총 {len(st.session_state.material_data)}개의 소재 데이터")
                
                edited_material = st.data_editor(
                    st.session_state.material_data,
                    use_container_width=True,
                    num_rows="dynamic"
                )
                
                if st.button("💾 소재 데이터 변경사항 저장"):
                    st.session_state.material_data = edited_material
                    st.success("✅ 변경사항이 저장되었습니다!")
                    st.rerun()
            else:
                st.warning("편집할 소재 데이터가 없습니다.")
    
    with tab3:
        st.subheader("🗑️ 데이터 삭제")
        st.warning("⚠️ **주의**: 삭제된 데이터는 복구할 수 없습니다! 먼저 백업을 다운로드하세요.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ 판매 데이터 전체 삭제", type="secondary"):
                st.session_state.sales_data = pd.DataFrame(columns=[
                    '주차', '품번', '컬러', '소재명', '핏', '기장', '판매수량', '판매금액'
                ])
                st.success("✅ 판매 데이터가 삭제되었습니다.")
                st.rerun()
        
        with col2:
            if st.button("🗑️ 소재 데이터 전체 삭제", type="secondary"):
                st.session_state.material_data = pd.DataFrame(columns=[
                    '소재명', '소재업체', '혼용율', '중량', '두께', '밀도'
                ])
                st.success("✅ 소재 데이터가 삭제되었습니다.")
                st.rerun()

# ========================================
# 푸터
# ========================================
st.sidebar.divider()
st.sidebar.info(f"""
📊 **현재 데이터 현황**
- 판매 데이터: {len(st.session_state.sales_data)}건
- 소재 데이터: {len(st.session_state.material_data)}건

💡 **주차 형식 안내**
- 24'W1 = 2024년 1주차
- 24'W52 = 2024년 52주차
- 25'W1 = 2025년 1주차
- 25'W52 = 2025년 52주차

⚠️ **중요 안내**
- 입력 가능 범위: 24'W1 ~ 25'W52
- 매일 Excel로 백업하세요
- 브라우저 종료 시 데이터 소멸
""")

st.sidebar.markdown("---")
st.sidebar.caption("© 2024 세르지오타키니 판매분석시스템 v1.1")
