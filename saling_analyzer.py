pythonimport streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="세르지오타키니 판매 분석",
    page_icon="👔",
    layout="wide"
)

# 품번 파싱 함수
def parse_item_code(code):
    if not code or len(code) < 8:
        return None
    try:
        return {
            'brand': code[0],
            'gender': code[1],
            'item_code': code[2:4],
            'sequence': code[4:7],
            'year': code[7],
            'season': code[8] if len(code) > 8 else None
        }
    except:
        return None

# 아이템 코드 매핑
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
MANUFACTURING_OPTIONS = ['컷앤소', '우븐', '스웨터']

# Session State 초기화
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = pd.DataFrame(columns=[
        '품번', '컬러', '제조방식', '소재명', '핏', '기장', '누적판매수량', '누적판매금액'
    ])

if 'material_data' not in st.session_state:
    st.session_state.material_data = pd.DataFrame(columns=[
        '소재명', '소재업체', '혼용율', '중량', '두께', '밀도'
    ])

# 데이터 처리 함수
def enrich_sales_data(df):
    enriched = df.copy()
    parsed_data = []
    
    for idx, row in enriched.iterrows():
        parsed = parse_item_code(row['품번'])
        if parsed:
            parsed_data.append({
                '성별': GENDER_MAPPING.get(parsed['gender'], '알수없음'),
                '아이템명': ITEM_MAPPING.get(parsed['item_code'], '알수없음'),
                '카테고리': CATEGORY_MAPPING.get(parsed['item_code'], '기타'),
                '연도': YEAR_MAPPING.get(parsed['year'], '알수없음'),
                '시즌': SEASON_MAPPING.get(parsed['season'], '알수없음')
            })
        else:
            parsed_data.append({
                '성별': '알수없음', '아이템명': '알수없음',
                '카테고리': '기타', '연도': '알수없음', '시즌': '알수없음'
            })
    
    return pd.concat([enriched.reset_index(drop=True), pd.DataFrame(parsed_data)], axis=1)

def predict_combination(gender, item_name, manufacturing, material, fit, length):
    if st.session_state.sales_data.empty:
        return None
    
    df = enrich_sales_data(st.session_state.sales_data)
    
    # 완전 일치
    exact = df[
        (df['성별'] == gender) &
        (df['아이템명'] == item_name) &
        (df['제조방식'] == manufacturing) &
        (df['소재명'] == material) &
        (df['핏'] == fit) &
        (df['기장'] == length)
    ]
    
    if not exact.empty:
        return {
            'type': 'exact',
            'avg_quantity': exact['누적판매수량'].mean(),
            'avg_price': exact['누적판매금액'].mean(),
            'count': len(exact),
            'confidence': 95
        }
    
    # 5개 일치
    similar = df[
        (df['성별'] == gender) &
        (df['아이템명'] == item_name) &
        (df['제조방식'] == manufacturing) &
        (df['소재명'] == material) &
        (df['핏'] == fit)
    ]
    
    if not similar.empty:
        return {'type': 'similar_5', 'avg_quantity': similar['누적판매수량'].mean(),
                'avg_price': similar['누적판매금액'].mean(), 'count': len(similar), 'confidence': 80}
    
    # 4개 일치
    similar = df[
        (df['성별'] == gender) &
        (df['아이템명'] == item_name) &
        (df['제조방식'] == manufacturing) &
        (df['소재명'] == material)
    ]
    
    if not similar.empty:
        return {'type': 'similar_4', 'avg_quantity': similar['누적판매수량'].mean(),
                'avg_price': similar['누적판매금액'].mean(), 'count': len(similar), 'confidence': 65}
    
    # 3개 일치
    similar = df[
        (df['성별'] == gender) &
        (df['아이템명'] == item_name) &
        (df['제조방식'] == manufacturing)
    ]
    
    if not similar.empty:
        return {'type': 'similar_3', 'avg_quantity': similar['누적판매수량'].mean(),
                'avg_price': similar['누적판매금액'].mean(), 'count': len(similar), 'confidence': 45}
    
    return None

# 사이드바
st.sidebar.title("👔 세르지오타키니")
st.sidebar.markdown("### 판매 분석 시스템")

menu = st.sidebar.radio(
    "메뉴",
    ["🎯 조합 예측", "📥 데이터 입력", "📊 대시보드", "🏆 랭킹", "🧵 소재 분석", "💾 데이터 관리"]
)

# 1. 조합 예측 시뮬레이터
if menu == "🎯 조합 예측":
    st.title("🎯 조합 예측 시뮬레이터")
    
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
            materials = sorted(df_enriched['소재명'].unique().tolist())
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
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("예상 누적판매량", f"{result['avg_quantity']:.0f}개")
                    with col_b:
                        st.metric("예상 누적판매금액", f"{result['avg_price']:,.0f}원")
                    with col_c:
                        st.metric("신뢰도", f"{result['confidence']}%")
                    
                    st.divider()
                    
                    if result['type'] == 'exact':
                        st.info(f"✨ 완전 일치: {result['count']}건 기반")
                    elif result['type'] == 'similar_5':
                        st.info(f"📌 유사 조합(5개 일치): {result['count']}건 기반")
                    elif result['type'] == 'similar_4':
                        st.warning(f"⚠️ 부분 일치(4개): {result['count']}건 기반")
                    else:
                        st.warning(f"⚠️ 낮은 신뢰도(3개): {result['count']}건 기반")
                else:
                    st.error("❌ 참고 데이터가 없습니다.")

# 2. 데이터 입력
elif menu == "📥 데이터 입력":
    st.title("📥 데이터 입력")
    
    tab1, tab2, tab3 = st.tabs(["📝 수동 입력", "📂 Excel 업로드", "🧵 소재 관리"])
    
    with tab1:
        st.subheader("판매 데이터 수동 입력")
        st.info("💡 누적 판매 데이터 입력 (현재까지 총 판매량)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_code = st.text_input("품번", placeholder="TWSK10954")
            
            if input_code:
                parsed = parse_item_code(input_code)
                if parsed:
                    st.success(f"✅ {GENDER_MAPPING.get(parsed['gender'])} / {ITEM_MAPPING.get(parsed['item_code'])}")
            
            input_color = st.text_input("컬러", placeholder="블랙")
            input_manufacturing = st.selectbox("제조방식", MANUFACTURING_OPTIONS)
            input_material = st.text_input("소재명", placeholder="면100%")
        
        with col2:
            input_fit = st.selectbox("핏", FIT_OPTIONS)
            input_length = st.selectbox("기장", LENGTH_OPTIONS)
            input_quantity = st.number_input("누적 판매수량", min_value=0, step=1)
            input_price = st.number_input("누적 판매금액", min_value=0, step=1000)
        
        if st.button("➕ 추가", type="primary"):
            if input_code and input_color and input_material:
                new_row = pd.DataFrame([{
                    '품번': input_code, '컬러': input_color, '제조방식': input_manufacturing,
                    '소재명': input_material, '핏': input_fit, '기장': input_length,
                    '누적판매수량': input_quantity, '누적판매금액': input_price
                }])
                st.session_state.sales_data = pd.concat([st.session_state.sales_data, new_row], ignore_index=True)
                st.success("✅ 추가 완료!")
                st.rerun()
            else:
                st.error("❌ 품번, 컬러, 소재명은 필수입니다.")
    
    with tab2:
        st.subheader("Excel 업로드")
        
        template = pd.DataFrame(columns=['품번', '컬러', '제조방식', '소재명', '핏', '기장', '누적판매수량', '누적판매금액'])
        template.loc[0] = ['TWRS10954', '블랙', '컷앤소', '면100%', 'slim', 'crop', 1250, 12500000]
        template.loc[1] = ['TMPO10953', '네이비', '우븐', '폴리80%', 'regular', 'mid', 850, 8500000]
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            template.to_excel(writer, index=False, sheet_name='판매데이터')
        
        st.download_button("📥 템플릿 다운로드", buffer.getvalue(), "판매데이터_템플릿.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        uploaded = st.file_uploader("Excel 파일 선택", type=['xlsx', 'xls'])
        
        if uploaded:
            try:
                df_upload = pd.read_excel(uploaded)
                st.dataframe(df_upload.head(10))
                
                if st.button("✅ 적용"):
                    st.session_state.sales_data = pd.concat([st.session_state.sales_data, df_upload], ignore_index=True)
                    st.success(f"✅ {len(df_upload)}개 추가!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ 오류: {e}")
    
    with tab3:
        st.subheader("소재 마스터 관리")
        
        template_mat = pd.DataFrame(columns=['소재명', '소재업체', '혼용율', '중량', '두께', '밀도'])
        template_mat.loc[0] = ['면100%', '태광섬유', '면100%', 180, 0.6, '고밀도']
        
        buffer2 = io.BytesIO()
        with pd.ExcelWriter(buffer2, engine='openpyxl') as writer:
            template_mat.to_excel(writer, index=False, sheet_name='소재데이터')
        
        st.download_button("📥 소재 템플릿", buffer2.getvalue(), "소재템플릿.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        uploaded_mat = st.file_uploader("소재 Excel", type=['xlsx', 'xls'])
        
        if uploaded_mat:
            try:
                df_mat = pd.read_excel(uploaded_mat)
                st.dataframe(df_mat)
                
                if st.button("✅ 소재 적용"):
                    st.session_state.material_data = pd.concat([st.session_state.material_data, df_mat], ignore_index=True)
                    st.success("✅ 소재 추가!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ 오류: {e}")
# 3. 대시보드
elif menu == "📊 대시보드":
    st.title("📊 판매 분석 대시보드")
    
    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 누적판매수량", f"{df['누적판매수량'].sum():,}개")
        with col2:
            st.metric("총 누적판매금액", f"{df['누적판매금액'].sum():,}원")
        with col3:
            total_qty = df['누적판매수량'].sum()
            total_amt = df['누적판매금액'].sum()
            avg_price = total_amt / total_qty if total_qty > 0 else 0
            st.metric("평균 판매가", f"{avg_price:,.0f}원")
        with col4:
            st.metric("총 SKU", f"{len(df)}개")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 성별 누적판매")
            gender_sales = df.groupby('성별')['누적판매수량'].sum()
            fig1 = px.pie(values=gender_sales.values, names=gender_sales.index, hole=0.4)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("🏭 제조방식별 누적판매")
            manu_sales = df.groupby('제조방식')['누적판매수량'].sum().sort_values(ascending=False)
            fig2 = px.bar(x=manu_sales.values, y=manu_sales.index, orientation='h',
                         color=manu_sales.values, color_continuous_scale='Blues')
            fig2.update_layout(showlegend=False, xaxis_title="누적판매수량", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("📁 카테고리별 누적판매")
            category_sales = df.groupby('카테고리')['누적판매수량'].sum().sort_values(ascending=False)
            fig3 = px.bar(x=category_sales.values, y=category_sales.index, orientation='h',
                         color=category_sales.values, color_continuous_scale='Greens')
            fig3.update_layout(showlegend=False, xaxis_title="누적판매수량", yaxis_title="")
            st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            st.subheader("👕 아이템별 TOP 10")
            item_sales = df.groupby('아이템명')['누적판매수량'].sum().sort_values(ascending=False).head(10)
            fig4 = px.bar(x=item_sales.values, y=item_sales.index, orientation='h',
                         color=item_sales.values, color_continuous_scale='Oranges')
            fig4.update_layout(showlegend=False, xaxis_title="누적판매수량", yaxis_title="")
            st.plotly_chart(fig4, use_container_width=True)
        
        st.divider()
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("🌈 컬러별 TOP 10")
            color_sales = df.groupby('컬러')['누적판매수량'].sum().sort_values(ascending=False).head(10)
            fig5 = px.bar(x=color_sales.values, y=color_sales.index, orientation='h',
                         color=color_sales.values, color_continuous_scale='Rainbow')
            fig5.update_layout(showlegend=False, xaxis_title="누적판매수량", yaxis_title="")
            st.plotly_chart(fig5, use_container_width=True)
        
        with col6:
            st.subheader("🌸 시즌별 누적판매")
            season_sales = df.groupby('시즌')['누적판매수량'].sum()
            fig6 = px.pie(values=season_sales.values, names=season_sales.index, hole=0.4)
            st.plotly_chart(fig6, use_container_width=True)

# 4. 조합 성과 랭킹
elif menu == "🏆 랭킹":
    st.title("🏆 조합별 성과 랭킹")
    
    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data)
        
        df['조합'] = df['성별'] + ' / ' + df['아이템명'] + ' / ' + df['제조방식'] + ' / ' + df['소재명'] + ' / ' + df['핏'] + ' / ' + df['기장']
        
        combo_stats = df.groupby('조합').agg({
            '누적판매수량': ['sum', 'mean', 'count'],
            '누적판매금액': ['sum', 'mean']
        }).round(0)
        
        combo_stats.columns = ['총누적판매수량', '평균누적판매수량', '데이터수', '총누적판매금액', '평균누적판매금액']
        combo_stats = combo_stats.reset_index()
        
        metric = st.radio("분석 기준", ["총누적판매수량", "평균누적판매수량", "총누적판매금액", "평균누적판매금액"], horizontal=True)
        top_n = st.slider("표시할 조합 수", 5, 20, 10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🥇 Best {top_n}")
            top_combos = combo_stats.nlargest(top_n, metric)
            
            fig_top = px.bar(top_combos, x=metric, y='조합', orientation='h',
                            color=metric, color_continuous_scale='Greens')
            fig_top.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)
            
            st.dataframe(top_combos, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader(f"🥉 Worst {top_n}")
            bottom_combos = combo_stats.nsmallest(top_n, metric)
            
            fig_bottom = px.bar(bottom_combos, x=metric, y='조합', orientation='h',
                               color=metric, color_continuous_scale='Reds')
            fig_bottom.update_layout(showlegend=False, yaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_bottom, use_container_width=True)
            
            st.dataframe(bottom_combos, use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("🔥 조합 히트맵")
        
        heatmap_x = st.selectbox("X축", ["아이템명", "제조방식", "소재명", "핏", "기장", "성별"])
        heatmap_y = st.selectbox("Y축", ["제조방식", "소재명", "핏", "기장", "아이템명", "성별"],
                                index=1 if heatmap_x == "아이템명" else 0)
        
        if heatmap_x != heatmap_y:
            pivot = df.pivot_table(values='누적판매수량', index=heatmap_y, columns=heatmap_x,
                                  aggfunc='sum', fill_value=0)
            fig_heat = px.imshow(pivot, color_continuous_scale='RdYlGn', aspect='auto')
            fig_heat.update_layout(xaxis_title=heatmap_x, yaxis_title=heatmap_y)
            st.plotly_chart(fig_heat, use_container_width=True)

# 5. 소재 분석
elif menu == "🧵 소재 분석":
    st.title("🧵 소재별 성과 분석")
    
    if st.session_state.sales_data.empty:
        st.warning("⚠️ 분석할 데이터가 없습니다.")
    else:
        df = enrich_sales_data(st.session_state.sales_data)
        
        material_stats = df.groupby('소재명').agg({
            '누적판매수량': ['sum', 'mean', 'count'],
            '누적판매금액': ['sum', 'mean'],
            '품번': 'nunique'
        }).round(0)
        
        material_stats.columns = ['총누적판매수량', '평균누적판매수량', '데이터수', '총누적판매금액', '평균누적판매금액', 'SKU수']
        material_stats = material_stats.reset_index().sort_values('총누적판매수량', ascending=False)
        
        st.subheader("📊 소재별 성과 요약")
        st.dataframe(material_stats, use_container_width=True, hide_index=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧵 소재별 총 누적판매수량")
            fig1 = px.bar(material_stats.head(10), x='총누적판매수량', y='소재명', orientation='h',
                         color='총누적판매수량', color_continuous_scale='Blues')
            fig1.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("💰 소재별 총 누적판매금액")
            fig2 = px.bar(material_stats.head(10), x='총누적판매금액', y='소재명', orientation='h',
                         color='총누적판매금액', color_continuous_scale='Greens')
            fig2.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()
        st.subheader("🔍 소재별 상세 분석")
        
        selected_material = st.selectbox("소재 선택", material_stats['소재명'].tolist())
        
        if selected_material:
            material_df = df[df['소재명'] == selected_material]
            
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("총 누적판매수량", f"{material_df['누적판매수량'].sum():,}개")
            with col4:
                st.metric("평균 누적판매수량", f"{material_df['누적판매수량'].mean():.0f}개")
            with col5:
                st.metric("사용 SKU", f"{material_df['품번'].nunique()}개")
            
            st.markdown(f"#### {selected_material} 아이템별 성과")
            item_perf = material_df.groupby('아이템명')['누적판매수량'].sum().sort_values(ascending=False)
            fig3 = px.bar(x=item_perf.values, y=item_perf.index, orientation='h',
                         color=item_perf.values, color_continuous_scale='Oranges')
            fig3.update_layout(showlegend=False, xaxis_title="누적판매수량", yaxis_title="")
            st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown(f"#### {selected_material} 제조방식별 성과")
            manu_perf = material_df.groupby('제조방식')['누적판매수량'].sum().sort_values(ascending=False)
            fig4 = px.pie(values=manu_perf.values, names=manu_perf.index, hole=0.4)
            st.plotly_chart(fig4, use_container_width=True)
            
            if not st.session_state.material_data.empty:
                material_info = st.session_state.material_data[
                    st.session_state.material_data['소재명'] == selected_material
                ]
                if not material_info.empty:
                    st.markdown("#### 📋 소재 상세 정보")
                    st.dataframe(material_info, use_container_width=True, hide_index=True)
# 6. 데이터 관리
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
                    '품번', '컬러', '제조방식', '소재명', '핏', '기장', '누적판매수량', '누적판매금액'
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

# 푸터
st.sidebar.divider()
st.sidebar.info(f"""
📊 **현재 데이터 현황**
- 판매 데이터: {len(st.session_state.sales_data)}건
- 소재 데이터: {len(st.session_state.material_data)}건

💡 **데이터 입력 방식**
- 누적 판매 방식 사용
- 현재까지의 총 판매량 입력

🏭 **제조방식**
- 컷앤소 (Cut & Sewn)
- 우븐 (Woven)
- 스웨터 (Sweater/Knit)

⚠️ **중요 안내**
- 매일 Excel로 백업하세요
- 브라우저 종료 시 데이터 소멸
""")

st.sidebar.markdown("---")
st.sidebar.caption("© 2024 세르지오타키니 판매분석시스템 v2.0")
