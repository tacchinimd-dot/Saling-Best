# ========================================
# Part 2: 대시보드, 조합 성과 랭킹, 소재 분석, 데이터 관리
# 이 코드를 Part 1의 마지막 부분에 이어서 붙여넣으세요
# ========================================

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
            st.metric("총 누적판매수량", f"{df['누적판매수량'].sum():,}개")
        with col2:
            st.metric("총 누적판매금액", f"{df['누적판매금액'].sum():,}원")
        with col3:
            st.metric("평균 판매가", f"{(df['누적판매금액'].sum() / df['누적판매수량'].sum()):,.0f}원")
        with col4:
            st.metric("총 SKU", f"{len(df)}개")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 성별 누적판매 분포")
            gender_sales = df.groupby('성별')['누적판매수량'].sum().sort_values(ascending=False)
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
            st.subheader("👕 아이템별 누적판매 TOP 10")
            item_sales = df.groupby('아이템명')['누적판매수량'].sum().sort_values(ascending=False).head(10)
            fig4 = px.bar(x=item_sales.values, y=item_sales.index, orientation='h',
                         color=item_sales.values, color_continuous_scale='Oranges')
            fig4.update_layout(showlegend=False, xaxis_title="누적판매수량", yaxis_title="")
            st.plotly_chart(fig4, use_container_width=True)
        
        st.divider()
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("🌈 컬러별 누적판매 TOP 10")
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

# ========================================
# 4. 조합 성과 랭킹
# ========================================
elif menu == "🏆 조합 성과 랭킹":
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
        
        heatmap_x = st.selectbox("X축", ["아이템명", "제조방식", "소재명", "핏", "기장", "성별"])
        heatmap_y = st.selectbox("Y축", ["제조방식", "소재명", "핏", "기장", "아이템명", "성별"],
                                index=1 if heatmap_x == "아이템명" else 0)
        
        if heatmap_x != heatmap_y:
            pivot = df.pivot_table(values='누적판매수량', index=heatmap_y, columns=heatmap_x,
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
            
            st.markdown(f"#### {selected_material} 사용 아이템별 성과")
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

# ========================================
# 푸터
# ========================================
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
