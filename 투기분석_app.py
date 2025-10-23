import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# 1. 유틸리티 함수 정의
# ==============================================================================

@st.cache_data
def load_data(uploaded_file):
    """업로드된 CSV 파일을 읽어 DataFrame을 반환 (cp949, utf-8 순차 시도)."""
    if uploaded_file is None:
        return None
    
    # Streamlit file_uploader는 BytesIO 객체를 반환하므로 .seek(0)을 사용해야 함
    for encoding_type in ['cp949', 'utf-8']:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding_type, header=0)
            uploaded_file.seek(0)
            
            # 컬럼명 정리: 공백을 밑줄로 변환
            df.columns = df.columns.str.replace(r'\s+', '_', regex=True).str.strip()
            return df
        except Exception:
            uploaded_file.seek(0)
            continue
    return None

@st.cache_data
def standardize_and_melt_data(df, region_col_name, value_col_name, national_key=None):
    """데이터프레임을 롱 포맷으로 변환하고 날짜 컬럼을 정리합니다."""
    
    # 키 컬럼을 제외한 나머지 컬럼 (날짜 컬럼) 식별
    # 2020년 1월과 같은 패턴을 가진 컬럼을 시간 컬럼으로 간주
    date_cols = [col for col in df.columns if '년' in col and '월' in col]

    if not date_cols:
        st.error(f"[{value_col_name} 파일 오류] '2020년 1월'과 같은 형식의 날짜 컬럼을 찾을 수 없습니다.")
        return None

    # 와이드 포맷 데이터를 숫자형으로 변환 (오류 발생 시 NaN)
    df[date_cols] = df[date_cols].apply(pd.to_numeric, errors='coerce')
    
    # Wide Format -> Long Format 변환 (Melt)
    df_long = pd.melt(
        df, 
        id_vars=[col for col in df.columns if col not in date_cols], # 모든 비-날짜 컬럼을 ID로 사용
        value_vars=date_cols,
        var_name='Date_Str', 
        value_name=value_col_name
    )
    
    # 날짜 문자열 정리 및 객체 변환
    df_long['Date_Str'] = df_long['Date_Str'].str.replace(r'[^\d\s]', '', regex=True).str.replace(' ', '')
    df_long['Date'] = pd.to_datetime(df_long['Date_Str'], format='%Y%m')
    
    # 최종적으로 필요한 컬럼만 선택
    df_long = df_long[[region_col_name, 'Date', value_col_name]].rename(
        columns={region_col_name: 'Region_ID_Clean'}
    )
    
    # 값이 NaN인 행 제거
    df_long = df_long.dropna(subset=['Region_ID_Clean', 'Date', value_col_name])
    
    # '전국' 행 제거 및 전국 데이터 추출
    df_national = df_long[df_long['Region_ID_Clean'].astype(str) == national_key].copy()
    df_national = df_national.rename(columns={value_col_name: f'National_{value_col_name}'})
    
    # 지역 분석 데이터 (전국 행 제외)
    df_long = df_long[df_long['Region_ID_Clean'].astype(str) != national_key].copy()
    
    return df_long, df_national.drop(columns=['Region_ID_Clean'])

# ==============================================================================
# 2. 분석 지표 계산 함수
# ==============================================================================

def calculate_analysis_metrics(df_combined, latest_date):
    """통합된 데이터프레임을 사용하여 P, V, S 지수를 계산하고 최근 시점 데이터를 반환합니다."""
    
    df = df_combined.copy()
    
    # 날짜를 기준으로 정렬
    df = df.sort_values(by=['Region_ID_Clean', 'Date']).reset_index(drop=True)
    
    # ------------------------------------------------------------------
    # 1. 가격 과열 지수 (P) 계산
    # ------------------------------------------------------------------

    # 3개월 전 가격지수 (분모)
    df['Price_Index_3M_Ago'] = df.groupby('Region_ID_Clean')['Price_Index'].shift(3)
    df['National_Price_Index_3M_Ago'] = df.groupby('Region_ID_Clean')['National_Price_Index'].shift(3)
    
    # 최근 3개월 가격 상승률
    df['P_Change_Local'] = df['Price_Index'] / df['Price_Index_3M_Ago'] - 1
    df['P_Change_National'] = df['National_Price_Index'] / df['National_Price_Index_3M_Ago'] - 1
    
    # **가격 과열 지수 (P): 전국 대비 상대적 상승률**
    epsilon = 1e-6
    # 분모(전국 상승률)가 0에 가까울 때 inf 방지 및 전국 대비 상승률 계산
    df['Price_Overheating_Index_P'] = df['P_Change_Local'] / (df['P_Change_National'].abs() + epsilon)

    # ------------------------------------------------------------------
    # 2. 거래량 모멘텀 지수 (V) 계산
    # ------------------------------------------------------------------

    # 12개월 월평균 거래량 (직전 1년 평균, 1개월 shift는 롤링 윈도우 계산에 포함)
    # min_periods=12: 12개월 데이터가 있어야 평균을 계산 (정확한 비교를 위해)
    df['Volume_12M_Avg'] = df.groupby('Region_ID_Clean')['Volume'].transform(
        lambda x: x.rolling(window=12, min_periods=12).mean().shift(1)
    )
    
    # 최근 3개월 월평균 거래량
    df['Volume_3M_Avg'] = df.groupby('Region_ID_Clean')['Volume'].transform(
        lambda x: x.rolling(window=3, min_periods=3).mean()
    )

    # **거래량 모멘텀 지수 (V)**
    # 12개월 평균이 0인 경우를 대비해 작은 값(epsilon)으로 나누어 inf 방지
    df['Volume_Momentum_Index_V'] = df['Volume_3M_Avg'] / (df['Volume_12M_Avg'] + epsilon)

    # ------------------------------------------------------------------
    # 3. 과열 종합 점수 (S = P * V)
    # ------------------------------------------------------------------
    df['Overheating_Score_S'] = df['Price_Overheating_Index_P'] * df['Volume_Momentum_Index_V']
    
    # 최종 결과 필터링: 계산에 필요한 최소 기간(15개월) 및 NaN 제거
    df_result = df.dropna(subset=['Overheating_Score_S'])
    
    # 최근 시점의 데이터만 추출
    df_latest = df_result[df_result['Date'] == latest_date].copy()
    
    return df_latest

def run_top10_analysis(df_price, df_volume, price_key, volume_key, national_key):
    """전체 통합 및 분석 프로세스를 실행합니다."""
    
    # 1. 롱 포맷 변환 및 시간 축 통일
    df_price_long, df_national_price = standardize_and_melt_data(
        df_price, price_key, 'Price_Index', national_key
    )
    df_volume_long, _ = standardize_and_melt_data(
        df_volume, volume_key, 'Volume' # 거래량은 전국 데이터가 필요 없음
    )

    if df_price_long is None or df_volume_long is None:
        return None, "[오류] 롱 포맷 변환 중 오류가 발생했습니다. 날짜 및 키 컬럼 지정을 확인해주세요."

    # 2. 키 매핑 대신 키 통일 후 병합
    # 두 파일의 키가 다르므로, 지역명(Region_ID_Clean)으로 통일한 뒤 병합을 시도합니다.
    # 단, 두 파일의 지역 컬럼 값이 정확히 일치해야 함.
    
    # 3. 전국 평균 가격지수 데이터 추가
    df_price_long = pd.merge(df_price_long, df_national_price, on='Date', how='left')
    
    # 4. 최종 통합 (Region_ID_Clean, Date 기준)
    df_combined = pd.merge(
        df_price_long,
        df_volume_long,
        on=['Region_ID_Clean', 'Date'],
        how='inner' # 두 데이터 모두에 존재하는 시점과 지역만 남김
    )
    
    if df_combined.empty:
        return None, "[오류] 통합된 데이터가 없습니다. (두 파일의 '지역 키 컬럼' 값 또는 날짜 범위 불일치)"
    
    # 5. 분석 지표 계산 및 Top 10 선정
    latest_date = df_combined['Date'].max()
    df_metrics = calculate_analysis_metrics(df_combined, latest_date)
    
    if df_metrics.empty:
        return None, "[오류] 분석에 필요한 최소 데이터 기간(15개월)을 충족하지 못했습니다."

    # 6. Top 10 선정 (과열 종합 점수 S 기준)
    # 최소 기준 적용: P >= 1.1 AND V >= 1.2
    df_filtered = df_metrics[
        (df_metrics['Price_Overheating_Index_P'] >= 1.1) & 
        (df_metrics['Volume_Momentum_Index_V'] >= 1.2)
    ].copy()
    
    top_10_results = df_filtered.nlargest(10, 'Overheating_Score_S')

    # 최종 결과 포맷팅
    top_10_results = top_10_results[[
        'Region_ID_Clean', 
        'Overheating_Score_S', 
        'Price_Overheating_Index_P', 
        'Volume_Momentum_Index_V'
    ]].sort_values(by='Overheating_Score_S', ascending=False)
    
    top_10_results.columns = [
        '지역명', 
        '과열 종합 점수 (S=P*V)', 
        '가격 과열 지수 (P)', 
        '거래량 모멘텀 지수 (V)'
    ]
    
    for col in ['과열 종합 점수 (S=P*V)', '가격 과열 지수 (P)', '거래량 모멘텀 지수 (V)']:
        top_10_results[col] = top_10_results[col].map('{:,.2f}'.format)
        
    return latest_date, top_10_results.set_index('지역명')

# ==============================================================================
# 3. Streamlit 메인 앱 실행
# ==============================================================================

def main():
    st.set_page_config(layout="wide", page_title="부동산 투기지역 검토 분석기")
    st.title("?? 부동산 투기지역 후보 검토 분석기 (P & V 지수 기반)")
    st.markdown("---")

    # --- 1. 파일 업로드 ---
    st.header("1. 데이터 파일 업로드")
    col1, col2 = st.columns(2)
    with col1:
        price_file = st.file_uploader("가격지수 파일 (Wide Format CSV)", type="csv", key='price')
    with col2:
        volume_file = st.file_uploader("거래량 파일 (Wide Format CSV)", type="csv", key='volume')

    # --- 2. 키 설정 (사이드바) ---
    st.sidebar.header("2. 데이터 키 설정")

    # 예시 컬럼명 자동 지정
    default_price_key = '지역' 
    default_volume_key = '통합지역'
    default_national_key = '전국'

    price_key = st.sidebar.text_input(
        "가격지수 파일의 지역 키 컬럼명", 
        value=default_price_key, 
        help="행정구역 이름이 있는 컬럼명 (예: 지역)"
    )
    volume_key = st.sidebar.text_input(
        "거래량 파일의 지역 키 컬럼명", 
        value=default_volume_key, 
        help="행정구역 이름이 있는 컬럼명 (예: 통합지역)"
    )
    national_key = st.sidebar.text_input(
        "전국 평균 데이터 행의 키 값", 
        value=default_national_key, 
        help="가격지수 파일에서 전국 평균을 나타내는 행의 '지역 키 컬럼' 값 (예: 전국)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("?? **분석 기준**")
    st.sidebar.markdown(
        "- **P (가격 과열)**: 최근 3개월 지역 상승률이 전국 대비 **1.1배 이상**\n"
        "- **V (거래량 모멘텀)**: 최근 3개월 거래량 평균이 직전 12개월 평균 대비 **1.2배 이상**"
    )

    # --- 3. 데이터 로드 ---
    if price_file is None or volume_file is None:
        st.info("두 파일을 모두 업로드하고 키 설정을 확인해주세요.")
        return

    df_price = load_data(price_file)
    df_volume = load_data(volume_file)
    
    if df_price is None or df_volume is None:
        st.error("파일 로드에 실패했습니다. 유효한 CSV 형식인지 확인해주세요.")
        return

    # --- 4. 분석 실행 ---
    st.markdown("---")
    st.header("3. 분석 실행 및 결과")
    
    # 사용자 입력 컬럼명 존재 여부 확인
    if price_key not in df_price.columns or volume_key not in df_volume.columns:
        st.error(f"지정한 키 컬럼명이 파일에 없습니다. 가격 파일: '{price_key}', 거래량 파일: '{volume_key}'")
        return

    with st.spinner("데이터 통합 및 투기지표를 계산 중입니다... (최소 15개월 데이터 필요)"):
        latest_date, results = run_top10_analysis(df_price, df_volume, price_key, volume_key, national_key)

    # --- 5. 결과 출력 ---
    if isinstance(results, str):
        st.error(results) # 오류 메시지 출력
    elif results is not None:
        
        st.subheader(f"?? {latest_date.strftime('%Y년 %m월')} 기준 과열 종합 점수 (S) TOP {len(results)} 지역")
        st.markdown(
            "잠재적 투기지역 후보를 **가격 과열 지수 (P)**와 **거래량 모멘텀 지수 (V)**의 곱인 **과열 종합 점수 (S)**를 기준으로 선정했습니다."
        )
        st.caption("선정 기준: **P ≥ 1.1** 및 **V ≥ 1.2**를 모두 만족하는 지역 중 S점수 상위 10개")
        
        st.dataframe(results, use_container_width=True)

if __name__ == "__main__":
    main()