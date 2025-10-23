import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# 1. ��ƿ��Ƽ �Լ� ����
# ==============================================================================

@st.cache_data
def load_data(uploaded_file):
    """���ε�� CSV ������ �о� DataFrame�� ��ȯ (cp949, utf-8 ���� �õ�)."""
    if uploaded_file is None:
        return None
    
    # Streamlit file_uploader�� BytesIO ��ü�� ��ȯ�ϹǷ� .seek(0)�� ����ؾ� ��
    for encoding_type in ['cp949', 'utf-8']:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding_type, header=0)
            uploaded_file.seek(0)
            
            # �÷��� ����: ������ ���ٷ� ��ȯ
            df.columns = df.columns.str.replace(r'\s+', '_', regex=True).str.strip()
            return df
        except Exception:
            uploaded_file.seek(0)
            continue
    return None

@st.cache_data
def standardize_and_melt_data(df, region_col_name, value_col_name, national_key=None):
    """�������������� �� �������� ��ȯ�ϰ� ��¥ �÷��� �����մϴ�."""
    
    # Ű �÷��� ������ ������ �÷� (��¥ �÷�) �ĺ�
    # 2020�� 1���� ���� ������ ���� �÷��� �ð� �÷����� ����
    date_cols = [col for col in df.columns if '��' in col and '��' in col]

    if not date_cols:
        st.error(f"[{value_col_name} ���� ����] '2020�� 1��'�� ���� ������ ��¥ �÷��� ã�� �� �����ϴ�.")
        return None

    # ���̵� ���� �����͸� ���������� ��ȯ (���� �߻� �� NaN)
    df[date_cols] = df[date_cols].apply(pd.to_numeric, errors='coerce')
    
    # Wide Format -> Long Format ��ȯ (Melt)
    df_long = pd.melt(
        df, 
        id_vars=[col for col in df.columns if col not in date_cols], # ��� ��-��¥ �÷��� ID�� ���
        value_vars=date_cols,
        var_name='Date_Str', 
        value_name=value_col_name
    )
    
    # ��¥ ���ڿ� ���� �� ��ü ��ȯ
    df_long['Date_Str'] = df_long['Date_Str'].str.replace(r'[^\d\s]', '', regex=True).str.replace(' ', '')
    df_long['Date'] = pd.to_datetime(df_long['Date_Str'], format='%Y%m')
    
    # ���������� �ʿ��� �÷��� ����
    df_long = df_long[[region_col_name, 'Date', value_col_name]].rename(
        columns={region_col_name: 'Region_ID_Clean'}
    )
    
    # ���� NaN�� �� ����
    df_long = df_long.dropna(subset=['Region_ID_Clean', 'Date', value_col_name])
    
    # '����' �� ���� �� ���� ������ ����
    df_national = df_long[df_long['Region_ID_Clean'].astype(str) == national_key].copy()
    df_national = df_national.rename(columns={value_col_name: f'National_{value_col_name}'})
    
    # ���� �м� ������ (���� �� ����)
    df_long = df_long[df_long['Region_ID_Clean'].astype(str) != national_key].copy()
    
    return df_long, df_national.drop(columns=['Region_ID_Clean'])

# ==============================================================================
# 2. �м� ��ǥ ��� �Լ�
# ==============================================================================

def calculate_analysis_metrics(df_combined, latest_date):
    """���յ� �������������� ����Ͽ� P, V, S ������ ����ϰ� �ֱ� ���� �����͸� ��ȯ�մϴ�."""
    
    df = df_combined.copy()
    
    # ��¥�� �������� ����
    df = df.sort_values(by=['Region_ID_Clean', 'Date']).reset_index(drop=True)
    
    # ------------------------------------------------------------------
    # 1. ���� ���� ���� (P) ���
    # ------------------------------------------------------------------

    # 3���� �� �������� (�и�)
    df['Price_Index_3M_Ago'] = df.groupby('Region_ID_Clean')['Price_Index'].shift(3)
    df['National_Price_Index_3M_Ago'] = df.groupby('Region_ID_Clean')['National_Price_Index'].shift(3)
    
    # �ֱ� 3���� ���� ��·�
    df['P_Change_Local'] = df['Price_Index'] / df['Price_Index_3M_Ago'] - 1
    df['P_Change_National'] = df['National_Price_Index'] / df['National_Price_Index_3M_Ago'] - 1
    
    # **���� ���� ���� (P): ���� ��� ����� ��·�**
    epsilon = 1e-6
    # �и�(���� ��·�)�� 0�� ����� �� inf ���� �� ���� ��� ��·� ���
    df['Price_Overheating_Index_P'] = df['P_Change_Local'] / (df['P_Change_National'].abs() + epsilon)

    # ------------------------------------------------------------------
    # 2. �ŷ��� ����� ���� (V) ���
    # ------------------------------------------------------------------

    # 12���� ����� �ŷ��� (���� 1�� ���, 1���� shift�� �Ѹ� ������ ��꿡 ����)
    # min_periods=12: 12���� �����Ͱ� �־�� ����� ��� (��Ȯ�� �񱳸� ����)
    df['Volume_12M_Avg'] = df.groupby('Region_ID_Clean')['Volume'].transform(
        lambda x: x.rolling(window=12, min_periods=12).mean().shift(1)
    )
    
    # �ֱ� 3���� ����� �ŷ���
    df['Volume_3M_Avg'] = df.groupby('Region_ID_Clean')['Volume'].transform(
        lambda x: x.rolling(window=3, min_periods=3).mean()
    )

    # **�ŷ��� ����� ���� (V)**
    # 12���� ����� 0�� ��츦 ����� ���� ��(epsilon)���� ������ inf ����
    df['Volume_Momentum_Index_V'] = df['Volume_3M_Avg'] / (df['Volume_12M_Avg'] + epsilon)

    # ------------------------------------------------------------------
    # 3. ���� ���� ���� (S = P * V)
    # ------------------------------------------------------------------
    df['Overheating_Score_S'] = df['Price_Overheating_Index_P'] * df['Volume_Momentum_Index_V']
    
    # ���� ��� ���͸�: ��꿡 �ʿ��� �ּ� �Ⱓ(15����) �� NaN ����
    df_result = df.dropna(subset=['Overheating_Score_S'])
    
    # �ֱ� ������ �����͸� ����
    df_latest = df_result[df_result['Date'] == latest_date].copy()
    
    return df_latest

def run_top10_analysis(df_price, df_volume, price_key, volume_key, national_key):
    """��ü ���� �� �м� ���μ����� �����մϴ�."""
    
    # 1. �� ���� ��ȯ �� �ð� �� ����
    df_price_long, df_national_price = standardize_and_melt_data(
        df_price, price_key, 'Price_Index', national_key
    )
    df_volume_long, _ = standardize_and_melt_data(
        df_volume, volume_key, 'Volume' # �ŷ����� ���� �����Ͱ� �ʿ� ����
    )

    if df_price_long is None or df_volume_long is None:
        return None, "[����] �� ���� ��ȯ �� ������ �߻��߽��ϴ�. ��¥ �� Ű �÷� ������ Ȯ�����ּ���."

    # 2. Ű ���� ��� Ű ���� �� ����
    # �� ������ Ű�� �ٸ��Ƿ�, ������(Region_ID_Clean)���� ������ �� ������ �õ��մϴ�.
    # ��, �� ������ ���� �÷� ���� ��Ȯ�� ��ġ�ؾ� ��.
    
    # 3. ���� ��� �������� ������ �߰�
    df_price_long = pd.merge(df_price_long, df_national_price, on='Date', how='left')
    
    # 4. ���� ���� (Region_ID_Clean, Date ����)
    df_combined = pd.merge(
        df_price_long,
        df_volume_long,
        on=['Region_ID_Clean', 'Date'],
        how='inner' # �� ������ ��ο� �����ϴ� ������ ������ ����
    )
    
    if df_combined.empty:
        return None, "[����] ���յ� �����Ͱ� �����ϴ�. (�� ������ '���� Ű �÷�' �� �Ǵ� ��¥ ���� ����ġ)"
    
    # 5. �м� ��ǥ ��� �� Top 10 ����
    latest_date = df_combined['Date'].max()
    df_metrics = calculate_analysis_metrics(df_combined, latest_date)
    
    if df_metrics.empty:
        return None, "[����] �м��� �ʿ��� �ּ� ������ �Ⱓ(15����)�� �������� ���߽��ϴ�."

    # 6. Top 10 ���� (���� ���� ���� S ����)
    # �ּ� ���� ����: P >= 1.1 AND V >= 1.2
    df_filtered = df_metrics[
        (df_metrics['Price_Overheating_Index_P'] >= 1.1) & 
        (df_metrics['Volume_Momentum_Index_V'] >= 1.2)
    ].copy()
    
    top_10_results = df_filtered.nlargest(10, 'Overheating_Score_S')

    # ���� ��� ������
    top_10_results = top_10_results[[
        'Region_ID_Clean', 
        'Overheating_Score_S', 
        'Price_Overheating_Index_P', 
        'Volume_Momentum_Index_V'
    ]].sort_values(by='Overheating_Score_S', ascending=False)
    
    top_10_results.columns = [
        '������', 
        '���� ���� ���� (S=P*V)', 
        '���� ���� ���� (P)', 
        '�ŷ��� ����� ���� (V)'
    ]
    
    for col in ['���� ���� ���� (S=P*V)', '���� ���� ���� (P)', '�ŷ��� ����� ���� (V)']:
        top_10_results[col] = top_10_results[col].map('{:,.2f}'.format)
        
    return latest_date, top_10_results.set_index('������')

# ==============================================================================
# 3. Streamlit ���� �� ����
# ==============================================================================

def main():
    st.set_page_config(layout="wide", page_title="�ε��� �������� ���� �м���")
    st.title("?? �ε��� �������� �ĺ� ���� �м��� (P & V ���� ���)")
    st.markdown("---")

    # --- 1. ���� ���ε� ---
    st.header("1. ������ ���� ���ε�")
    col1, col2 = st.columns(2)
    with col1:
        price_file = st.file_uploader("�������� ���� (Wide Format CSV)", type="csv", key='price')
    with col2:
        volume_file = st.file_uploader("�ŷ��� ���� (Wide Format CSV)", type="csv", key='volume')

    # --- 2. Ű ���� (���̵��) ---
    st.sidebar.header("2. ������ Ű ����")

    # ���� �÷��� �ڵ� ����
    default_price_key = '����' 
    default_volume_key = '��������'
    default_national_key = '����'

    price_key = st.sidebar.text_input(
        "�������� ������ ���� Ű �÷���", 
        value=default_price_key, 
        help="�������� �̸��� �ִ� �÷��� (��: ����)"
    )
    volume_key = st.sidebar.text_input(
        "�ŷ��� ������ ���� Ű �÷���", 
        value=default_volume_key, 
        help="�������� �̸��� �ִ� �÷��� (��: ��������)"
    )
    national_key = st.sidebar.text_input(
        "���� ��� ������ ���� Ű ��", 
        value=default_national_key, 
        help="�������� ���Ͽ��� ���� ����� ��Ÿ���� ���� '���� Ű �÷�' �� (��: ����)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("?? **�м� ����**")
    st.sidebar.markdown(
        "- **P (���� ����)**: �ֱ� 3���� ���� ��·��� ���� ��� **1.1�� �̻�**\n"
        "- **V (�ŷ��� �����)**: �ֱ� 3���� �ŷ��� ����� ���� 12���� ��� ��� **1.2�� �̻�**"
    )

    # --- 3. ������ �ε� ---
    if price_file is None or volume_file is None:
        st.info("�� ������ ��� ���ε��ϰ� Ű ������ Ȯ�����ּ���.")
        return

    df_price = load_data(price_file)
    df_volume = load_data(volume_file)
    
    if df_price is None or df_volume is None:
        st.error("���� �ε忡 �����߽��ϴ�. ��ȿ�� CSV �������� Ȯ�����ּ���.")
        return

    # --- 4. �м� ���� ---
    st.markdown("---")
    st.header("3. �м� ���� �� ���")
    
    # ����� �Է� �÷��� ���� ���� Ȯ��
    if price_key not in df_price.columns or volume_key not in df_volume.columns:
        st.error(f"������ Ű �÷����� ���Ͽ� �����ϴ�. ���� ����: '{price_key}', �ŷ��� ����: '{volume_key}'")
        return

    with st.spinner("������ ���� �� ������ǥ�� ��� ���Դϴ�... (�ּ� 15���� ������ �ʿ�)"):
        latest_date, results = run_top10_analysis(df_price, df_volume, price_key, volume_key, national_key)

    # --- 5. ��� ��� ---
    if isinstance(results, str):
        st.error(results) # ���� �޽��� ���
    elif results is not None:
        
        st.subheader(f"?? {latest_date.strftime('%Y�� %m��')} ���� ���� ���� ���� (S) TOP {len(results)} ����")
        st.markdown(
            "������ �������� �ĺ��� **���� ���� ���� (P)**�� **�ŷ��� ����� ���� (V)**�� ���� **���� ���� ���� (S)**�� �������� �����߽��ϴ�."
        )
        st.caption("���� ����: **P �� 1.1** �� **V �� 1.2**�� ��� �����ϴ� ���� �� S���� ���� 10��")
        
        st.dataframe(results, use_container_width=True)

if __name__ == "__main__":
    main()