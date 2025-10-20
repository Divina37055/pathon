import streamlit as st

# 제목
st.title("💰 소득세 계산기")

# 소득 입력
income = st.number_input("연 소득을 입력하세요 (원):", min_value=0, step=100000, format="%.0f")

# 계산 함수
def calculate_income_tax(income):
    """
    소득세 계산 함수
    :param income: 연 소득 (단위: 원)
    :return: 세금액 (단위: 원)
    """
    tax = 0

    if income <= 12000000:
        tax = income * 0.06
    elif income <= 46000000:
        tax = 720000 + (income - 12000000) * 0.15
    elif income <= 88000000:
        tax = 5820000 + (income - 46000000) * 0.24
    elif income <= 150000000:
        tax = 15900000 + (income - 88000000) * 0.35
    elif income <= 300000000:
        tax = 37600000 + (income - 150000000) * 0.38
    elif income <= 500000000:
        tax = 94600000 + (income - 300000000) * 0.40
    elif income <= 1000000000:
        tax = 174600000 + (income - 500000000) * 0.42
    else:
        tax = 384600000 + (income - 1000000000) * 0.45

    return int(tax)

# 계산 버튼
if st.button("세금 계산하기"):
    tax = calculate_income_tax(income)
    st.subheader("📊 계산 결과")
    st.write(f"연 소득: {income:,.0f} 원")
    st.success(f"예상 세금: {tax:,.0f} 원")
