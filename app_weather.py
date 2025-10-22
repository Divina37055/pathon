import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
import folium
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.title("🌦️ Open-Meteo Interactive Weather Dashboard")
st.write("지도에서 위치를 클릭하면 해당 지역의 시간별 기온 데이터를 불러옵니다.")

m = folium.Map(location=[37.5665, 126.9780], zoom_start=5)
clicked = st_folium(m, width=700, height=500)

if clicked and clicked.get("last_clicked"):
    lat = float(clicked["last_clicked"]["lat"])
    lon = float(clicked["last_clicked"]["lng"])
    st.success(f"📍 선택된 위치: 위도 {lat:.4f}, 경도 {lon:.4f}")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, verify=False, timeout=20)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame({
            "time": data["hourly"]["time"],
            "temperature (°C)": data["hourly"]["temperature_2m"]
        })

        fig = px.line(df, x="time", y="temperature (°C)",
                      title=f"{lat:.2f}, {lon:.2f} 지역의 시간별 기온",
                      labels={"time": "시간", "temperature (°C)": "기온(℃)"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.head(24))
    except Exception as e:
        st.error(f"데이터 요청 중 오류 발생: {e}")
else:
    st.info("지도를 클릭하면 해당 지역의 날씨 데이터를 가져옵니다.")
