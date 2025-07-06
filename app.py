import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title="Road Accident Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/accidents_2015_2023.csv")
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + "-" + df['Month'].astype(str) + "-01")
    return df

df = load_data()

st.title("🚦 Road Accident Pattern Analysis in India (2015–2023)")

states = df['State'].unique()
years = sorted(df['Year'].unique())

st.sidebar.header("📊 Filters")
selected_state = st.sidebar.selectbox("Select State", ["All"] + list(states))
year_range = st.sidebar.slider("Select Year Range", int(min(years)), int(max(years)), (2018, 2022))

filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
if selected_state != "All":
    filtered_df = filtered_df[filtered_df['State'] == selected_state]

# Summary Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Accidents", int(filtered_df['Accidents'].sum()))
col2.metric("Total Injuries", int(filtered_df['Injuries'].sum()))
col3.metric("Total Deaths", int(filtered_df['Deaths'].sum()))

# Time Series Plot
st.subheader("📈 Monthly Accident Trend")
monthly = filtered_df.groupby(['Year', 'Month'])[['Accidents']].sum().reset_index()
monthly['Date'] = pd.to_datetime(monthly['Year'].astype(str) + "-" + monthly['Month'].astype(str) + "-01")
monthly = monthly.set_index('Date').sort_index()
fig, ax = plt.subplots()
sns.lineplot(data=monthly, x=monthly.index, y="Accidents", ax=ax)
st.pyplot(fig)

# Seasonal Decomposition
st.subheader("📉 Seasonal Decomposition (Additive)")
if len(monthly) >= 24:
    decomposition = seasonal_decompose(monthly["Accidents"], model='additive', period=12)
    fig2 = decomposition.plot()
    st.pyplot(fig2)
else:
    st.warning("Not enough data points for decomposition (need at least 24).")

# Choropleth Map
st.subheader("🗺️ Accident Map (Choropleth by State)")
gdf = gpd.read_file("data/india_states_shapefile/india_states.shp")
state_data = filtered_df.groupby("State")[["Accidents"]].sum().reset_index()
merged = gdf.merge(state_data, left_on="st_nm", right_on="State")
m = folium.Map(location=[22.5, 80], zoom_start=5)
folium.Choropleth(
    geo_data=merged,
    name='choropleth',
    data=merged,
    columns=["State", "Accidents"],
    key_on="feature.properties.st_nm",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Total Accidents"
).add_to(m)
folium.LayerControl().add_to(m)
folium_static(m)
