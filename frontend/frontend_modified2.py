import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import zipfile
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.plot_utils import plot_prediction

# Initialize session state for the map
if "map_created" not in st.session_state:
    st.session_state.map_created = False

# Load shape data function (this is unchanged)
def load_shape_data_file(
    data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True
):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"
    shapefile_path = extract_path / "taxi_zones.shp"

    if not zip_path.exists():
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)

    if not shapefile_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    return gpd.read_file(shapefile_path).to_crs("epsg:4326")


# Set page title
current_date = pd.Timestamp.now(tz="Etc/UTC")
st.title(f"New York Yellow Taxi Cab Demand Next Hour")
st.header(f'{current_date.strftime("%Y-%m-%d %H:%M:%S")}')

progress_bar = st.sidebar.header("Working Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 4

with st.spinner(text="Download shape file for taxi zones"):
    geo_df = load_shape_data_file(DATA_DIR)
    st.sidebar.write("Shape file was downloaded")
    progress_bar.progress(1 / N_STEPS)

with st.spinner(text="Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write("Inference features fetched from the store")
    progress_bar.progress(2 / N_STEPS)

with st.spinner(text="Fetching predictions"):
    predictions = fetch_next_hour_predictions()
    st.sidebar.write("Model was loaded from the registry")
    progress_bar.progress(3 / N_STEPS)

# Step 1 - Load taxi_zone_lookup.csv and merge to add pickup_location_name
lookup_path = r"C:\Users\vasub\Downloads\taxi_zone_lookup.csv"
taxi_zone_lookup = pd.read_csv(lookup_path)

predictions = predictions.merge(
    taxi_zone_lookup[['LocationID', 'Zone']],
    left_on='pickup_location_id',
    right_on='LocationID',
    how='left'
)

predictions.rename(columns={'Zone': 'pickup_location_name'}, inplace=True)
predictions.drop(columns=['LocationID'], inplace=True)

# Step 2 - Add Dropdown to Select Location
selected_location = st.sidebar.selectbox(
    'Select Pickup Location',
    predictions['pickup_location_name'].unique()
)

# Filter data for selected location
selected_location_data = predictions[predictions['pickup_location_name'] == selected_location]
selected_location_id = selected_location_data['pickup_location_id'].iloc[0]

# Filter features (for historical data if available)
selected_features = features[features['pickup_location_id'] == selected_location_id]

# Plot Predictions for Selected Location
st.subheader(f'Predicted Demand for {selected_location}')

plt.figure(figsize=(12, 6))
plt.plot(selected_location_data['pickup_hour'], selected_location_data['predicted_demand'], label='Predicted Demand', color='green')
plt.xlabel('Time')
plt.ylabel('Predicted Rides')
plt.legend()
st.pyplot(plt)

# Optional - Plot Actual Rides if Available
if 'actual_rides' in selected_features.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(selected_features['pickup_hour'], selected_features['actual_rides'], label='Actual Rides', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Actual Rides')
    plt.legend()
    st.pyplot(plt)

# Top 10 Locations Section (This is your original logic)
st.subheader("Top 10 Predicted Demand Locations")
st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))

top10 = (
    predictions.sort_values("predicted_demand", ascending=False)
    .head(10)["pickup_location_id"]
    .to_list()
)

for location_id in top10:
    fig = plot_prediction(
        features=features[features["pickup_location_id"] == location_id],
        prediction=predictions[predictions["pickup_location_id"] == location_id],
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Mapping and Visualization (Unchanged)
shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

def create_taxi_map(shapefile_path, prediction_data):
    nyc_zones = gpd.read_file(shapefile_path)

    nyc_zones = nyc_zones.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="LocationID",
        right_on="pickup_location_id",
        how="left",
    )
    nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)

    nyc_zones = nyc_zones.to_crs(epsg=4326)
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")

    colormap = LinearColormap(
        colors=["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026"],
        vmin=nyc_zones["predicted_demand"].min(),
        vmax=nyc_zones["predicted_demand"].max(),
    )
    colormap.add_to(m)

    def style_function(feature):
        predicted_demand = feature["properties"].get("predicted_demand", 0)
        return {
            "fillColor": colormap(float(predicted_demand)),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        }

    folium.GeoJson(
        nyc_zones.to_json(),
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone", "predicted_demand"],
            aliases=["Zone:", "Predicted Demand:"],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
        ),
    ).add_to(m)

    st.session_state.map_obj = m
    st.session_state.map_created = True
    return m

with st.spinner(text="Plot predicted rides demand"):
    st.subheader("Taxi Ride Predictions Map")
    map_obj = create_taxi_map(shapefile_path, predictions)

    if st.session_state.map_created:
        st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Rides", f"{predictions['predicted_demand'].mean():.0f}")
    with col2:
        st.metric("Maximum Rides", f"{predictions['predicted_demand'].max():.0f}")
    with col3:
        st.metric("Minimum Rides", f"{predictions['predicted_demand'].min():.0f}")

    st.sidebar.write("Finished plotting taxi rides demand")
    progress_bar.progress(4 / N_STEPS)
