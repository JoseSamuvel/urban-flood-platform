"""
flood_data_engine.py
Generates a realistic, domain-grounded Urban Flood Impact dataset aligned with:
- Suggested Inputs: Rainfall, water levels, drainage indicators, elevation, land use, historical events, location.
- Predefined Flood-Impact Levels (Classification):
    0: Low Risk (Normal drainage, minor surface wetness)
    1: Moderate Risk (Waterlogging on roads 10-30cm, traffic diversion)
    2: High Risk (Inundation 30-75cm, residential ingress, relief centers)
    3: Severe Inundation (>75cm, critical hazard, evacuation required)
"""

import os
import random
import numpy as np
import pandas as pd

# Define 20 representative urban wards across diverse topographical sectors
WARDS = [
    {"ward_id": "W001", "ward_name": "T. Nagar Commercial Hub", "zone": "Central", "lat": 13.0418, "lon": 80.2341, "elevation_base": 6.2, "slope_base": 0.8, "land_use": "Commercial", "drain_density_base": 3.8, "hist_freq": 8, "hist_max_depth": 90},
    {"ward_id": "W002", "ward_name": "Velachery Lowlands", "zone": "South", "lat": 12.9815, "lon": 80.2180, "elevation_base": 3.2, "slope_base": 0.4, "land_use": "High_Density_Residential", "drain_density_base": 2.1, "hist_freq": 12, "hist_max_depth": 140},
    {"ward_id": "W003", "ward_name": "Madipakkam Lake Basin", "zone": "South", "lat": 12.9647, "lon": 80.1961, "elevation_base": 2.5, "slope_base": 0.3, "land_use": "High_Density_Residential", "drain_density_base": 1.8, "hist_freq": 14, "hist_max_depth": 160},
    {"ward_id": "W004", "ward_name": "Guindy Ridge Sector", "zone": "South-Central", "lat": 13.0067, "lon": 80.2025, "elevation_base": 14.2, "slope_base": 2.1, "land_use": "Industrial", "drain_density_base": 4.2, "hist_freq": 4, "hist_max_depth": 45},
    {"ward_id": "W005", "ward_name": "Adyar Estuary Buffer", "zone": "Coastal", "lat": 13.0012, "lon": 80.2565, "elevation_base": 4.1, "slope_base": 0.6, "land_use": "Green_Space", "drain_density_base": 2.9, "hist_freq": 9, "hist_max_depth": 110},
    {"ward_id": "W006", "ward_name": "Anna Nagar High Ridge", "zone": "West", "lat": 13.0850, "lon": 80.2101, "elevation_base": 18.5, "slope_base": 2.8, "land_use": "Low_Density_Residential", "drain_density_base": 5.0, "hist_freq": 1, "hist_max_depth": 20},
    {"ward_id": "W007", "ward_name": "Mylapore Heritage Sector", "zone": "East", "lat": 13.0368, "lon": 80.2676, "elevation_base": 8.5, "slope_base": 1.2, "land_use": "High_Density_Residential", "drain_density_base": 4.5, "hist_freq": 5, "hist_max_depth": 55},
    {"ward_id": "W008", "ward_name": "Tambaram Outer Plateau", "zone": "South-West", "lat": 12.9249, "lon": 80.1000, "elevation_base": 22.0, "slope_base": 3.2, "land_use": "Low_Density_Residential", "drain_density_base": 3.2, "hist_freq": 3, "hist_max_depth": 40},
    {"ward_id": "W009", "ward_name": "Perambur Railway Lowlands", "zone": "North", "lat": 13.1114, "lon": 80.2437, "elevation_base": 4.8, "slope_base": 0.5, "land_use": "Commercial", "drain_density_base": 2.5, "hist_freq": 10, "hist_max_depth": 125},
    {"ward_id": "W010", "ward_name": "Kolathur Urban Catchment", "zone": "North-West", "lat": 13.1235, "lon": 80.2188, "elevation_base": 8.8, "slope_base": 1.4, "land_use": "High_Density_Residential", "drain_density_base": 3.0, "hist_freq": 7, "hist_max_depth": 85},
    {"ward_id": "W011", "ward_name": "Sholinganallur IT Corridor", "zone": "South-East", "lat": 12.9010, "lon": 80.2279, "elevation_base": 4.5, "slope_base": 0.5, "land_use": "Commercial", "drain_density_base": 3.4, "hist_freq": 9, "hist_max_depth": 115},
    {"ward_id": "W012", "ward_name": "Manali Petrochemical Lowland", "zone": "North-East", "lat": 13.1706, "lon": 80.2605, "elevation_base": 3.0, "slope_base": 0.4, "land_use": "Industrial", "drain_density_base": 2.0, "hist_freq": 11, "hist_max_depth": 135},
    {"ward_id": "W013", "ward_name": "Pallikaranai Marsh Reserve", "zone": "South-East", "lat": 12.9360, "lon": 80.2140, "elevation_base": 2.0, "slope_base": 0.2, "land_use": "Green_Space", "drain_density_base": 1.5, "hist_freq": 15, "hist_max_depth": 175},
    {"ward_id": "W014", "ward_name": "Porur Lake Fringe", "zone": "West", "lat": 13.0330, "lon": 80.1580, "elevation_base": 5.5, "slope_base": 0.7, "land_use": "High_Density_Residential", "drain_density_base": 2.4, "hist_freq": 9, "hist_max_depth": 105},
    {"ward_id": "W015", "ward_name": "St. Thomas Mount Ridge", "zone": "South-Central", "lat": 13.0040, "lon": 80.1910, "elevation_base": 24.0, "slope_base": 3.5, "land_use": "Low_Density_Residential", "drain_density_base": 4.8, "hist_freq": 1, "hist_max_depth": 15},
    {"ward_id": "W016", "ward_name": "Koyambedu Market Heights", "zone": "West", "lat": 13.0690, "lon": 80.1940, "elevation_base": 16.8, "slope_base": 2.4, "land_use": "Commercial", "drain_density_base": 4.0, "hist_freq": 3, "hist_max_depth": 35},
    {"ward_id": "W017", "ward_name": "Saidapet Riverine Basin", "zone": "Central", "lat": 13.0210, "lon": 80.2230, "elevation_base": 3.8, "slope_base": 0.5, "land_use": "High_Density_Residential", "drain_density_base": 2.8, "hist_freq": 11, "hist_max_depth": 130},
    {"ward_id": "W018", "ward_name": "Ambattur Industrial Zone", "zone": "North-West", "lat": 13.1140, "lon": 80.1540, "elevation_base": 11.2, "slope_base": 1.8, "land_use": "Industrial", "drain_density_base": 3.6, "hist_freq": 5, "hist_max_depth": 60},
    {"ward_id": "W019", "ward_name": "Royapettah Central", "zone": "Central", "lat": 13.0530, "lon": 80.2610, "elevation_base": 9.5, "slope_base": 1.5, "land_use": "Commercial", "drain_density_base": 4.2, "hist_freq": 4, "hist_max_depth": 50},
    {"ward_id": "W020", "ward_name": "Mudichur Lake Sink", "zone": "South-West", "lat": 12.9120, "lon": 80.0680, "elevation_base": 2.2, "slope_base": 0.3, "land_use": "High_Density_Residential", "drain_density_base": 1.6, "hist_freq": 14, "hist_max_depth": 165}
]

LAND_USE_RUNOFF_COEFF = {
    "Commercial": 0.85,
    "Industrial": 0.80,
    "High_Density_Residential": 0.75,
    "Low_Density_Residential": 0.55,
    "Green_Space": 0.25
}

def generate_flood_dataset(n_samples: int = 3500, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic realistic records mapping hydrological conditions
    to ground-truth flood impact levels.
    """
    np.random.seed(seed)
    random.seed(seed)

    records = []

    for i in range(n_samples):
        # Pick a ward
        ward = random.choice(WARDS)

        # 1. Weather Inputs (Rainfall)
        weather_regime = np.random.choice(["light", "moderate", "heavy", "extreme"], p=[0.35, 0.30, 0.23, 0.12])
        if weather_regime == "light":
            rainfall_24h = np.random.uniform(2.0, 35.0)
            rainfall_intensity = np.random.uniform(1.0, 10.0)
            reservoir_discharge = np.random.uniform(100, 1000)
        elif weather_regime == "moderate":
            rainfall_24h = np.random.uniform(35.0, 90.0)
            rainfall_intensity = np.random.uniform(8.0, 25.0)
            reservoir_discharge = np.random.uniform(800, 3500)
        elif weather_regime == "heavy":
            rainfall_24h = np.random.uniform(90.0, 180.0)
            rainfall_intensity = np.random.uniform(20.0, 50.0)
            reservoir_discharge = np.random.uniform(3000, 8000)
        else: # extreme cloudburst/cyclone
            rainfall_24h = np.random.uniform(180.0, 340.0)
            rainfall_intensity = np.random.uniform(45.0, 95.0)
            reservoir_discharge = np.random.uniform(7500, 18000)

        # 2. Terrain & Elevation with micro-topography variance
        elevation = max(1.5, ward["elevation_base"] + np.random.normal(0, 0.4))
        slope = max(0.2, ward["slope_base"] + np.random.normal(0, 0.15))

        # Water level physically accumulates in lowlands
        elev_accum = max(0.0, 18.0 - elevation) * 0.18
        water_level = max(0.4, elev_accum + (rainfall_24h / 45.0) * (1.2 - min(0.8, elevation / 25.0)) + np.random.normal(0, 0.2))

        # 3. Drainage Indicators
        siltation_level = np.clip(np.random.beta(2, 3) * 100, 5.0, 95.0)
        drain_density = max(1.0, ward["drain_density_base"] + np.random.normal(0, 0.2))
        drain_capacity_pct = np.clip(100.0 - siltation_level + np.random.normal(0, 5), 5.0, 100.0)

        # 4. Land Use & Runoff
        land_use = ward["land_use"]
        runoff_coeff = LAND_USE_RUNOFF_COEFF[land_use]

        # 5. Historical metrics
        hist_freq = ward["hist_freq"]
        hist_max_depth = ward["hist_max_depth"]

        # 6. Physical Inundation Vulnerability Index (Hydrological Calculation)
        water_burden = (rainfall_24h * 0.40) + (rainfall_intensity * 1.1) + (water_level * 16.0) + (reservoir_discharge / 600.0)
        runoff_factor = 1.0 + (runoff_coeff * 0.7)
        drainage_damping = (drain_capacity_pct * 0.4) + (drain_density * 6.0) + (elevation * 7.5) + (slope * 10.0)

        raw_flood_index = (water_burden * runoff_factor) - drainage_damping + (hist_freq * 1.2) + np.random.normal(0, 6.0)

        if raw_flood_index < 35:
            impact_level = 0
            impact_label = "Low"
        elif raw_flood_index < 85:
            impact_level = 1
            impact_label = "Moderate"
        elif raw_flood_index < 145:
            impact_level = 2
            impact_label = "High"
        else:
            impact_level = 3
            impact_label = "Severe Inundation"

        records.append({
            "ward_id": ward["ward_id"],
            "ward_name": ward["ward_name"],
            "zone": ward["zone"],
            "latitude": ward["lat"],
            "longitude": ward["lon"],
            "rainfall_24h_mm": round(float(rainfall_24h), 2),
            "rainfall_intensity_mm_hr": round(float(rainfall_intensity), 2),
            "water_level_m": round(float(water_level), 2),
            "reservoir_discharge_cusecs": round(float(reservoir_discharge), 1),
            "elevation_m": round(float(elevation), 2),
            "slope_degrees": round(float(slope), 2),
            "drain_density_km_sqkm": round(float(drain_density), 2),
            "siltation_level_pct": round(float(siltation_level), 1),
            "drain_capacity_pct": round(float(drain_capacity_pct), 1),
            "land_use": land_use,
            "past_flood_frequency": hist_freq,
            "historical_max_depth_cm": hist_max_depth,
            "flood_impact_level": impact_level,
            "flood_impact_label": impact_label
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    output_path = "urban_flood_dataset.csv"
    print("Generating Urban Flood Impact dataset...")
    df = generate_flood_dataset(n_samples=3500)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created with {len(df)} records at: {output_path}")
    print("\nClass distribution:")
    print(df["flood_impact_label"].value_counts(normalize=True).round(3))
