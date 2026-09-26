"""
flood_engine.py
Core inference, preparedness decision matrix, historical benchmarking,
and scenario simulation engine for the Urban Flood Platform.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Standard Wards metadata
from flood_data_engine import WARDS

HISTORICAL_BENCHMARKS = [
    {
        "event_id": "hist_2015",
        "name": "2015 Historic Deluge (Extreme Cloudburst)",
        "year": 2015,
        "rainfall_24h_mm": 340.0,
        "rainfall_intensity_mm_hr": 65.0,
        "water_level_m": 6.8,
        "reservoir_discharge_cusecs": 15000,
        "siltation_level_pct": 60.0,
        "description": "Catastrophic flooding across multiple river basins. City-wide grid failure."
    },
    {
        "event_id": "hist_2021",
        "name": "2021 Cyclone Michaung Surge",
        "year": 2021,
        "rainfall_24h_mm": 210.0,
        "rainfall_intensity_mm_hr": 42.0,
        "water_level_m": 4.9,
        "reservoir_discharge_cusecs": 8500,
        "siltation_level_pct": 50.0,
        "description": "Severe coastal surge and prolonged waterlogging in south urban basins."
    },
    {
        "event_id": "hist_2023",
        "name": "2023 Monsoon Outflow Event",
        "year": 2023,
        "rainfall_24h_mm": 115.0,
        "rainfall_intensity_mm_hr": 26.0,
        "water_level_m": 3.4,
        "reservoir_discharge_cusecs": 3200,
        "siltation_level_pct": 35.0,
        "description": "Moderate to high arterial waterlogging; mitigated by newly constructed storm drains."
    },
    {
        "event_id": "hist_normal",
        "name": "Standard Seasonal Monsoon Shower",
        "year": 2024,
        "rainfall_24h_mm": 28.0,
        "rainfall_intensity_mm_hr": 7.0,
        "water_level_m": 1.2,
        "reservoir_discharge_cusecs": 500,
        "siltation_level_pct": 25.0,
        "description": "Routine rainfall; successfully handled by primary and secondary stormwater drains."
    }
]

ACTION_PROTOCOLS = {
    0: {
        "status": "GREEN / NORMAL",
        "color": "#10b981",
        "summary": "Low Risk: Minor surface runoff.",
        "actions": [
            "Maintain routine municipal stormwater network inspection.",
            "Verify real-time water gauge telemetry every 4 hours.",
            "Normal traffic movements permitted on all arterial roads."
        ]
    },
    1: {
        "status": "YELLOW / ADVISORY",
        "color": "#f59e0b",
        "summary": "Moderate Risk: Localized road waterlogging (10-30 cm).",
        "actions": [
            "Pre-stage mobile dewatering pumps at known subway/underpass choke points.",
            "Issue commuter advisories for low-lying arterial corridors.",
            "Deploy civic teams to clear surface trash racks and storm drain grates."
        ]
    },
    2: {
        "status": "ORANGE / HIGH ALERT",
        "color": "#f97316",
        "summary": "High Risk: Significant residential ingress (30-75 cm).",
        "actions": [
            "Activate Zonal Disaster Response Units and field rescue stations.",
            "Open designated multi-purpose emergency relief shelters.",
            "Deploy high-capacity diesel suction pumps to residential colonies.",
            "Isolate ground-level electrical transformers in waterlogged sectors."
        ]
    },
    3: {
        "status": "RED / CRITICAL INUNDATION",
        "color": "#ef4444",
        "summary": "Severe Inundation: Extreme flood hazard (>75 cm). Life threat.",
        "actions": [
            "IMMEDIATE EVACUATION: Mobilize NDRF, SDRF, and inflatable rescue boats.",
            "Broadcast emergency cell broadcast alerts to all residents in zone.",
            "Shut down low-lying high-tension power grids to prevent electrocution.",
            "Initiate food packet drops and emergency medical camp mobilization."
        ]
    }
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "output_models", "flood_classifier_pipeline.joblib")
DEFAULT_META_PATH = os.path.join(BASE_DIR, "output_models", "model_metadata.json")


class FloodPreparednessEngine:
    def __init__(self, model_path=None, meta_path=None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        elif not os.path.isabs(model_path) and not os.path.exists(model_path):
            alt_path = os.path.join(BASE_DIR, model_path)
            if os.path.exists(alt_path):
                model_path = alt_path

        if meta_path is None:
            meta_path = DEFAULT_META_PATH
        elif not os.path.isabs(meta_path) and not os.path.exists(meta_path):
            alt_path = os.path.join(BASE_DIR, meta_path)
            if os.path.exists(alt_path):
                meta_path = alt_path

        self.wards = WARDS
        loaded_successfully = False

        if os.path.exists(model_path) and os.path.exists(meta_path):
            try:
                self.pipeline = joblib.load(model_path)
                with open(meta_path, "r") as f:
                    self.metadata = json.load(f)
                loaded_successfully = True
            except Exception as e:
                print(f"[FloodEngine Warning] Checkpoint load failed ({e}). Retraining model pipeline for local environment compatibility...")

        if not loaded_successfully:
            from train_flood_model import train_ml_pipeline
            data_csv = os.path.join(BASE_DIR, "urban_flood_dataset.csv")
            out_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else os.path.join(BASE_DIR, "output_models")
            self.pipeline, self.metadata = train_ml_pipeline(data_csv, output_dir=out_dir)

    def predict_location(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Classifies flood risk for a specific location input."""
        full_record = dict(input_dict)
        if "latitude" not in full_record:
            full_record["latitude"] = 13.0827
        if "longitude" not in full_record:
            full_record["longitude"] = 80.2707
        if "zone" not in full_record:
            full_record["zone"] = "Central"

        df = pd.DataFrame([full_record])
        pred_class = int(self.pipeline.predict(df)[0])
        probabilities = self.pipeline.predict_proba(df)[0].tolist()

        protocol = ACTION_PROTOCOLS[pred_class]

        return {
            "predicted_class": pred_class,
            "impact_label": self.metadata["class_names"][pred_class],
            "confidence": round(float(probabilities[pred_class]), 4),
            "probabilities": {
                self.metadata["class_names"][i]: round(float(p), 4) for i, p in enumerate(probabilities)
            },
            "status": protocol["status"],
            "color": protocol["color"],
            "summary": protocol["summary"],
            "action_protocol": protocol["actions"]
        }

    def simulate_city_wards(
        self,
        rainfall_24h: float = 75.0,
        rainfall_intensity: float = 18.0,
        water_level_factor: float = 1.0,
        siltation_factor: float = 1.0,
        reservoir_discharge: float = 2000.0
    ) -> List[Dict[str, Any]]:
        """
        Simulates city-wide conditions across all known wards under custom weather parameters.
        Returns GeoJSON-ready feature list for map dashboard.
        """
        results = []
        for w in self.wards:
            # Ward-specific parameters derived from geography and user sliders
            siltation = min(95.0, max(5.0, 35.0 * siltation_factor))
            drain_capacity = max(10.0, 100.0 - siltation)
            elev_accum = max(0.0, 18.0 - w["elevation_base"]) * 0.18
            water_level = max(0.4, (elev_accum + (rainfall_24h / 45.0) * (1.2 - min(0.8, w["elevation_base"] / 25.0))) * water_level_factor)

            record = {
                "latitude": w["lat"],
                "longitude": w["lon"],
                "zone": w["zone"],
                "rainfall_24h_mm": rainfall_24h,
                "rainfall_intensity_mm_hr": rainfall_intensity,
                "water_level_m": round(water_level, 2),
                "reservoir_discharge_cusecs": reservoir_discharge,
                "elevation_m": w["elevation_base"],
                "slope_degrees": w["slope_base"],
                "drain_density_km_sqkm": w["drain_density_base"],
                "siltation_level_pct": siltation,
                "drain_capacity_pct": drain_capacity,
                "land_use": w["land_use"],
                "past_flood_frequency": w["hist_freq"],
                "historical_max_depth_cm": w["hist_max_depth"]
            }

            pred = self.predict_location(record)

            results.append({
                "ward_id": w["ward_id"],
                "ward_name": w["ward_name"],
                "zone": w["zone"],
                "latitude": w["lat"],
                "longitude": w["lon"],
                "elevation_m": w["elevation_base"],
                "land_use": w["land_use"],
                "inputs": record,
                "prediction": pred
            })

        return results

    def compare_with_historical_events(self, current_rainfall: float, current_water_level: float) -> List[Dict[str, Any]]:
        """Compares current weather scenario with historical disaster benchmarks."""
        comparisons = []
        for event in HISTORICAL_BENCHMARKS:
            diff_rain = current_rainfall - event["rainfall_24h_mm"]
            diff_water = current_water_level - event["water_level_m"]

            if abs(diff_rain) < 25:
                similarity = "Very High (Matching Scenario)"
            elif abs(diff_rain) < 60:
                similarity = "Moderate"
            else:
                similarity = "Substantially Different"

            comparisons.append({
                "event": event["name"],
                "year": event["year"],
                "historical_rainfall_mm": event["rainfall_24h_mm"],
                "historical_water_level_m": event["water_level_m"],
                "rainfall_variance_mm": round(diff_rain, 1),
                "description": event["description"],
                "match_assessment": similarity
            })
        return comparisons


if __name__ == "__main__":
    engine = FloodPreparednessEngine()
    print("Testing single location prediction...")
    test_loc = {
        "rainfall_24h_mm": 195.0,
        "rainfall_intensity_mm_hr": 48.0,
        "water_level_m": 5.2,
        "reservoir_discharge_cusecs": 8000,
        "elevation_m": 3.2,
        "slope_degrees": 0.4,
        "drain_density_km_sqkm": 2.1,
        "siltation_level_pct": 65.0,
        "drain_capacity_pct": 35.0,
        "land_use": "High_Density_Residential",
        "past_flood_frequency": 12,
        "historical_max_depth_cm": 140
    }
    result = engine.predict_location(test_loc)
    print("\nResult:")
    print(f"Status: {result['status']}")
    print(f"Impact: {result['impact_label']} (Confidence: {result['confidence']*100:.1f}%)")
    print(f"Summary: {result['summary']}")
    print("Actions:")
    for a in result["action_protocol"]:
        print(f" - {a}")
