import pandas as pd
import numpy as np

df_rwl = pd.read_csv('rwl_tel_hr_tamil_nadu_sw_gw_27_2026_2030.csv')
df_rain = pd.read_csv('rainfall_tel_hr_tamil_nadu_sw_gw_tn_2026_2030.csv')

df_rwl['datetime'] = pd.to_datetime(df_rwl['Data Acquisition Time'], format='%d-%m-%Y %H:%M', errors='coerce')
df_rain['datetime'] = pd.to_datetime(df_rain['Data Acquisition Time'], format='%d-%m-%Y %H:%M', errors='coerce')

station_pairs = [
    ('Kamuthi Bridge', 'Kamudhi -ARG'),
    ('Nandambakkam CheckDam', 'SGSWRDC Campus (Tharamani site)'),
    ('Parthibanur Regulator', 'Parthibanur Regulator'),
    ('Virudhachalam Anicut', 'Sethiathope Anicut TARG')
]

for rwl_st, rain_st in station_pairs:
    sub_rwl = df_rwl[df_rwl['Station'] == rwl_st][['datetime', 'River Water Level Telemetry Hourly (meter)']]
    sub_rain = df_rain[df_rain['Station'] == rain_st][['datetime', 'Telemetry Hourly Rainfall (mm)']]
    merged = pd.merge(sub_rwl, sub_rain, on='datetime', how='inner')
    print(f"Pair '{rwl_st}' & '{rain_st}': RWL count={len(sub_rwl)}, Rain count={len(sub_rain)}, Aligned={len(merged)}")
