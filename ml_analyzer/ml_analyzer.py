#!/usr/bin/env python3
import os
import time
from datetime import datetime, timedelta
import pandas as pd
from influxdb_client import InfluxDBClient, Point
from sklearn.ensemble import IsolationForest
import requests
import traceback

# Config
INFLUX_URL = os.environ.get("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "metrics")
ORG = os.environ.get("INFLUX_INIT_ORG", "myorg")
raw_window = os.environ.get("ML_WINDOW_MINUTES", "15")
try:
    ML_WINDOW_MINUTES = int(raw_window)
except Exception as e:
    print(f"⚠️ Invalid ML_WINDOW_MINUTES='{raw_window}': {e}. Using 15.")
    ML_WINDOW_MINUTES = 15
ALERT_WEBHOOK = os.environ.get("ALERT_WEBHOOK_URL", "")

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=ORG)
query_api = client.query_api()
write_api = client.write_api()


def query_metrics(window_minutes):
    now = datetime.utcnow()
    start = now - timedelta(minutes=window_minutes)

    flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {start.isoformat()}Z, stop: {now.isoformat()}Z)
  |> filter(fn: (r) => r._measurement == "zabbix_metric")
  |> filter(fn: (r) => r._field == "value")
  |> filter(fn: (r) => exists r.key and r.key != "")
  |> map(fn: (r) => ({{
      _time: r._time,
      _value: r._value,
      _measurement: r.key
    }}))
  |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
  |> drop(columns: ["result", "table"])
'''

    print(f"Querying Influx for last {window_minutes} min")
    try:
        tables = query_api.query(flux)
    except Exception as e:
        print("Influx query failed:", e)
        return None

    rows = []
    for table in tables:
        for record in table.records:
            try:
                ts = record.get_time()
                measurement = record.values.get("_measurement", "unknown")
                val = record.values.get("_value")
                if val is None:
                    continue
                val = float(val)
                rows.append({"_time": ts, "_measurement": measurement, "value": val})
            except (ValueError, TypeError, KeyError):
                continue

    if not rows:
        print("⚠️ No numeric records found.")
        return None

    df = pd.DataFrame(rows)
    try:
        df_pivot = df.pivot_table(index="_time", columns="_measurement", values="value", aggfunc="mean")
        df_pivot = df_pivot.sort_index()
    except Exception as e:
        print("Pivot failed:", e)
        return None

    numeric_cols = df_pivot.select_dtypes(include=["number"]).columns
    if len(numeric_cols) < 2:
        print(f"⚠️ Only {len(numeric_cols)} numeric metrics. Need ≥2 for ML.")
        return None

    df_clean = df_pivot[numeric_cols].fillna(method="ffill").fillna(method="bfill").fillna(0)
    print(f"Got {df_clean.shape[0]} timestamps, {df_clean.shape[1]} metrics.")
    return df_clean


def run_once():
    try:
        df = query_metrics(ML_WINDOW_MINUTES)
        if df is None or df.shape[0] < 5:
            print("Not enough data for ML (rows):", df.shape if df is not None else "None")
            return

        model = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
        preds = model.fit_predict(df.values)
        anomaly_idx = [i for i, p in enumerate(preds) if p == -1]

        print(f"Window: {df.shape}, anomalies found: {len(anomaly_idx)}")

        for idx in anomaly_idx:
            ts = df.index[idx]
            row = df.iloc[idx].to_dict()

            p = Point("anomalies").time(ts)
            for k, v in row.items():
                try:
                    field_name = k.replace(".", "_").replace("[", "_").replace("]", "_")
                    p = p.field(field_name, float(v))
                except (ValueError, TypeError):
                    pass
            try:
                write_api.write(bucket=INFLUX_BUCKET, org=ORG, record=p)
                print(f" Wrote anomaly at {ts}")
            except Exception as e:
                print("Failed to write anomaly:", e)

            if ALERT_WEBHOOK:
                try:
                    requests.post(ALERT_WEBHOOK, json={"time": ts.isoformat(), "metrics": row}, timeout=10)
                except Exception as e:
                    print("Webhook failed:", e)

    except Exception:
        print(" ML crashed:", traceback.format_exc())


if __name__ == "__main__":
    print(f" ML analyzer started | window: {ML_WINDOW_MINUTES} min")
    while True:
        run_once()
        time.sleep(30)
