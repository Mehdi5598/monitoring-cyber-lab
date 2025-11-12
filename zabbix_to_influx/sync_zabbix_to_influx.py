#!/usr/bin/env python3
import os
import time
import requests
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WriteOptions

# === Configuration ===
ZABBIX_URL = os.getenv("ZABBIX_URL", "http://zabbix-web:8080/api_jsonrpc.php")
ZABBIX_TOKEN = os.getenv("ZABBIX_TOKEN")
ZABBIX_USER = os.getenv("ZABBIX_USER", "Admin")
ZABBIX_PASSWORD = os.getenv("ZABBIX_PASSWORD", "zabbix")
INFLUX_URL = os.getenv("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG", "myorg")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "metrics")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 15))

# === Zabbix login / auth ===
def zabbix_login():
    """Authenticate with Zabbix and return a valid auth mechanism."""
    if ZABBIX_TOKEN:
        print("[INFO] Using Zabbix API token authentication (Bearer mode)")
        return {"type": "token", "value": ZABBIX_TOKEN}

    print("[INFO] Using username/password authentication")
    payload = {
        "jsonrpc": "2.0",
        "method": "user.login",
        "params": {
            "username": ZABBIX_USER,
            "password": ZABBIX_PASSWORD
        },
        "id": 1
    }
    r = requests.post(ZABBIX_URL, json=payload, timeout=10)
    r.raise_for_status()
    result = r.json()
    if "result" not in result:
        raise Exception(f"Login failed: {result}")
    print("[INFO] Authenticated successfully with Zabbix API")
    return {"type": "session", "value": result["result"]}

# === Get monitored items ===
def get_zabbix_items(auth):
    headers = {"Content-Type": "application/json-rpc"}
    payload = {
        "jsonrpc": "2.0",
        "method": "item.get",
        "params": {
            "output": ["itemid", "name", "lastvalue", "hostid", "key_"],
            "monitored": True
        },
        "id": 2
    }

    # Adapt auth for Zabbix version
    if auth["type"] == "session":
        payload["auth"] = auth["value"]
    else:
        headers["Authorization"] = f"Bearer {auth['value']}"

    r = requests.post(ZABBIX_URL, headers=headers, json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()

    if "result" not in data:
        print(f"[WARN] Failed to fetch items: {data}")
        return []
    return data["result"]

# === Push data to InfluxDB ===
def write_to_influx(items):
    if not items:
        print("[WARN] No items to write to InfluxDB")
        return

    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        write_api = client.write_api(write_options=WriteOptions(batch_size=500, flush_interval=10_000))
        points = []
        for item in items:
            try:
                value = float(item["lastvalue"])
            except (ValueError, KeyError):
                continue

            point = (
                Point("zabbix_metric")
                .tag("hostid", item.get("hostid"))
                .tag("key", item.get("key_"))
                .field("value", value)
                .time(datetime.utcnow())
            )
            points.append(point)

        if points:
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)
            print(f"[INFO] Wrote {len(points)} metrics to InfluxDB ({datetime.utcnow().isoformat()} UTC)")
        else:
            print("[INFO] No valid metrics to send")

# === Main loop ===
def main():
    print("[START] Zabbix → InfluxDB Sync Service")
    auth = zabbix_login()
    while True:
        try:
            items = get_zabbix_items(auth)
            write_to_influx(items)
        except Exception as e:
            print(f"[ERROR] {e}")
            auth = zabbix_login()
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()

