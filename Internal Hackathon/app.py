import os
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime
from flood_engine import FloodPreparednessEngine, HISTORICAL_BENCHMARKS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "flood_action_log.json")

# Initialize ML engine
engine = FloodPreparednessEngine()

def load_action_logs():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_action_log(entry):
    logs = load_action_logs()
    logs.insert(0, entry) # newest first
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)
    return logs


class FloodDashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            dash_path = os.path.join(BASE_DIR, "dashboard.html")
            with open(dash_path, "rb") as f:
                self.wfile.write(f.read())
            return

        elif path.startswith("/static/"):
            # Serve static files (e.g. png images)
            fname = path.replace("/static/", "")
            file_path = os.path.join(BASE_DIR, fname)
            model_file_path = os.path.join(BASE_DIR, "output_models", fname)
            if os.path.exists(file_path):
                self.send_response(200)
                if fname.endswith(".png"):
                    self.send_header("Content-type", "image/png")
                elif fname.endswith(".json"):
                    self.send_header("Content-type", "application/json")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            elif os.path.exists(model_file_path):
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.end_headers()
                with open(model_file_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        elif path == "/api/simulate":
            query = urllib.parse.parse_qs(parsed.query)
            rainfall_24h = float(query.get("rainfall_24h", [75.0])[0])
            rainfall_intensity = float(query.get("rainfall_intensity", [18.0])[0])
            water_level_factor = float(query.get("water_level_factor", [1.0])[0])
            siltation_factor = float(query.get("siltation_factor", [1.0])[0])
            reservoir_discharge = float(query.get("reservoir_discharge", [2000.0])[0])

            wards_data = engine.simulate_city_wards(
                rainfall_24h=rainfall_24h,
                rainfall_intensity=rainfall_intensity,
                water_level_factor=water_level_factor,
                siltation_factor=siltation_factor,
                reservoir_discharge=reservoir_discharge
            )

            # Calculate summary stats
            counts = {"Low": 0, "Moderate": 0, "High": 0, "Severe Inundation": 0}
            for w in wards_data:
                label = w["prediction"]["impact_label"]
                # strip class number prefix if present
                for key in counts.keys():
                    if key in label:
                        counts[key] += 1
                        break

            # Historical comparisons
            hist_comps = engine.compare_with_historical_events(
                current_rainfall=rainfall_24h,
                current_water_level=2.5 * water_level_factor
            )

            response_data = {
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "rainfall_24h_mm": rainfall_24h,
                    "rainfall_intensity_mm_hr": rainfall_intensity,
                    "water_level_factor": water_level_factor,
                    "siltation_factor": siltation_factor,
                    "reservoir_discharge_cusecs": reservoir_discharge
                },
                "summary": counts,
                "wards": wards_data,
                "historical_comparisons": hist_comps
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        elif path == "/api/logs":
            logs = load_action_logs()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(logs).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/log_action":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            entry = json.loads(post_data.decode("utf-8"))
            entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            updated_logs = save_action_log(entry)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "logs": updated_logs}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")


def run_server(port=8050):
    server_address = ("", port)
    httpd = HTTPServer(server_address, FloodDashboardHandler)
    print(f"Urban Flood Platform Server running at: http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
