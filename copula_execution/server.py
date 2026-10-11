"""
FastAPI Backend and Static Server for the Copula Coherence Scanner & Dashboard.
Runs locally on http://localhost:8080 and serves the live execution dashboard.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser


def run_dashboard_server(port: int = 8080):
    dashboard_path = Path(r"C:\Users\Belia\.gemini\antigravity\brain\e701cb00-6086-4c6f-a515-0075ca8c67df\copula_dashboard.html")
    if not dashboard_path.exists():
        print(f"Error: Dashboard HTML not found at {dashboard_path}")
        return

    print("=" * 60)
    print(f" COPULA DASHBOARD SERVER RUNNING")
    print(f" URL: http://localhost:{port}")
    print(f" Source: {dashboard_path}")
    print("=" * 60)

    class CustomHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in ["/", "/index.html", "/dashboard"]:
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                with open(dashboard_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                super().do_GET()

    server = HTTPServer(("localhost", port), CustomHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    run_dashboard_server()
