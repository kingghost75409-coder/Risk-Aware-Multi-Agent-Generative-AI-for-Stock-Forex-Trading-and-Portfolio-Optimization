#!/usr/bin/env python3
"""Entry point: `python run.py` starts the Flask API + dashboard.

Visit http://localhost:5000 once running.
"""
from app.api.app import app
from app.config import settings

if __name__ == "__main__":
    print(f"\nStock AI Trader — starting on http://localhost:{settings.flask_port}\n"
          f"Mode: PAPER TRADING / RESEARCH SIMULATION — no real orders are ever placed.\n")
    app.run(host=settings.flask_host, port=settings.flask_port,
            debug=settings.flask_debug, use_reloader=False, threaded=True)
