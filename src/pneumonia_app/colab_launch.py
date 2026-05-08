"""Run Streamlit + ngrok inside Colab."""
import os
import subprocess
import time

NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")

if not NGROK_TOKEN:
    raise SystemExit("Set NGROK_TOKEN env var. Get a free token from https://dashboard.ngrok.com/")

# Install
subprocess.run(["pip", "install", "-q", "streamlit", "pyngrok"], check=True)

from pyngrok import ngrok, conf

conf.get_default().auth_token = NGROK_TOKEN
ngrok.kill()  # close stale tunnels

PORT = 8501
APP_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(APP_DIR, "app.py")

proc = subprocess.Popen([
    "streamlit", "run", APP_PATH,
    "--server.port", str(PORT),
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false",
])

time.sleep(5)
public = ngrok.connect(PORT, "http").public_url

print("\n" + "=" * 50)
print(f"  🌐  PUBLIC URL:  {public}")
print("=" * 50 + "\n")

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    ngrok.kill()