from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    # Запускаем страницу через Streamlit.
    app = Path(__file__).resolve().parent / "profile_app.py"
    print("Запускаю страницу...", flush=True)
    try:
        result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(app),
            "--server.address", "127.0.0.1", "--server.headless", "false",
            "--server.showEmailPrompt", "false", "--client.toolbarMode", "minimal",
            "--browser.gatherUsageStats", "false"], cwd=app.parent)
        raise SystemExit(result.returncode)
    except KeyboardInterrupt:
        pass
