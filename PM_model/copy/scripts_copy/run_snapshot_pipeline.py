import subprocess
import sys
import os

# Use the same Python that runs this script (best for conda/env reliability)
PYTHON = sys.executable

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)

    # 1) Polymarket snapshot
    subprocess.run([PYTHON, "scripts/fetch_pm_data.py"], check=True)

    # 2) Options snapshot + RN probabilities
    subprocess.run([PYTHON, "scripts/fetch_yf_data.py"], check=True)

    # 3) Build alpha ΔP CSV
    subprocess.run([PYTHON, "scripts/deltaP_calculator.py"], check=True)

if __name__ == "__main__":
    main()
