import subprocess
import sys
from datetime import datetime, timezone
import pandas_market_calendars as mcal

PYTHON = sys.executable
PIPELINE = "scripts/run_snapshot_pipeline.py"

def is_nyse_open_now():
    nyse = mcal.get_calendar("NYSE")
    now_utc = datetime.now(timezone.utc)

    schedule = nyse.schedule(
        start_date=now_utc.date().isoformat(),
        end_date=now_utc.date().isoformat()
    )

    if schedule.empty:
        return False

    market_open = schedule.iloc[0]["market_open"].to_pydatetime()
    market_close = schedule.iloc[0]["market_close"].to_pydatetime()

    return market_open <= now_utc <= market_close

def main():
    if not is_nyse_open_now():
        print("Market closed -> skipping.")
        return

    print("Market open -> running snapshot pipeline.")
    subprocess.run([PYTHON, PIPELINE], check=True)

if __name__ == "__main__":
    main()
