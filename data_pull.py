import os
from datetime import datetime, timedelta
from tardis_dev import datasets

TARDIS_TYPE = 'derivative_ticker'
QUOTE = 'BTC'
BASE = 'USDT'
START = '2023-01-01'
END = '2023-02-01'
EXCHANGE = 'binance-futures'

def get_api_key():
    api_key = os.environ.get('TARDIS_API_KEY')
    if not api_key:
        raise ValueError("TARDIS_API_KEY not found in environment variables")
    return api_key

def download_tardis_data(exchange, start_date, end_date, tardis_sym, instrument):
    api_key = get_api_key()

    cur_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    while cur_date < end_date:
        cur_str = cur_date.strftime("%Y-%m-%d")
        tom_str = (cur_date + timedelta(days=1)).strftime("%Y-%m-%d")

        filename = f"{exchange}_{tardis_sym}_{instrument}_{cur_str}.csv.gz"
        if os.path.exists(filename):
            cur_date += timedelta(days=1)
            continue

        datasets.download(
            exchange=exchange,
            data_types=[
                "derivative_ticker",
            ],
            from_date=cur_str,
            to_date=tom_str,
            symbols=["btcusdt"],
            api_key=api_key,
        )
        
        print(f"[DOWNLOADED] {cur_str}")
        cur_date += timedelta(days=1)

if __name__ == "__main__":
    download_tardis_data(EXCHANGE, START, END, TARDIS_TYPE, f"{QUOTE.lower()}{BASE.lower()}")