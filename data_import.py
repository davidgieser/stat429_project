import pandas as pd
import gzip
import os
from datetime import datetime, timedelta

def normalize_tardis_data(exchange, tardis_type, start_date, end_date, instrument, parent_dir):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    file_pattern = f"{parent_dir}/datasets/{exchange}_{tardis_type}_{{date}}_{instrument}.csv.gz"
    
    dfs = []
    
    date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]
    for date in date_list:
        date_str = date.strftime('%Y-%m-%d')
        filename = file_pattern.format(date=date_str)
        
        if os.path.exists(filename):
            with gzip.open(filename, 'rt') as f:
                df = pd.read_csv(f)

            # This handles the normalization by taking the last entry of every 1 hour bucket.
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
            df = df.set_index('timestamp')
            df_hourly = df.resample('h').last()
            dfs.append(df_hourly)

            assert(len(df_hourly == 24))
        else:
            raise ValueError(f'Invalid file at date: {date_str}')

        print(f"[PROCESSED] {date_str}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()