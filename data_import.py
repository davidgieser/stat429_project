import pandas as pd
import gzip
import os
from datetime import datetime, timedelta

def normalize_tardis_data(exchange, tardis_type, start_date, end_date, instrument, parent_dir):
    # Convert start and end dates to datetime objects
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Updated file pattern to include 'normalized_datasets' folder
    file_pattern = f"{parent_dir}/normalized_datasets/{exchange}_{tardis_type}_{{date}}_{instrument}.csv.gz"

    dfs = []
    
    # Generate a list of dates within the given range
    date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]
    
    for date in date_list:
        date_str = date.strftime('%Y-%m-%d')
        filename = file_pattern.format(date=date_str)
        
        if os.path.exists(filename):
            with gzip.open(filename, 'rt') as f:
                df = pd.read_csv(f)

            # Convert 'timestamp' to datetime and set it as the index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
            df = df.set_index('timestamp')

            # Resample to hourly frequency and take the last available entry
            df_hourly = df.resample('h').last()

            # Check if there are 24 data points; if not, skip the date
            if len(df_hourly) == 24:
                dfs.append(df_hourly)
                print(f"[PROCESSED] {date_str}")
            else:
                print(f"Skipping incomplete data for date: {date_str}")
        else:
            print(f"File not found: {date_str}")

    # Combine all DataFrames if there are any, or return an empty DataFrame
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        print("No valid data found in the given date range.")
        return pd.DataFrame()