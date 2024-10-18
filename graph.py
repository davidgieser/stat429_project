import matplotlib.pyplot as plt
import pandas as pd

def plot_funding_rate(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
    df = df.set_index('timestamp')

    plt.plot(df.index, df['funding_rate'] * 10_000)
    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")
    plt.ylabel("Funding Rate (bips)")
    plt.title("Funding Rate vs. Time")
    plt.show()