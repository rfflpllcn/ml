import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.family'] = 'serif'


def get_data(symbols, begin_date=None, end_date=None):
    df = yf.download(symbols, start=begin_date,
                     auto_adjust=True,  # only download adjusted data
                     end=end_date)
    # my convention: always lowercase
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    return df


if __name__ == "__main__":
    Apple_stock = get_data('AAPL', '2000-01-01', '2020-12-12')
    price = Apple_stock['close']

    price.plot()
    plt.show()