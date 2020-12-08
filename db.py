import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import bs4 as bs
import requests


# Scrap sp500 tickers
def save_sp500_tickers():

    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'html')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        if not '.' in ticker:
            tickers.append(ticker.replace('\n',''))

    return tickers

tickers = save_sp500_tickers()
prices = yf.download(tickers, start='2020-01-01', end='2020-12-07')['Adj Close']

with open('./data/sp500.json', 'w') as f:
    prices.to_json(f, date_format='iso')
print(prices)

