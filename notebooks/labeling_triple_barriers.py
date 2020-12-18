"""

"""

from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlfinlab as ml
from mlfinlab.portfolio_optimization.estimators import RiskEstimators, ReturnsEstimators

from definitions import ROOT_DIR
from src.labeling import get_3_barriers, get_labels
from src.risk import get_daily_volatility
from src.download import get_data


# Getting the data

# asset_name = 'SPY'
# asset = get_data(asset_name, '2000-01-01', '2020-12-12')
asset = get_data('AAPL', '2000-01-01', '2010-12-31')
price = asset['close']

# triple barrier setup
# set the boundary of barriers, based on 20 days EWM
daily_volatility = get_daily_volatility(price)
# how many days we hold the stock which set the vertical barrier
t_final = 10
# the up and low boundary multipliers
upper_lower_multipliers = [2, 2]
# allign the index
prices = price[daily_volatility.index]

# vol.plot()
# plt.show()

barriers = get_3_barriers(prices, daily_volatility, upper_lower_multipliers, t_final)
barriers = get_labels(barriers)

plt.plot(barriers.out,'bo')
plt.show()

# count how many profit taking and stop loss limit were triggered
print(barriers.out.value_counts())

# pick a random date and show it on a graph.
fig,ax = plt.subplots()
ax.set(title='Apple stock price',
       xlabel='date', ylabel='price')
ax.plot(barriers.price[100: 200])
start = barriers.index[120]
end = barriers.vert_barrier[120]
upper_barrier = barriers.top_barrier[120]
lower_barrier = barriers.bottom_barrier[120]
ax.plot([start, end], [upper_barrier, upper_barrier], 'r--');
ax.plot([start, end], [lower_barrier, lower_barrier], 'r--');
ax.plot([start, end], [(lower_barrier + upper_barrier)*0.5, \
                       (lower_barrier + upper_barrier)*0.5], 'r--');
ax.plot([start, start], [lower_barrier, upper_barrier], 'r-');
ax.plot([end, end], [lower_barrier, upper_barrier], 'r-');
plt.show()