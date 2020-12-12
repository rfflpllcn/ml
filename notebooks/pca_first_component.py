import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

random.seed(1)

with open('../data/sp500.json', 'r') as f:
    prices = pd.read_json(f)

T, N = prices.shape

# our portfolio allocation
aa_w = np.array([random.random() for i in range(N)])
aa_w = aa_w / aa_w.sum()

# our portfolio values
prices_weighted = prices.apply(lambda x: x*aa_w, axis=1)
portfolio = prices_weighted.sum(axis=1)

# calc log returns
rs = prices_weighted.apply(np.log).diff(1)

# first pca component
pca = PCA(1).fit(rs.fillna(0))
pc1 = pd.Series(index=rs.columns, data=pca.components_[0])
weights_pca1 = abs(pc1) / sum(abs(pc1))

# the first pca component portfolio
portfolio_pca1 = (weights_pca1 * rs).sum(1)
# portfolio_pca1.cumsum().apply(np.exp).plot()

# compare our original portfolio with its pca 1 reduction
rs_df = pd.concat([portfolio_pca1, portfolio.apply(np.log).diff(1)], 1)
rs_df.columns = ["PCA Portfolio", "S&P500"]

rs_df.dropna().cumsum().apply(np.exp).plot(subplots=True, figsize=(10,6), grid=True, linewidth=3);
plt.tight_layout()
plt.show()
# plt.savefig('tmp.png')

# stocks with most/least negative pca weights
fig, ax = plt.subplots(2,1, figsize=(10,6))
pc1.nsmallest(10).plot.bar(ax=ax[0], color='green', grid=True, title='Stocks with Most Negative PCA Weights')
pc1.nlargest(10).plot.bar(ax=ax[1], color='blue', grid=True, title='Stocks with Least Negative PCA Weights')
plt.tight_layout()
plt.show()
# plt.savefig('tmp.png')


# plot mst/least negative pca weights portfolios vs original
# ws = [-1,]*10+[1,]*10
# myrs = (rs[list(pc1.nsmallest(10).index)+list(pc1.nlargest(10).index)]*ws).mean(1)
myrs = rs[pc1.nlargest(10).index].mean(1)
myrs.cumsum().apply(np.exp).plot(figsize=(15,5), grid=True, linewidth=3, title='PCA Portfolio vs. S&P500')
portfolio['2020':].apply(np.log).diff(1).cumsum().apply(np.exp).plot(figsize=(10,6), grid=True, linewidth=3)
plt.legend(['PCA Selection', 'S&P500'])
plt.tight_layout()
plt.show()
# plt.savefig('tmp.png')

