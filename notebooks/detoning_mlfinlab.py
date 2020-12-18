"""
De-toning covariance matrix
By removing the market component, we allow a greater portion of the correlation to be explained by components that
affect specific subsets of the securities.
It is similar to removing a loud tone that prevents us from hearing other sounds.

The detoned correlation matrix is singular, as a result of eliminating (at least)
one eigenvector. This is not a problem for clustering applications, as most
approaches do not require the invertibility of the correlation matrix.

"""
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlfinlab.portfolio_optimization.estimators import RiskEstimators, ReturnsEstimators

from definitions import ROOT_DIR

# Getting the data
stock_prices = pd.read_csv(join(ROOT_DIR, 'data/stock_prices.csv'), parse_dates=True, index_col='Date')
stock_prices = stock_prices.dropna(axis=1)
stock_prices.head()

# Leaving only x stocks in the dataset
stock_prices = stock_prices.iloc[:, :]
stock_prices.head()

# estimators
returns_estimation = ReturnsEstimators()
risk_estimators = RiskEstimators()

# stock returns
stock_returns = returns_estimation.calculate_returns(stock_prices)

# the simple covariance matrix
cov_matrix = stock_returns.cov()

# the De-noised Сovariance matrix
tn_relation = stock_prices.shape[0] / stock_prices.shape[1]
kde_bwidth = 0.25

cov_matrix_detoned = risk_estimators.denoise_covariance(cov_matrix, tn_relation, kde_bwidth, detone=True)
cov_matrix_detoned = pd.DataFrame(cov_matrix_detoned, index=cov_matrix.index, columns=cov_matrix.columns)

# plot eigenvalues denoised vs simple
evals_detoned, _ = risk_estimators._get_pca(risk_estimators.cov_to_corr(cov_matrix_detoned))
evals, _ = risk_estimators._get_pca(risk_estimators.cov_to_corr(cov_matrix))

plt.figure()
plt.plot(range(len(evals)), np.diag(evals), 'g^', range(len(evals)), np.diag(evals_detoned), 'g-')
plt.title('Eigenvalues, detoned vs simple correlations')
plt.yscale('log')
plt.ylabel('Eigenvalue (log-scale)')
plt.xlabel('Eigenvalue number')
plt.legend(['simple', 'detoned'])
plt.show()
