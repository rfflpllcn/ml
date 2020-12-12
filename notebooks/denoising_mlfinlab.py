"""
De-noising covariance matrix
The main idea behind de-noising the covariance matrix is to eliminate the eigenvalues of the covariance matrix that are representing noise and not useful information.

This is done by determining the maximum theoretical value of the eigenvalue of such matrix as a threshold and then setting all the calculated eigenvalues above the threshold to the same value.

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

# Leaving only 5 stocks in the dataset
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
cov_matrix_denoised = risk_estimators.denoise_covariance(cov_matrix, tn_relation, kde_bwidth)
cov_matrix_denoised = pd.DataFrame(cov_matrix_denoised, index=cov_matrix.index, columns=cov_matrix.columns)

# plot eigenvalues denoised vs simple
evals_denoised, _ = risk_estimators._get_pca(risk_estimators.cov_to_corr(cov_matrix_denoised))
evals, _ = risk_estimators._get_pca(risk_estimators.cov_to_corr(cov_matrix))

plt.figure()
plt.plot(range(len(evals)), np.diag(evals), 'g^', range(len(evals)), np.diag(evals_denoised), 'g-')
plt.title('Eigenvalues, denoised vs simple correlations')
plt.yscale('log')
plt.ylabel('Eigenvalue (log-scale)')
plt.xlabel('Eigenvalue number')
plt.legend(['simple', 'denoised'])
plt.show()


