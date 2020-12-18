"""
Working with denoised and detoned covariance matrices renders substantial
benefits. Those benefits result from the mathematical properties of those treated
matrices, and can be evaluated through Monte Carlo experiments

Code computes the true minimum variance portfolio, derived from the true covariance matrix.
Using those allocations as benchmark, it then computes the root-mean-square errors (RMSE) across all weights, with and
without denoising.

the denoised minimum variance portfolio incurs only 40.15% of the RMSE incurred by the minimum variance portfolio without
denoising. That is a 59.85% reduction in RMSE from denoising alone

"""

import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization import RiskEstimators
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf

from src.risk import corr2cov


def optPort(cov, mu=None):
    # derive the minimum variance portfolio
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None: mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w


def formTrueMatrix(nBlocks, bSize, bCorr):
    """
    forms a vector
    of means and a covariance matrix out of ten blocks of size fifty each, where off-
    diagonal elements within each block have a correlation of 0.5. This covariance
    matrix is a stylized representation of a true (nonempirical) detoned correlation
    matrix of the S&P 500, where each block is associated with an economic sector.

    Without loss of generality, the variances are drawn from a uniform distribution
    bounded between 5% and 20%, and the vector of means is drawn from a Normal
    distribution with mean and standard deviation equal to the standard deviation
    from the covariance matrix. This is consistent with the notion that in an efficient
    market all securities have the same expected Sharpe ratio

    :param nBlocks:
    :param bSize:
    :param bCorr:
    :return:
    """
    def formBlockMatrix(nBlocks, bSize, bCorr):
        block = np.ones((bSize, bSize)) * bCorr
        block[range(bSize), range(bSize)] = 1
        corr = block_diag(*([block] * nBlocks))
        return corr

    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(.05, .2, corr0.shape[0])
    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)
    return mu0, cov0


def simCovMu(mu0, cov0, nObs, shrink=False):
    """
    GENERATING THE EMPIRICAL COVARIANCE MATRIX
    uses the true (nonempirical) covariance matrix cov0 to draw a
    random matrix X of size TxN, and it derives the associated empirical covariance
    matrix and vector of means.

    :param mu0: the true expected returns
    :param cov0: the true covariance matrix
    :param nObs: the value of T
    :param shrink: When shrink=True, the function performs a Ledoit–Wolf shrinkage of the empirical covariance matrix.
    :return:
    """

    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=nObs)
    mu1 = x.mean(axis=0).reshape(-1, 1)
    if shrink:
        cov1 = LedoitWolf().fit(x).covariance_
    else:
        cov1 = np.cov(x, rowvar=0)
    return mu1, cov1


if __name__ == "__main__":

    nBlocks, bSize, bCorr = 10, 50, .5
    kde_bwidth = 0.25
    np.random.seed(0)
    mu0, cov0 = formTrueMatrix(nBlocks, bSize, bCorr)

    risk_estimators = RiskEstimators()

    nObs, nTrials, bWidth, shrink, minVarPortf = 1000, 1000, .01, False, True
    w1 = pd.DataFrame(columns=range(cov0.shape[0]),
                      index=range(nTrials), dtype=float)
    w1_d = w1.copy(deep=True)
    for i in range(nTrials):
        mu1, cov1 = simCovMu(mu0, cov0, nObs, shrink=shrink)
        if minVarPortf: mu1 = None

        tn_relation = nObs * 1. / cov1.shape[1]
        cov1_d = risk_estimators.denoise_covariance(cov1, tn_relation, kde_bwidth)

        w1.loc[i] = optPort(cov1, mu1).flatten()
        w1_d.loc[i] = optPort(cov1_d, mu1).flatten()

    w0 = optPort(cov0, None if minVarPortf else mu0)
    w0 = np.repeat(w0.T, w1.shape[0], axis=0)
    rmsd = np.mean((w1 - w0).values.flatten() ** 2) ** .5  # RMSE
    rmsd_d = np.mean((w1_d - w0).values.flatten() ** 2) ** .5  # RMSE
    print(rmsd, rmsd_d)
