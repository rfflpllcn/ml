import numpy as np


def getRndCov(nCols, nFacts):
    w = np.random.normal(size=(nCols, nFacts))
    cov = np.dot(w, w.T)  # random cov matrix, however not full rank
    cov += np.diag(np.random.uniform(size=nCols))  # full rank cov
    return cov


def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr


def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov


def denoisedCorr(eVal, eVec, nFacts):
    # Remove noise from corr by fixing random eigenvalues
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)
    eVal_ = np.diag(eVal_)
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)
    corr1 = cov2corr(corr1)
    return corr1


def get_daily_volatility(close, span0=20):
    """
    computes daily volatility for daily close prices
    Note: in case of intraday prices, use: vol = mlfinlab.util.get_daily_vol(close=stock_prices[stock_name], lookback=50)

    :param close:
    :param span0:
    :return:
    """
    # simple percentage returns
    df0 = close.pct_change()
    # 20 days, a month EWM's std as boundary
    df0 = df0.ewm(span=span0).std()
    df0.dropna(inplace=True)
    return df0
