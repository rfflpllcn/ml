import pandas as pd


def cusum_filter_symmetric(gRaw, h):
    """
    A bar t is sampled if and only if S_t ≥ h, at which point S_t is
    reset. entation of the symmetric CUSUM filter, where
    E t−1 [y t ] = y t−1 .

    :param gRaw: the raw time series we wish to filter
    :param h: the threshold
    :return:
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
    if sNeg < -h:
        sNeg = 0;
        tEvents.append(i)
    elif sPos > h:
        sPos = 0;
        tEvents.append(i)
    return pd.DatetimeIndex(tEvents)
