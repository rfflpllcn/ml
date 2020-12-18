import numpy as np
import pandas as pd


def get_3_barriers(prices, daily_volatility, upper_lower_multipliers, t_final):
    # create a container
    barriers = pd.DataFrame(columns=['days_passed',
                                     'price', 'vert_barrier', \
                                     'top_barrier', 'bottom_barrier'], \
                            index=daily_volatility.index)
    for day, vol in daily_volatility.iteritems():
        days_passed = len(daily_volatility.loc \
                              [daily_volatility.index[0]: day])

        # set the vertical barrier
        if (days_passed + t_final < len(daily_volatility.index) and t_final != 0):
            vert_barrier = daily_volatility.index[days_passed + t_final]
        else:
            vert_barrier = np.nan

        # set the top barrier
        if upper_lower_multipliers[0] > 0:
            top_barrier = prices.loc[day] + prices.loc[day] * upper_lower_multipliers[0] * vol
        else:
            # set it to NaNs
            top_barrier = pd.Series(index=prices.index)

        # set the bottom barrier
        if upper_lower_multipliers[1] > 0:
            bottom_barrier = prices.loc[day] - prices.loc[day] * upper_lower_multipliers[1] * vol
        else:
            # set it to NaNs
            bottom_barrier = pd.Series(index=prices.index)

        barriers.loc[day, ['days_passed', 'price', 'vert_barrier', 'top_barrier', 'bottom_barrier']] = \
            days_passed, prices.loc[day], vert_barrier, top_barrier, bottom_barrier

    return barriers


def get_labels(barriers):
    """
    start: first day of the window
    end:last day of the window
    price_initial: first day stock price
    price_final:last day stock price
    top_barrier: profit taking limit
    bottom_barrier:stop loss limt
    condition_pt:top_barrier touching conditon
    condition_sl:bottom_barrier touching conditon
    """
    barriers['out'] = None
    for i in range(len(barriers.index)):
        start = barriers.index[i]
        end = barriers.vert_barrier[i]
        if pd.notna(end):
            # assign the initial and final price
            price_initial = barriers.price[start]
            price_final = barriers.price[end]

            # assign the top and bottom barriers
            top_barrier = barriers.top_barrier[i]
            bottom_barrier = barriers.bottom_barrier[i]

            # set the profit taking and stop loss conditons
            condition_pt = (barriers.price[start: end] >= top_barrier).any()
            condition_sl = (barriers.price[start: end] <= bottom_barrier).any()

            # assign the labels
            if condition_pt:
                barriers['out'][i] = 1
            elif condition_sl:
                barriers['out'][i] = -1
            else:
                barriers['out'][i] = max(
                    [(price_final - price_initial) /
                     (top_barrier - price_initial), \
                     (price_final - price_initial) / \
                     (price_initial - bottom_barrier)], \
                    key=abs)
    return barriers
