"""
The trading strategy is as follows:
Say at time t, the total amount in the portfolio is x.
At time t, each of the 15 stocks recommends we either buy or short that stock.
For each stock, we wager x/30 units in buying/shorting.
We calculate the next days value of the portfolio as follows:

x_{t+1} = x_t / 2 + \sum((-1)^(short) x_t / 30 * (1 + pct change in stock i))

where short = 1 if we choose to short stock i and 0 otherwise

That is, we carry over to the next day the half we didn't invest.
For each of the stocks, we close our position and sum up the new values from each of our positions.
"""

"""
How to accomplish this:

For a given model, collect ALL the trading predictions on validation sets and ALL the price histories on validation sets.

Combine them into a single data frame. Iterate through and update x. Report x_T / x_0
"""

# Inputs:
# trade_dict -- dictionary of numpy arrays containing trading instructions (1:buy, 0:do nothing, -1:sell) on the test set
# test_dict -- dictionary of numpy arrays containing the opening price values on the test set
# Output:
# (Simple) Percent growth of the portfolio over that period.
def get_performance(trade_dict, test_dict):
    n = len(trade_dict["AAPL"])
    x_t = [1] * n
    for i in range(1 , n):
        x_t[i] = x_t[i-1] / 2
        for tick in trade_dict:
            x_t[i] += (x_t[i-1] / 30) * (1 + trade_dict[tick][i-1] * (test_dict[tick][i] - test_dict[tick][i-1]) / test_dict[tick][i-1])
    return x_t[-1]

