import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scs
from futu import *

pd.options.mode.chained_assignment = None

quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

date_list = pd.date_range(start="2020-11-04", end="2022-11-04")
date = date_list.strftime("%Y-%m-%d").tolist()
print(date)

contract = 10
commission = 21.2
returns = []
buy = []
sell = []
btime = []
stime = []
side = []
tday = []
for day in date:
    ret, data, page_req_key = quote_ctx.request_history_kline('HK.MHImain', start=day, end=day,
                                                              ktype=KLType.K_3M, max_count=9999999)
    if ret != RET_OK:
        print('Failed to get data：', data)
    data.index = pd.to_datetime(data['time_key']).dt.strftime('%H:%M')
    data = data[(data.index > '09:14') & (data.index <= '16:59')]
    df = data[['time_key', 'open', 'close', 'high', 'low', 'volume']]
    df['ema1'] = df['close'].ewm(span=13, adjust=False).mean()
    df['ema2'] = df['close'].ewm(span=22, adjust=False).mean()
    open = df['open']
    close = df['close']
    position = [0]
    money = []

    for i in range(2, len(df)):
        if sum(position) == 0:
            if df['ema1'][i - 1] > df['ema2'][i - 1] and df['ema1'][i - 2] < df['ema2'][i - 2]:
                position.append(1)
                buy.append(open[i])
                btime.append(df.index[i])
                side.append('Long')
                tday.append(day)
            if df['ema1'][i - 1] < df['ema2'][i - 1] and df['ema1'][i - 2] > df['ema2'][i - 2]:
                position.append(-1)
                sell.append(open[i])
                stime.append(df.index[i])
                side.append('Short')
                tday.append(day)
        if sum(position) == 1:
            if df['ema1'][i - 1] < df['ema2'][i - 1]:
                position.append(-1)
                sell.append(open[i])
                stime.append(df.index[i])
                profit = (open[i] - buy[-1]) * contract - commission * contract / 10 - 2 * contract
                money.append(profit)
                position.append(-1)
                sell.append(open[i])
                stime.append(df.index[i])
                side.append('Short')
                tday.append(day)
        if sum(position) == -1:
            if df['ema1'][i - 1] > df['ema2'][i - 1]:
                position.append(1)
                buy.append(open[i])
                btime.append(df.index[i])
                profit = (sell[-1] - open[i]) * contract - commission * contract / 10 - 2 * contract
                money.append(profit)
                position.append(1)
                buy.append(open[i])
                btime.append(df.index[i])
                side.append('Long')
                tday.append(day)
    if sum(position) == 1:
        position.append(-1)
        sell.append(open[len(df) - 2])
        stime.append(df.index[len(df) - 2])
        profit = (open[len(df) - 2] - buy[-1]) * contract - commission * contract / 10 - 2 * contract
        money.append(profit)
    if sum(position) == -1:
        position.append(1)
        buy.append(open[len(df) - 2])
        btime.append(df.index[len(df) - 2])
        profit = (sell[-1] - open[len(df) - 2]) * contract - commission * contract / 10 - 2 * contract
        money.append(profit)
    P_L = sum(money)
    returns.append(P_L)
    print(P_L)
    time.sleep(0.5)

returns = np.array(returns)
check = pd.DataFrame({'tradedate': tday, 'tradeside': side, 'buytime': btime, 'selltime': stime, 'buyprice': buy,
                      'sellprice': sell})

check['returns'] = 0
check['profit'] = 0.0
for i in range(len(check)):
    if check.tradeside[i] == 'Long':
        check['profit'][i] = (check.sellprice[i] - check.buyprice[i]) * contract - commission * contract / 10 - 2 * contract
        check['returns'][i] = (check['profit'][i] / 2 / check.buyprice[i] * 100).round(3)
    if check.tradeside[i] == 'Short':
        check['profit'][i] = - (check.buyprice[i] - check.sellprice[i]) * contract - commission * contract / 10 - 2 * contract
        check['returns'][i] = (check['profit'][i] / 2 / check.sellprice[i] * 100).round(3)
cumsum = check['profit'].cumsum()


def VaR(returns):
    percs = [10.0]
    var1 = scs.scoreatpercentile(returns, percs)
    for pair in zip(percs, var1):
        return -pair[1]


def MD(returns):
    df = pd.DataFrame(returns, columns=['returns'])
    cum_ret = df.cumsum()
    df['DD'] = cum_ret.cummax() - cum_ret
    MaxDDD = 0
    days = []
    for i in range(len(df['DD'])):
        if df['DD'][i] == 0:
            MaxDDD = 0
            days.append(MaxDDD)
        else:
            MaxDDD += 1
            days.append(MaxDDD)
    return df['DD'].max(), np.max(np.array(days))


win_rate = len([i for i in check.profit if i > 0]) / len(check)
print(check)
print('%-25s:' % 'No. of Trades', len(check), '\n%-25s:' % 'Win Rate',
      f'{round((len([i for i in check.profit if i > 0]) * 100 / len(check)), 3)}%')
print('%-25s:' % 'Total Net Profit', (check.profit.sum()).round(0), returns.sum().round(0))
print('%-25s:' % 'Profit Factor',
      -(check.profit[check.profit > 0].mean() / check.profit[check.profit < 0].mean()).round(2))
print('%-25s:' % 'Average Net Profit', (check.profit.sum() / len(check)).round(0))
print('%-25s:' % 'Tharp Expectancy', ((win_rate * (check.profit[check.profit > 0].mean()) +
                                       (1 - win_rate) * check.profit[check.profit < 0].mean()) /
                                      -check.profit[check.profit < 0].mean()).round(2))
print('%-25s:' % 'Maximum Drawdown', MD(returns)[0].round(0))
print('%-25s:' % 'Drawdown Duration', f'{MD(returns)[1]}Days')
print('%-25s:' % 'Sharpe Ratio', (check.returns.mean() / check.returns.std()).round(3))
print('%-25s:' % 'Largest Day Profit', max(returns[returns > 0]).round(0))
print('%-25s:' % 'Largest Day Loss', min(returns[returns < 0]).round(0))
print('%-25s:' % 'Average Day Profit', (returns[returns > 0].mean()).round(0))
print('%-25s:' % 'Average Day Loss', (returns[returns < 0].mean()).round(0))
print('%-25s:' % 'Average Trade Profit', (check.profit[check.profit > 0].mean()).round(0))
print('%-25s:' % 'Average Trade Loss', (check.profit[check.profit < 0].mean()).round(0))
print('%-25s:' % 'Average Return', f'{(check.returns.mean()).round(3)}%')
print('%-25s:' % 'Standard Deviation', (check.returns.std()).round(2))
print('%-25s:' % 'Value at Risk', f'90%< {(VaR(check.returns)).round(3)}%')

chart = pd.DataFrame(returns.cumsum())
chart.index = pd.to_datetime(date)
chart = chart.loc[~(chart == 0).all(axis=1)]
chart.plot()
plt.show()

quote_ctx.close()
