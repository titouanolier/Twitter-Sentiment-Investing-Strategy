import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import datetime as dt
import yfinance as yf
import os
plt.style.use('ggplot')


sentiment_df = pd.read_csv('sentiment_data.csv')

# exclure certains symboles dès le début car plus de donnée pas disponible dans yfinance
exclude_symbols = ["ATVI", "MRO"]
sentiment_df = sentiment_df[~sentiment_df["symbol"].isin(exclude_symbols)]

sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

sentiment_df = sentiment_df.set_index(['date', 'symbol'])

#try to capture the engagement ratio to avoid bots

sentiment_df['engagement_ratio'] = sentiment_df['twitterComments']/sentiment_df['twitterLikes']

#filter to avoid white noise

sentiment_df = sentiment_df[(sentiment_df['twitterLikes']>20)&(sentiment_df['twitterComments']>10)]

#Aggregate Monthly and calculate average sentiment for the month

aggragated_df = (sentiment_df.reset_index('symbol').groupby([pd.Grouper(freq='M'), 'symbol'])
                    [['engagement_ratio']].mean())

aggragated_df['rank'] = (aggragated_df.groupby(level=0)['engagement_ratio']
                         .transform(lambda x: x.rank(ascending=False)))

#Select Top 5 Stocks based on their cross-sectional ranking for each month

filtered_df = aggragated_df[aggragated_df['rank']<6].copy()

filtered_df = filtered_df.reset_index(level=1)

filtered_df.index = filtered_df.index+pd.DateOffset(1)

filtered_df = filtered_df.reset_index().set_index(['date', 'symbol'])

#Extract the stocks to form portfolios with at the start of each new month

dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}

for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()

#Download fresh stock prices for only selected/shortlisted stocks

stocks_list = sentiment_df.index.get_level_values('symbol').unique().tolist()


prices_df = yf.download(tickers=stocks_list,
                        start='2021-01-01',
                        end='2023-03-01')

#Calculate Portfolio Returns with monthly rebalancing


returns_df = np.log(prices_df['Close']).diff().dropna()

portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():
    end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd()).strftime('%Y-%m-%d')

    cols = fixed_dates[start_date]

    temp_df = returns_df.loc[start_date:end_date, cols].mean(axis=1).to_frame('portfolio_return')

    portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)


#Download NASDAQ/QQQ prices and calculate returns to compare to our strategy

nasdaq_return = yf.download(tickers='QQQ',
                     start='2021-01-01',
                     end='2023-03-01')

nasdaq_return = np.log(nasdaq_return['Close']).diff()

# Renommer la colonne
nasdaq_return = nasdaq_return.rename(columns={"QQQ": "nasdaq_return"})

portfolio_df = portfolio_df.merge(nasdaq_return,
                                  left_index=True,
                                  right_index=True)

portfolios_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()).sub(1)

portfolios_cumulative_return.plot(figsize=(16,6))

plt.title('Twitter Engagement Ratio Strategy Return Over Time')

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.ylabel('Return')

plt.show()


# metrics to show


returns = portfolio_df[['portfolio_return', 'nasdaq_return']]

# 2. Calcul des statistiques
stats = pd.DataFrame(index=['portfolio_return', 'nasdaq_return'])

# Nombre de périodes par an (trading days ~ 252)
trading_days = 252

# Annualized Return
stats['Annual Return'] = (returns.mean() * trading_days * 100).round(2).astype(str) + '%'

# Annualized Volatility
stats['Volatility'] = (returns.std() * np.sqrt(trading_days) * 100).round(2).astype(str) + '%'

# Sharpe Ratio
stats['Sharpe Ratio'] = (returns.mean() * trading_days / (returns.std() * np.sqrt(trading_days))).round(2)


print(stats)








