# we load the necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs
from functions.position_VB import positionVB

quarters = ['2023_Q1', '2023_Q3', '2023_Q4',
            '2024_Q2', '2024_Q4',
            '2025_Q1', '2025_Q2']

# Strategy: 
# mrentum based on two intersecting moving averages
# applied to NQ futures only

# fast: EMA10 vs slow: EMA60

def mySR(x, scale):
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)

# create an empty DataFrame to store summary for all quarters
summary_data2_all_quarters = pd.DataFrame()

for quarter in quarters:

    print(f'Processing quarter: {quarter}')

    data2 = pd.read_parquet(f'data/data2_{quarter}.parquet')

    # Lets set the datetime index
    data2.set_index('datetime', inplace = True)


    # assumption
    # let's create an object named "pos_flat" 
    # = 1 if position has to be flat (= 0) - we do not trade
    # = 0 otherwise

    # let's fill it first with zeros
    pos_flat = np.zeros(len(data2))

    # 
    breaks = (data2.index.time >= pd.to_datetime("16:41").time()) & \
          (data2.index.time <= pd.to_datetime("18:10").time())
    
    pos_flat[breaks] = 1

    dweek_ = data2.index.dayofweek + 1
    time_ = data2.index.time
    pos_flat[((dweek_ == 5) & (time_ > pd.to_datetime('17:00').time())) |      # end of Friday
          (dweek_ == 6) |                                                  # whole Saturday (just in case)
          ((dweek_ == 7) & (time_ <= pd.to_datetime('18:00').time()))] = 1
                                                               # beginning of Sunday
    # apply the strategy
    ##############################################################
    
    # We calculate the appropriate EMA
    signalEMA_values = data2['XAG'].ewm(span = 30).mean().to_numpy()
    slowEMA_values = data2['XAG'].ewm(span = 240).mean().to_numpy()
                                
    # We calculate the standard deviation
    volat_sd_values = data2['XAG'].rolling(window = 60).std().to_numpy()

    # Insert NaNs wherever the original price is missing
    signalEMA_values[data2['XAG'].isna()] = np.nan
    slowEMA_values[data2['XAG'].isna()] = np.nan 
    volat_sd_values[data2['XAG'].isna()] = np.nan 

    # Calculate position for momentum strategy
    pos_mom = positionVB(signal = signalEMA_values, 
                        lower = slowEMA_values - 1 * volat_sd_values,
                        upper = slowEMA_values + 1 * volat_sd_values,
                        pos_flat = pos_flat,
                        strategy = "mom")
    pos_mr = -pos_mom
    pos_mr[pos_flat == 1] = 0
    pos_mom[pos_flat == 1] = 0

    # Calculate gross pnl
    pnl_gross_mr = np.where(np.isnan(pos_mom * data2['XAG'].diff()), 0, pos_mom * data2['XAG'].diff() * 5000) 
    pnl_gross_mr_pct = pnl_gross_mr / data2["XAG"].shift(1)

    # Add stop loss condition
                    # Calculate cumulative PnL for each day and apply stop loss
    pnl_gross_mr_series = pd.Series(pnl_gross_mr, index=data2.index)
                    
    # Define stop loss threshold (e.g., -1000 per day)
    stop_loss_threshold = -1000
                    
    # Calculate cumulative daily PnL
    daily_cumul_pnl_mr = pnl_gross_mr_series.groupby(data2.index.date).cumsum()
                    
    # Create stop loss mask (stop trading for rest of day if threshold hit)
    stop_loss_triggered_mr = (daily_cumul_pnl_mr <= stop_loss_threshold).groupby(data2.index.date).cummax()
                    
    # Apply stop loss by setting position to 0 after trigger
    pos_mr_sl = pos_mom.copy()
    pos_mr_sl[stop_loss_triggered_mr] = 0

                    
    # Recalculate PnL with stop loss
    pnl_gross_mr = np.where(np.isnan(pos_mr_sl * data2['XAG'].diff()), 0, pos_mr_sl * data2['XAG'].diff() * 5000)
    pnl_gross_mr_pct = pnl_gross_mr / data2["XAG"].shift(1)
    # Calculate number of transactions
    ntrans = np.abs(np.diff(pos_mom, prepend = 0))

    # Calculate net pnl
    pnl_net_mr = pnl_gross_mr - ntrans * 10  # cost $10 per transaction on E6
    pnl_net_mr_pct = pnl_net_mr / data2["XAG"].shift(1)

    # Aggregate to daily data
    pnl_gross_mr = pd.Series(pnl_gross_mr)
    pnl_gross_mr.index = data2['XAG'].index.time
    pnl_gross_mr_d = pnl_gross_mr.groupby(data2['XAG'].index.date).sum()
    pnl_gross_mr_pct_d = pnl_gross_mr_pct.groupby(data2.index.date).sum()

    pnl_net_mr = pd.Series(pnl_net_mr)
    pnl_net_mr.index = data2['XAG'].index.time
    pnl_net_mr_d = pnl_net_mr.groupby(data2['XAG'].index.date).sum()
    pnl_net_mr_pct_d = pnl_net_mr_pct.groupby(data2.index.date).sum()

    ntrans = pd.Series(ntrans)
    ntrans.index = data2['XAG'].index.time
    ntrans_d = ntrans.groupby(data2['XAG'].index.date).sum()

    gross_SR_mr = mySR(pnl_gross_mr_d, scale=252)
    net_SR_mr = mySR(pnl_net_mr_d, scale=252)
    gross_PnL_mr = pnl_gross_mr_d.sum()
    net_PnL_mr = pnl_net_mr_d.sum()
    gross_CR_mr = qs.stats.calmar(pnl_gross_mr_pct_d.dropna()).round(4)
    net_CR_mr = qs.stats.calmar(pnl_net_mr_pct_d.dropna()).round(4)
    
    av_daily_ntrans = ntrans_d.mean()
    stat = (net_SR_mr - 0.5) * np.maximum(0, np.log(np.abs(net_PnL_mr/1000)))


                    # Collect the necessary results into one object
    summary = pd.DataFrame({'quarter': quarter,
                            'gross_SR': gross_SR_mr,
                            'net_SR': net_SR_mr,
                            'gross_PnL': gross_PnL_mr,
                            'net_PnL': net_PnL_mr,
                            'gross_CR': gross_CR_mr,
                            'net_CR': net_CR_mr,
                            'av_daily_ntrans': av_daily_ntrans,
                            'stat': stat
                        }, index=[0])

                
              
    # Append results to the summary
    summary_data2_all_quarters = pd.concat([summary_data2_all_quarters, summary], ignore_index=True)

    # plot of cumulative gros and net returns
    # and save it as a png file

    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(pnl_gross_mr_d.fillna(0)), label = 'Gross PnL', color='blue')
    plt.plot(np.cumsum(pnl_net_mr_d.fillna(0)), label = 'Net PnL', color='red')
    plt.title('Cumulative Gross and Net PnL (' + quarter + ')')
    plt.legend()
    plt.grid(axis='x')

    plt.savefig(f"data2_{quarter}.png", dpi = 300, bbox_inches = "tight")
    plt.close()

    # remove ALL created objects to free memory
    # and prevent potential bugs in the next iteration
    del data2, pos_flat, signalEMA_values, slowEMA_values, volat_sd_values
    del pos_mr, pnl_gross_mr, pnl_gross_mr_pct, pnl_net_mr, pnl_net_mr_pct
    del ntrans, pnl_gross_mr_d, pnl_gross_mr_pct_d, pnl_net_mr_d, pnl_net_mr_pct_d
    del ntrans_d, summary

# save the summary for all quarters to a csv file
summary_data2_all_quarters.to_csv('summary_data2_all_quarters.csv', index=False)