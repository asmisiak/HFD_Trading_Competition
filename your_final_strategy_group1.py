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
# momentum based on two intersecting moving averages
# applied to NQ futures only

# fast: EMA10 vs slow: EMA60

def mySR(x, scale):
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)

# create an empty DataFrame to store summary for all quarters
summary_data1_all_quarters = pd.DataFrame()

for quarter in quarters:

    print(f'Processing quarter: {quarter}')

    data1 = pd.read_parquet(f'data/data1_{quarter}.parquet')

    # Lets set the datetime index
    data1.set_index('datetime', inplace = True)

    # assumption 1
    # do not use in calculations the data from the first and last 10 minutes 
    # of the session (9:31-9:40 and 15:51-16:00) – put missing values there,
    data1.loc[data1.between_time("9:31", "9:40").index] = np.nan
    data1.loc[data1.between_time("15:51", "16:00").index] = np.nan

    # assumption 2
    # let's create an object named "pos_flat" 
    # = 1 if position has to be flat (= 0) - we do not trade
    # = 0 otherwise

    # let's fill it first with zeros
    pos_flat = np.zeros(len(data1))

    # do not trade within the first 25 minutes of stocks quotations (9:31-9:55),
    pos_flat[data1.index.time <= pd.to_datetime("9:55").time()] = 1

    # do not hold positions overnight (exit all positions 20 minutes 
    # before the session end, i.e. at 15:40),
    pos_flat[data1.index.time >= pd.to_datetime("15:40").time()] = 1

    # apply the strategy
    ##############################################################
    
    # We calculate the appropriate EMA
    signalEMA_values = data1['SP'].ewm(span = 30).mean().to_numpy()
    slowEMA_values = data1['SP'].ewm(span = 240).mean().to_numpy()
                                
    # We calculate the standard deviation
    volat_sd_values = data1['SP'].rolling(window = 120).std().to_numpy()

    # Insert NaNs wherever the original price is missing
    signalEMA_values[data1['SP'].isna()] = np.nan
    slowEMA_values[data1['SP'].isna()] = np.nan 
    volat_sd_values[data1['SP'].isna()] = np.nan 

    # Calculate position for momentum strategy
    pos_mom = positionVB(signal = signalEMA_values, 
                        lower = slowEMA_values - 1 * volat_sd_values,
                        upper = slowEMA_values + 1 * volat_sd_values,
                        pos_flat = pos_flat,
                        strategy = "mom")
                
    # Calculate gross pnl
    pnl_gross_mom = np.where(np.isnan(pos_mom * data1['SP'].diff()), 0, pos_mom * data1['SP'].diff() * 50) 
    pnl_gross_mom_pct = pnl_gross_mom / data1["SP"].shift(1)

    # Add stop loss condition
                    # Calculate cumulative PnL for each day and apply stop loss
    pnl_gross_mom_series = pd.Series(pnl_gross_mom, index=data1.index)
                    
    # Define stop loss threshold (e.g., -1000 per day)
    stop_loss_threshold = -500
                    
    # Calculate cumulative daily PnL
    daily_cumul_pnl_mom = pnl_gross_mom_series.groupby(data1.index.date).cumsum()
                    
    # Create stop loss mask (stop trading for rest of day if threshold hit)
    stop_loss_triggered_mom = (daily_cumul_pnl_mom <= stop_loss_threshold).groupby(data1.index.date).cummax()
                    
    # Apply stop loss by setting position to 0 after trigger
    pos_mom_sl = pos_mom.copy()
    pos_mom_sl[stop_loss_triggered_mom] = 0

                    
    # Recalculate PnL with stop loss
    pnl_gross_mom = np.where(np.isnan(pos_mom_sl * data1['SP'].diff()), 0, pos_mom_sl * data1['SP'].diff() * 50)

    # Calculate number of transactions
    ntrans = np.abs(np.diff(pos_mom, prepend = 0))

    # Calculate net pnl
    pnl_net_mom = pnl_gross_mom - ntrans * 12  # cost $10 per transaction on E6
    pnl_net_mom_pct = pnl_net_mom / data1["SP"].shift(1)

    # Aggregate to daily data
    pnl_gross_mom = pd.Series(pnl_gross_mom)
    pnl_gross_mom.index = data1['SP'].index.time
    pnl_gross_mom_d = pnl_gross_mom.groupby(data1['SP'].index.date).sum()
    pnl_gross_mom_pct_d = pnl_gross_mom_pct.groupby(data1.index.date).sum()

    pnl_net_mom = pd.Series(pnl_net_mom)
    pnl_net_mom.index = data1['SP'].index.time
    pnl_net_mom_d = pnl_net_mom.groupby(data1['SP'].index.date).sum()
    pnl_net_mom_pct_d = pnl_net_mom_pct.groupby(data1.index.date).sum()

    ntrans = pd.Series(ntrans)
    ntrans.index = data1['SP'].index.time
    ntrans_d = ntrans.groupby(data1['SP'].index.date).sum()

    gross_SR_mom = mySR(pnl_gross_mom_d, scale=252)
    net_SR_mom = mySR(pnl_net_mom_d, scale=252)
    gross_PnL_mom = pnl_gross_mom_d.sum()
    net_PnL_mom = pnl_net_mom_d.sum()
    gross_CR_mom = qs.stats.calmar(pnl_gross_mom_pct_d.dropna())
    net_CR_mom = qs.stats.calmar(pnl_net_mom_pct_d.dropna())
    
    av_daily_ntrans = ntrans_d.mean()
    stat = (net_SR_mom - 0.5) * np.maximum(0, np.log(np.abs(net_PnL_mom/1000)))

                    # Collect the necessary results into one object
    summary = pd.DataFrame({'quarter': quarter,
                            'gross_SR': gross_SR_mom,
                            'net_SR': net_SR_mom,
                            'gross_PnL': gross_PnL_mom,
                            'net_PnL': net_PnL_mom,
                            'gross_CR': gross_CR_mom,
                            'net_CR': net_CR_mom,
                            'av_daily_ntrans': av_daily_ntrans,
                            'stat': stat
                        }, index=[0])

                
              
    # Append results to the summary
    summary_data1_all_quarters = pd.concat([summary_data1_all_quarters, summary], ignore_index=True)

    # plot of cumulative gros and net returns
    # and save it as a png file

    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(pnl_gross_mom_d.fillna(0)), label = 'Gross PnL', color='blue')
    plt.plot(np.cumsum(pnl_net_mom_d.fillna(0)), label = 'Net PnL', color='red')
    plt.title('Cumulative Gross and Net PnL (' + quarter + ')')
    plt.legend()
    plt.grid(axis='x')

    plt.savefig(f"data1_{quarter}.png", dpi = 300, bbox_inches = "tight")
    plt.close()

    # remove ALL created objects to free memory
    # and prevent potential bugs in the next iteration
    del data1, pos_flat, signalEMA_values, slowEMA_values, volat_sd_values
    del pos_mom, pnl_gross_mom, pnl_gross_mom_pct, pnl_net_mom, pnl_net_mom_pct
    del ntrans, pnl_gross_mom_d, pnl_gross_mom_pct_d, pnl_net_mom_d, pnl_net_mom_pct_d
    del ntrans_d, summary

# save the summary for all quarters to a csv file
summary_data1_all_quarters.to_csv('summary_data1_all_quarters.csv', index=False)