import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

case_schiller_data = "./indices"
rf_rate_path = "./market/3month_tbill.csv"
market_return_path = "./market/sp500.csv"

# Import risk-free rate data (3-month Tbills)
# Eliminate faulty rows that have no data
rf_rate = pd.read_csv(rf_rate_path)
rf_rate = rf_rate[rf_rate["DTB3"] != "."]

# Import data on annual S&P 500 returns
market_r = pd.read_csv(market_return_path)

results_data = []

def calculate_apr(today_price, before_price, years):
    """ Calculates the annual rate of change between two prices """
    return (today_price / before_price)**(1/years) - 1

def get_period_returns(index_df: pd.DataFrame) -> list:
    """ Gets the 1-yr, 5-yr, 10-yr, 30-yr returns from Oct 23 for each index """
    current_price = index_df.iloc[-1, 1]
    one_yr_price = index_df.iloc[-12 - 1, 1]
    five_yr_price = index_df.iloc[5 * -12 - 1, 1]
    ten_yr_price = index_df.iloc[10 * -12 - 1, 1]

    one_yr_return = calculate_apr(current_price, one_yr_price, 1)
    five_yr_return = calculate_apr(current_price, five_yr_price, 5)
    ten_yr_return = calculate_apr(current_price, ten_yr_price, 10)

    # Every index except Dallas has enough data to do 30-yr return
    if len(index_df) >= 360:
        thirty_yr_price = index_df.iloc[30 * -12 - 1, 1]
        thirty_yr_return = calculate_apr(current_price, thirty_yr_price, 30)
    else:
        thirty_yr_price = index_df.iloc[23 * -12 - 1, 1]
        thirty_yr_return = calculate_apr(current_price, thirty_yr_price, 23)

    return [one_yr_return, five_yr_return, ten_yr_return, thirty_yr_return]

def get_index_annual_returns(index_df: pd.DataFrame) -> pd.DataFrame:
    """ Get the annual returns for each year since the start of the data """
    yoy_changes = index_df.iloc[:, 1].pct_change(periods=12) * 100
    index_df["YoY"] = yoy_changes
    
    yearly = index_df[index_df["DATE"].dt.month == 1]
    yearly = yearly.iloc[1:, :]
    yearly["Year"] = yearly["DATE"].dt.year - 1
    yearly.drop(columns=["DATE"])

    return yearly

def calculate_beta(index_df: pd.DataFrame, market_r: pd.DataFrame, name: str) -> float:
    """ Calculate annual beta of the index to the S&P 500 """
    yearly = get_index_annual_returns(index_df)    

    joined = pd.merge(yearly, market_r, how="inner", on="Year")
    
    market_returns = joined["Return"].values
    index_returns = joined["YoY"].values

    plt.scatter(market_returns, index_returns)
    plt.title(name)
    plt.xlabel("Market Returns")
    plt.ylabel("Asset Returns")
    plt.savefig(name + "_beta.png")

    plt.clf()

    beta, _ = np.polyfit(market_returns, index_returns, 1)

    return beta

def calculate_sharpe_ratio(index_df: pd.DataFrame, rf_rate: pd.DataFrame):
    """ Calculate the Sharpe ratio of the index with 3-month T bills"""
    annual_index_returns = get_index_annual_returns(index_df)

    rf_rate["DATE"] = pd.to_datetime(rf_rate["DATE"])
    rf_rate["Year"] = rf_rate["DATE"].dt.year

    rf_rate["Return"] = rf_rate["DTB3"].astype("float32")

    # The risk free rate for a certain year is the mean rate during that year
    annual_rf_returns = rf_rate.groupby("Year")["Return"].mean().reset_index()

    joined = pd.merge(annual_index_returns, annual_rf_returns, how="inner", on="Year")
    rf_returns = joined["Return"].to_numpy()
    asset_returns = joined["YoY"].to_numpy()

    excess_returns = asset_returns - rf_returns

    excess_return = np.mean(excess_returns)
    volatility = np.std(excess_returns)    

    sharpe = excess_return / volatility

    print (excess_return, volatility, sharpe)

    return excess_return / 100, volatility / 100, sharpe

for index in os.listdir(case_schiller_data):
    
    # Get the name of the metro index
    name = index.split(".")[0]

    index_df = pd.read_csv(os.path.join(case_schiller_data, index))
    index_df["DATE"] = pd.to_datetime(index_df["DATE"])

    # Initialize results
    result = [name]

    # Get 1,5,10,30-yr returns and add to results
    period_returns = get_period_returns(index_df)
    result.extend(period_returns)

    # Get beta
    beta = calculate_beta(index_df, market_r, name)
    result.append(beta)

    # Get Sharpe Ratio
    excess, volatility, sharpe = calculate_sharpe_ratio(index_df, rf_rate)
    result.append(excess)
    result.append(volatility)
    result.append(sharpe)

    # Add data
    results_data.append(result)

column_names = [
    "Name", 
    "1yr return", 
    "5yr return", 
    "10yr return", 
    "30yr return", 
    "30Y Annual Beta", 
    "Excess Return", 
    "Volatility", 
    "Sharpe"]
results_df = pd.DataFrame(results_data, columns=column_names)

results_df.to_csv("results.csv", index=False)