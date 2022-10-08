import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.linear_model import LinearRegression


def download_stock_pool_data():
    # get data directory path
    cur_path = os.path.dirname(__file__)
    data_directory_path = os.path.relpath('../Data', cur_path)

    # get latest russell 1000 constituents
    russell1000_info = pd.read_excel(io=data_directory_path + "\\Russell_1000_Constituents_20221007.xlsx",
                                     sheet_name="Holdings", skiprows=range(7))
    # get list of tickers
    stocks_pool_list = list(russell1000_info.Ticker.values)
    # get historical market data of current Russel 1000 constituents
    stocks_pool_data = yf.Tickers(stocks_pool_list).history(start="2012-01-01")["Close"]

    # get historical market data of S&P500
    sp500_data = pd.DataFrame(yf.Ticker("^GSPC").history(start="2012-01-01")["Close"])
    # rename S&P 500 data column
    sp500_data.columns = ["SP500"]

    # merge two dataframes
    raw_data = stocks_pool_data.join(sp500_data)

    # drop stocks with more than 1000 NaNs
    raw_data = raw_data.dropna(axis="columns", thresh=2500)

    # save stock pool data into csv
    raw_data.to_csv(data_directory_path + "\\Raw_Data_20221007.csv")


def generate_residual_matrices():
    # get data directory path
    cur_path = os.path.dirname(__file__)
    data_directory_path = os.path.relpath('../Data', cur_path)
    # read data from file
    raw_data = pd.read_csv(data_directory_path + "\\Raw_Data_20221007.csv", index_col=0)

    # set rebalancing frequency: every month
    rebalance_freq_period = relativedelta(months=1)
    # set business day convention for rebalancing
    business_day_convention = "Modified Following"
    # training set length
    train_set_length_period = relativedelta(months=6)
    # set date range
    first_date = datetime.strptime(raw_data.index[0], "%Y-%m-%d")
    last_date = datetime.strptime(raw_data.index[-1], "%Y-%m-%d")

    # initialize date range
    train_start_date = first_date
    train_end_date = train_start_date + train_set_length_period - relativedelta(days=1)

    test_start_date = train_end_date + relativedelta(days=1)
    test_end_date = test_start_date + rebalance_freq_period

    # traverse the data set
    while test_end_date < last_date:
        # do regression
        temp_train_data = raw_data.loc[train_start_date.__str__()[:10]:test_start_date.__str__()[:10], :]
        # save the residuals
        temp_residuals = pd.DataFrame(index=temp_train_data.index, columns=temp_train_data.columns.drop("SP500"))
        for ticker in temp_residuals.columns:
            # get data
            y_x = temp_train_data[[ticker, "SP500"]]
            # drop nas
            y_x = y_x.dropna(axis="index", how="any")
            # rename columns
            y_x.columns = ["y", "x"]
            # whether there's sufficient trading days
            if len(y_x.index) < len(temp_train_data) * 0.9:
                continue
            else:
                # calculate returns
                y_x = np.log(y_x).diff().dropna(axis="index", how="any")
                y = np.array(y_x["y"])
                x = np.array(y_x["x"]).reshape(-1, 1)
                # do regression
                reg = LinearRegression(fit_intercept=True).fit(x, y)
                # calculate residual
                y_x.loc[:, "res"] = np.subtract(y, (reg.intercept_ - reg.coef_[0] * x)[:, 0])
                # add residual to temp_residuals
                temp_residuals.loc[y_x.index, ticker] = y_x.res
        # drop nans in dataframe
        temp_residuals = temp_residuals.dropna(axis="index", how="all")
        temp_residuals = temp_residuals.dropna(axis="columns", how="any")
        # calculate correlations
        temp_residuals = temp_residuals.astype(float)
        temp_corr_matrix = temp_residuals.corr()

        # save the matrix to file
        temp_corr_matrix.to_csv(data_directory_path + "\\Corr_Mat\\" + train_start_date.__str__()[:10] + ".csv")

        # update dates
        train_start_date += rebalance_freq_period
        test_start_date += rebalance_freq_period
        test_end_date += rebalance_freq_period

