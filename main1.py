import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller

st.set_page_config(layout="wide")

st.sidebar.markdown("Select the Stock you want to Analyze below:")
st.sidebar.markdown("LUV is for Southwest Airlines")
st.sidebar.markdown("NVDA is for NVIDIA")
st.sidebar.markdown("AMZN is for Amazon")
st.sidebar.markdown("WMT is for Walmart")

option = st.sidebar.selectbox("Select one stock", ("LUV", "NVDA", "AMZN", "WMT"))
import datetime

today = datetime.date.today()
before = today - datetime.timedelta(days=700)
start_date = st.sidebar.date_input("Start date", before)
end_date = st.sidebar.date_input("End date", today)
if start_date < end_date:
    st.sidebar.success("Start date: `%s`\n\nEnd date:`%s`" % (start_date, end_date))
else:
    st.sidebar.error("Error: End date must fall after start date.")

number = st.sidebar.number_input(
    "Enter the number of days of recent data you want", min_value=1, value=10
)
window = st.sidebar.number_input(
    "Select the rolling window for volatility of Stock", min_value=2, value=12
)
nolags = st.sidebar.selectbox("Select the lags you want", (5, 1, 2, 3, 4, 6, 7, 8, 9, 10))

# Download stock data
df = yf.download(option, start=start_date, end=end_date, progress=False)

# Drop unnecessary columns and flatten the multi-index column
df_stock = df.drop(["Open", "High", "Low", "Volume"], axis=1)
df_stock.columns = df_stock.columns.get_level_values(1)

# Rename column for clarity (optional)
df_stock = df_stock.rename(columns={option: "Price"})

if option == "LUV":
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Southwest Airlines Stock Closing Prices over time</h1>",
        unsafe_allow_html=True,
    )
elif option == "NVDA":
    st.markdown(
        "<h1 style='text-align: center; color: black;'>NVIDIA Stock Closing Prices over time</h1>",
        unsafe_allow_html=True,
    )
elif option == "AMZN":
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Amazon Stock Closing Prices over time</h1>",
        unsafe_allow_html=True,
    )
elif option == "WMT":
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Walmart Stock Closing Prices over time</h1>",
        unsafe_allow_html=True,
    )

# Plot closing prices
st.line_chart(df_stock)

progress_bar = st.progress(0)

# Recent data display
st.markdown("<h1 style='text-align: center; color: black;'>Recent Data of the selected Stock</h1>", unsafe_allow_html=True)


def recent_data(number):
    if isinstance(number, int) and number > 0:
        # Rename columns to remove ticker for display
        df_cleaned = df.copy()
        df_cleaned.columns = ["Close", "High", "Low", "Open", "Volume"]
        st.dataframe(df_cleaned.tail(number))
    else:
        st.error("Please enter a valid positive integer.")

recent_data(number)

# Calculate daily returns
df_stock["daily_returns"] = df_stock["Price"].pct_change() * 100
fig_daily, ax = plt.subplots(figsize=(12, 6))
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
plt.plot(df_stock["daily_returns"], label="Daily Returns")
plt.legend(loc="best")
st.pyplot(fig_daily)

# Calculate average stock price
avg_stock = df_stock["Price"].mean()
if option == "LUV":
    st.markdown(
        f"<h3 style='text-align: left; color: black;'>Average Stock Price of Southwest Airlines is {avg_stock:.2f}</h3>",
        unsafe_allow_html=True,
    )
elif option == "NVDA":
    st.markdown(
        f"<h3 style='text-align: left; color: black;'>Average Stock Price of NVIDIA is {avg_stock:.2f}</h3>",
        unsafe_allow_html=True,
    )
elif option == "AMZN":
    st.markdown(
        f"<h3 style='text-align: left; color: black;'>Average Stock Price of Amazon is {avg_stock:.2f}</h3>",
        unsafe_allow_html=True,
    )
elif option == "WMT":
    st.markdown(
        f"<h3 style='text-align: left; color: black;'>Average Stock Price of Walmart is {avg_stock:.2f}</h3>",
        unsafe_allow_html=True,
    )

# Plot stock volatility
st.markdown(
    f"<h1 style='text-align: center; color: black;'>Volatility Graph</h1>", unsafe_allow_html=True
)


def plot_stock_volatility(stock_df, window):
    daily_returns = stock_df["Price"].pct_change()
    rolling_std = daily_returns.rolling(window=window).std()

    plt.figure(figsize=(12, 6))
    rolling_std.plot()
    plt.title(f"Volatility of {option} Stock (Rolling {window}-Day Std Dev)")
    plt.xlabel("Date")
    plt.ylabel("Volatility (Std Dev)")
    plt.grid(True)
    st.pyplot(plt)


plot_stock_volatility(df_stock, window)

# Autocorrelation plot
def autocorr(timeseries, nolags):
    fig1 = plt.figure()
    lag_plot(timeseries["Price"], lag=nolags)
    plt.title(f"Autocorrelation plot with lag = {nolags}")
    plt.show()
    st.pyplot(fig1)


st.markdown(
    "<h1 style='text-align: center; color: black;'>Autocorrelation plot with selected number of lags</h1>",
    unsafe_allow_html=True,
)
autocorr(df_stock, nolags)

# Test for stationarity
def test_stationarity(timeseries):
    rolmean = timeseries["Price"].rolling(12).mean()
    rolstd = timeseries["Price"].rolling(12).std()
    fig = plt.figure()
    plt.plot(timeseries["Price"], color="blue", label="Original")
    plt.plot(rolmean, color="red", label="Rolling Mean")
    plt.plot(rolstd, color="black", label="Rolling Std")
    plt.legend(loc="best")
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Rolling Mean and Standard Deviation</h1>",
        unsafe_allow_html=True,
    )
    plt.title("Rolling Mean and Standard Deviation")
    plt.show(block=False)
    adft = adfuller(timeseries["Price"], autolag="AIC")
    output = pd.Series(
        adft[0:4], index=["Test Statistics", "p-value", "No. of lags used", "Number of observations used"]
    )
    for key, values in adft[4].items():
        output[f"critical value ({key})"] = values
    st.pyplot(fig)

    st.markdown(
        "<h1 style='text-align: center; color: black;'>Statistic values of Dickey Fuller Test</h1>",
        unsafe_allow_html=True,
    )
    st.dataframe(output)
    return output


test_stationarity(df_stock)
