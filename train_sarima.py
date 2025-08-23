#!/usr/bin/env python3
"""
Command-line tool to fit a SARIMA model to a time series and generate
forecasts.  If no CSV is provided, the AirPassengers dataset is
downloaded and used by default.

Usage::

    python train_sarima.py --csv_path path/to/data.csv --date_col Date \
        --value_col Value --order 1 1 1 --seasonal_order 1 1 1 12

The script prints the model summary, generates a 24-period forecast and
saves a plot of the historical data and forecast to ``forecast.png``.
"""

import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def load_data(csv_path: str | None) -> pd.DataFrame:
    """Load a time series from a CSV file or download the AirPassengers data.

    The returned DataFrame has a datetime index and a single column named
    ``value``.

    Parameters
    ----------
    csv_path : str | None
        Path to a CSV file with a date column and a numeric value column.
        If ``None``, the AirPassengers dataset is downloaded.

    Returns
    -------
    pd.DataFrame
        Data frame with datetime index and a single column ``value``.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            raise ValueError("CSV must contain at least two columns: date and value")
        # Let the user specify which columns represent date and value via CLI
        return df
    # Download the AirPassengers dataset from GitHub
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    )
    df = pd.read_csv(url)
    df.rename(columns={"Passengers": "value", "Month": "date"}, inplace=True)
    return df


def prepare_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    """Convert a DataFrame into a time series with a DatetimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame.
    date_col : str
        Name of the column containing the dates.
    value_col : str
        Name of the column containing the numeric values.

    Returns
    -------
    pd.Series
        Time series indexed by datetime.
    """
    ts = df[[date_col, value_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], infer_datetime_format=True)
    ts.set_index(date_col, inplace=True)
    # Ensure the index is sorted
    ts.sort_index(inplace=True)
    series = ts[value_col].astype(float)
    return series


def parse_order(order_args: list[int]) -> Tuple[int, int, int]:
    """Parse a non-seasonal order from a list of integers."""
    if len(order_args) != 3:
        raise ValueError("Order must be three integers p d q")
    return tuple(int(x) for x in order_args)


def parse_seasonal_order(order_args: list[int]) -> Tuple[int, int, int, int]:
    """Parse a seasonal order from a list of integers."""
    if len(order_args) != 4:
        raise ValueError("Seasonal order must be four integers P D Q s")
    return tuple(int(x) for x in order_args)


def fit_sarima(series: pd.Series, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int]):
    """Fit a SARIMA model to the series and return the fitted model."""
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def forecast_and_plot(series: pd.Series, fitted_model, periods: int = 24, output_path: str = "forecast.png"):
    """Generate forecast and plot the original series along with forecast."""
    forecast = fitted_model.get_forecast(steps=periods)
    forecast_index = pd.date_range(series.index[-1], periods=periods + 1, freq=pd.infer_freq(series.index)).drop(series.index[-1])
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
    conf_int = forecast.conf_int(alpha=0.05)
    # Align confidence intervals with forecast index
    conf_int.index = forecast_index
    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]

    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series.values, label="Historical")
    plt.plot(forecast_series.index, forecast_series.values, label="Forecast", color="C1")
    plt.fill_between(forecast_series.index, lower, upper, color="C1", alpha=0.3, label="95% CI")
    plt.title("SARIMA Forecast")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Forecast plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a SARIMA model and forecast future values.")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to CSV file containing the time series.")
    parser.add_argument("--date_col", type=str, default="date", help="Name of the date column in the CSV file.")
    parser.add_argument("--value_col", type=str, default="value", help="Name of the value column in the CSV file.")
    parser.add_argument("--order", nargs=3, type=int, default=[1, 1, 1], help="Non-seasonal order p d q")
    parser.add_argument("--seasonal_order", nargs=4, type=int, default=[1, 1, 1, 12], help="Seasonal order P D Q s")
    parser.add_argument("--periods", type=int, default=24, help="Number of future periods to forecast")
    args = parser.parse_args()

    if args.csv_path:
        df = load_data(args.csv_path)
    else:
        df = load_data(None)
        args.date_col = "date"
        args.value_col = "value"

    series = prepare_series(df, args.date_col, args.value_col)
    order = parse_order(args.order)
    seasonal_order = parse_seasonal_order(args.seasonal_order)

    print(f"Fitting SARIMA model with order={order} and seasonal_order={seasonal_order}...")
    model = fit_sarima(series, order, seasonal_order)
    print(model.summary())

    forecast_and_plot(series, model, periods=args.periods, output_path="forecast.png")


if __name__ == "__main__":
    main()