# Time Series Forecasting with SARIMA

This repository provides everything you need to build and forecast a seasonal
autoregressive integrated moving average (SARIMA) model in Python.  SARIMA
is a classical statistical model designed to capture trend, seasonality and
noise in time–series data.  It extends the widely used ARIMA model with
seasonal components and can incorporate exogenous variables【852651330896506†L82-L110】.

## Why SARIMA?

Time–series data often exhibit recurring patterns such as weekly, monthly or
yearly cycles.  Capturing these *seasonal* patterns is important for
generating accurate forecasts.  A SARIMA model decomposes a series into
autoregressive (AR), integrated (I), moving–average (MA) and seasonal
components【852651330896506†L100-L110】.  It is particularly useful when your data
exhibits both trend and seasonality and you want a model that is easier to
interpret and faster to train than complex deep‑learning architectures.

## Project structure

```text
sarima-time-series-forecasting/
├── README.md                # This document
├── requirements.txt         # Python dependencies
├── notebooks/
│   └── training_sarima.ipynb  # Jupyter notebook with the full workflow
└── scripts/
    └── train_sarima.py       # Python script to run the training from the command line
```

## Setup

1. **Clone the repository and create a virtual environment (optional).**
   ```bash
   git clone <REPO_URL>
   cd sarima-time-series-forecasting
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the required libraries.**  The SARIMA example uses
   `statsmodels`, `pandas`, `numpy` and `matplotlib`.  These are listed in
   `requirements.txt` and can be installed with pip:
   ```bash
   pip install -r requirements.txt
   ```

## Data

For demonstration purposes, the code downloads the well–known
*AirPassengers* dataset, which contains monthly international airline
passenger numbers from 1949 to 1960.  The dataset exhibits a clear
monthly seasonality, making it a good example for a SARIMA model【852651330896506†L82-L96】.
You can substitute your own time–series by providing a CSV file with a
timestamp column and a numeric target column.

## Running the script

The `train_sarima.py` script loads the data, fits a SARIMA model with
predefined order parameters and plots the forecast.  You can specify a
custom dataset via the `--csv_path` argument:

```bash
python scripts/train_sarima.py --csv_path path/to/your/data.csv \
    --date_col Date --value_col Value --order 1 1 1 --seasonal_order 1 1 1 12
```

If no CSV is provided, the script automatically downloads the
AirPassengers dataset and uses default seasonal order `(1, 1, 1, 12)` and
non‑seasonal order `(1, 1, 1)`.

The script prints the model summary, generates a forecast for the next 24
periods and saves a plot of the historical series and forecast in
`forecast.png`.

## Model tuning and limitations

### Choosing SARIMA orders

SARIMA is defined by a non‑seasonal order `(p, d, q)` and a seasonal
order `(P, D, Q, s)` where `s` is the seasonal period.  The orders
control how many autoregressive lags (`p` and `P`), differencing
operations (`d` and `D`) and moving–average terms (`q` and `Q`) the model
uses.  A simple way to choose reasonable values is to plot the
**autocorrelation function (ACF)** and **partial autocorrelation
function (PACF)** of your data and look for significant lags.  In
practice you typically start with small orders such as `(1,1,1)` and
seasonal order `(1,1,1,s)`, then evaluate candidate models with
information criteria.

### Using BIC to avoid overfitting

When comparing different SARIMA configurations, it is important not to
add too many parameters.  Too many AR or MA terms can lead to
overfitting – the model may fit the training data well but generalise
poorly to new observations【455193929966286†L378-L385】.  To objectively compare
models, one can compute the **Bayesian information criterion (BIC)** or
**Akaike information criterion (AIC)** for each candidate.  These
metrics balance model fit and complexity; lower values indicate a
better trade‑off, and comparing BIC values across models helps to
identify a simpler model that still explains the data well【706520277493290†L360-L367】.
In the notebook and script you can modify the `--order` and
`--seasonal_order` arguments to try different combinations and inspect
the resulting BIC printed in the model summary.

### Linear model and hybrids

SARIMA (and its parent ARIMA) belongs to a class of **linear models** –
the current value of the series is modeled as a linear function of
past values and past forecast errors【263046113061295†L265-L281】.  As a result,
SARIMA excels at capturing linear trends and seasonal patterns but
cannot describe complex nonlinear relationships inherent in some time
series.  In practice, this limitation is often addressed by
**hybrid models** that combine SARIMA with machine‑learning or
deep‑learning methods: the linear SARIMA component captures the bulk
of the structure, while a nonlinear model learns any remaining
patterns.  Such hybrids have been used in winning solutions of
forecasting competitions like Kaggle and the M3/M4 series.  Even if
you eventually deploy a neural network, a well‑tuned SARIMA model
provides a strong baseline and interpretable benchmark.

## Notebook

The Jupyter notebook in `notebooks/training_sarima.ipynb` follows the
same workflow as the script but allows you to explore the data and
model interactively.  It loads the dataset, performs a seasonal
decomposition to visualise trend and seasonality, fits a SARIMA model
using `statsmodels.tsa.statespace.SARIMAX`, and generates forecasts.

## License

This project is released under the MIT License.  See the `LICENSE` file
for details.
