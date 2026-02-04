# assignment1.py 
# Adam Nutt

# Set up
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_url  = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

# Load data
train_df = pd.read_csv(train_url)
test_df  = pd.read_csv(test_url)

# Columns
dt_col = "Timestamp"
y_col  = "trips"

# Parse timestamps + sort
train_df[dt_col] = pd.to_datetime(train_df[dt_col])
test_df[dt_col]  = pd.to_datetime(test_df[dt_col])

train_df = train_df.sort_values(dt_col)
test_df  = test_df.sort_values(dt_col)

# Build hourly training series
y_train = train_df.set_index(dt_col)[y_col].astype(float)
y_train = y_train.asfreq('h')

# Exponential Smoothing model
model = ExponentialSmoothing(
    y_train,
    trend="add",
    seasonal="add",
    seasonal_periods=168,
    initialization_method="estimated"
)

modelFit = model.fit(optimized=True)

# Forecast test period
n_steps = len(test_df)
pred = modelFit.forecast(n_steps)
pred = pred.to_numpy()


