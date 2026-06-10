Stock Price Prediction Streamlit App
A Streamlit web app for analyzing and forecasting stock prices. The default ticker is TSLA, but users can enter any Yahoo Finance ticker.


Supports any Yahoo Finance ticker, for example AAPL, MSFT, NVDA, AMZN
Forecast targets:
Next minute
Next day
Next week
Next month
Next year
Custom number of business days
Historical price chart
Trading volume chart
Moving averages: 20-period, 50-period, and 200-period
Linear Regression or Polynomial Regression forecasting
Backtest metrics: MAE, RMSE, and R²
CSV downloads for historical data and forecast results
Files
text

app.py                 # Main Streamlit app
requirements.txt       # Python dependencies
README.md              # Project documentation
.streamlit/config.toml # Streamlit theme settings

Disclaimer
This app is for educational purposes only. The prediction model is simple and should not be used as financial advice
