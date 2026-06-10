import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


st.set_page_config(
    page_title="TSLA Stock Price Prediction App",
    page_icon="📈",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


HORIZON_OPTIONS = {
    "Next minute": {"steps": 1, "interval": "1m", "unit": "minute"},
    "Next day": {"steps": 1, "interval": "1d", "unit": "business day"},
    "Next week": {"steps": 5, "interval": "1d", "unit": "business days"},
    "Next month": {"steps": 21, "interval": "1d", "unit": "business days"},
    "Next year": {"steps": 252, "interval": "1d", "unit": "business days"},
    "Custom": {"steps": 30, "interval": "1d", "unit": "business days"},
}


@st.cache_data(ttl=60 * 30)
def load_stock_data(
    ticker: str,
    start_date: dt.date,
    end_date: dt.date,
    interval: str,
) -> pd.DataFrame:
    """Download market data from Yahoo Finance.

    Daily data uses the selected date range.
    1-minute data uses Yahoo Finance's recent intraday data window.
    """
    if interval == "1m":
        data = yf.download(
            ticker,
            period="7d",
            interval="1m",
            progress=False,
            auto_adjust=False,
            prepost=False,
        )
    else:
        # yfinance end date is exclusive, so add one day to include the chosen end date.
        download_end = end_date + dt.timedelta(days=1)
        data = yf.download(
            ticker,
            start=start_date,
            end=download_end,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna(how="all")
    return data


@st.cache_data(ttl=60 * 60)
def load_company_info(ticker: str) -> dict:
    """Fetch basic company metadata."""
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def get_price_column(data: pd.DataFrame) -> str:
    if "Adj Close" in data.columns and data["Adj Close"].notna().any():
        return "Adj Close"
    return "Close"


def add_indicators(data: pd.DataFrame, price_col: str) -> pd.DataFrame:
    df = data.copy()
    df["MA20"] = df[price_col].rolling(window=20).mean()
    df["MA50"] = df[price_col].rolling(window=50).mean()
    df["MA200"] = df[price_col].rolling(window=200).mean()
    df["Return"] = df[price_col].pct_change()
    return df


def make_forecast(
    data: pd.DataFrame,
    price_col: str,
    forecast_steps: int,
    model_type: str,
    degree: int,
    interval: str,
):
    """Train a simple time-index regression model and predict future prices.

    This app does not use a saved pre-trained model. It trains a small regression
    model live from the currently downloaded data.
    """
    df = data[[price_col]].dropna().copy()
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Step"] = np.arange(len(df))

    X = df[["Step"]]
    y = df[price_col]

    if len(df) < 30:
        raise ValueError("Not enough data for prediction. Please choose a longer date range or a different ticker.")

    split_index = max(int(len(df) * 0.8), 1)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if model_type == "Polynomial Regression":
        model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)

    # Refit on the full data before forecasting the future.
    model.fit(X, y)
    future_steps = pd.DataFrame({"Step": np.arange(len(df), len(df) + forecast_steps)})
    future_predictions = model.predict(future_steps)

    last_date = pd.to_datetime(data.index[-1])
    if interval == "1m":
        future_dates = pd.date_range(last_date + pd.Timedelta(minutes=1), periods=forecast_steps, freq="min")
    else:
        future_dates = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=forecast_steps)

    forecast_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted Price": future_predictions,
        }
    ).set_index("Date")

    metrics = {
        "MAE": mean_absolute_error(y_test, test_predictions) if len(y_test) else np.nan,
        "RMSE": np.sqrt(mean_squared_error(y_test, test_predictions)) if len(y_test) else np.nan,
        "R2": r2_score(y_test, test_predictions) if len(y_test) > 1 else np.nan,
    }

    backtest_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": test_predictions,
        },
        index=df.loc[split_index:, "Date"],
    )

    return forecast_df, metrics, backtest_df


def format_money(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


# Header
st.markdown('<div class="main-title">📈 Stock Price Prediction App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Analyze TSLA or any ticker and forecast the next minute, day, week, month, or year.</div>',
    unsafe_allow_html=True,
)


# Sidebar controls
st.sidebar.header("⚙️ App Settings")

ticker = st.sidebar.text_input(
    "Stock ticker",
    value="TSLA",
    help="Examples: TSLA, AAPL, MSFT, AMZN, NVDA",
).upper().strip()

prediction_target = st.sidebar.selectbox(
    "Prediction target",
    list(HORIZON_OPTIONS.keys()),
    index=1,
)

selected_horizon = HORIZON_OPTIONS[prediction_target]
interval = selected_horizon["interval"]
forecast_steps = selected_horizon["steps"]

if prediction_target == "Custom":
    forecast_steps = st.sidebar.slider("Custom forecast business days", min_value=1, max_value=252, value=30, step=1)

if prediction_target == "Next minute":
    st.sidebar.info("Next-minute prediction uses recent 1-minute intraday data from Yahoo Finance, usually limited to the last 7 days.")


today = dt.date.today()
default_start = today - dt.timedelta(days=365 * 5)

if interval == "1d":
    start_date = st.sidebar.date_input("Start date", value=default_start, max_value=today)
    end_date = st.sidebar.date_input("End date", value=today, max_value=today)
else:
    start_date = today - dt.timedelta(days=7)
    end_date = today

model_type = st.sidebar.selectbox("Prediction model", ["Linear Regression", "Polynomial Regression"])
degree = 2
if model_type == "Polynomial Regression":
    degree = st.sidebar.slider("Polynomial degree", min_value=2, max_value=4, value=2)

show_volume = st.sidebar.checkbox("Show volume chart", value=True)
show_company_info = st.sidebar.checkbox("Show company info", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("No separate trained model file is required. The app trains a small regression model live from the selected data.")
st.sidebar.info("This app is for educational purposes only. Predictions are not financial advice.")


if not ticker:
    st.warning("Please enter a stock ticker in the sidebar.")
    st.stop()

if interval == "1d" and start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()


try:
    with st.spinner(f"Loading data for {ticker}..."):
        data = load_stock_data(ticker, start_date, end_date, interval)

    if data.empty:
        st.error("No data found. Please check the ticker symbol or choose a different date range.")
        st.stop()

    price_col = get_price_column(data)
    data = add_indicators(data, price_col)

    latest_price = data[price_col].dropna().iloc[-1]
    first_price = data[price_col].dropna().iloc[0]
    price_change = latest_price - first_price
    price_change_pct = (price_change / first_price) * 100

    high_price = data["High"].max() if "High" in data else np.nan
    low_price = data["Low"].min() if "Low" in data else np.nan

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Price", format_money(latest_price))
    col2.metric("Period Change", format_money(price_change), f"{price_change_pct:,.2f}%")
    col3.metric("Highest Price", format_money(high_price))
    col4.metric("Lowest Price", format_money(low_price))

    data_frequency_label = "1-minute intraday" if interval == "1m" else "daily"
    st.caption(
        f"Using {data_frequency_label} data. Target: {prediction_target}. "
        f"Forecast steps: {forecast_steps} {selected_horizon['unit']}"
    )

    if show_company_info:
        info = load_company_info(ticker)
        company_name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        website = info.get("website", "")
        summary = info.get("longBusinessSummary", "")

        with st.expander(f"🏢 Company Info: {company_name}", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Ticker:** {ticker}")
            c2.write(f"**Sector:** {sector}")
            c3.write(f"**Industry:** {industry}")
            if website:
                st.write(f"**Website:** {website}")
            if summary:
                st.write(summary[:900] + ("..." if len(summary) > 900 else ""))

    tabs = st.tabs(["📊 Price Chart", "🔮 Forecast", "📄 Data", "ℹ️ About"])

    with tabs[0]:
        st.subheader(f"{ticker} Historical Price")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[price_col],
                mode="lines",
                name=price_col,
                line=dict(color="#2563eb", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MA20"],
                mode="lines",
                name="20-period MA",
                line=dict(color="#f97316", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MA50"],
                mode="lines",
                name="50-period MA",
                line=dict(color="#16a34a", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MA200"],
                mode="lines",
                name="200-period MA",
                line=dict(color="#9333ea", width=1.5),
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=560,
            hovermode="x unified",
            xaxis_title="Date/Time",
            yaxis_title="Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_volume and "Volume" in data.columns:
            st.subheader("Trading Volume")
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color="#64748b"))
            volume_fig.update_layout(
                template="plotly_white",
                height=300,
                xaxis_title="Date/Time",
                yaxis_title="Volume",
            )
            st.plotly_chart(volume_fig, use_container_width=True)

    with tabs[1]:
        st.subheader(f"{ticker} Forecast: {prediction_target}")
        try:
            forecast_df, metrics, backtest_df = make_forecast(
                data=data,
                price_col=price_col,
                forecast_steps=forecast_steps,
                model_type=model_type,
                degree=degree,
                interval=interval,
            )

            predicted_final_price = forecast_df["Predicted Price"].iloc[-1]
            forecast_change = predicted_final_price - latest_price
            forecast_change_pct = (forecast_change / latest_price) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Forecasted Price", format_money(predicted_final_price), f"{forecast_change_pct:,.2f}%")
            m2.metric("Backtest MAE", format_money(metrics["MAE"]))
            m3.metric("Backtest RMSE", format_money(metrics["RMSE"]))
            m4.metric("Backtest R²", "N/A" if pd.isna(metrics["R2"]) else f"{metrics['R2']:,.3f}")

            forecast_fig = go.Figure()
            forecast_fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[price_col],
                    mode="lines",
                    name="Historical Price",
                    line=dict(color="#2563eb", width=2),
                )
            )
            forecast_fig.add_trace(
                go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df["Predicted Price"],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#dc2626", width=2, dash="dash"),
                    marker=dict(size=5),
                )
            )
            forecast_fig.update_layout(
                template="plotly_white",
                height=560,
                hovermode="x unified",
                xaxis_title="Date/Time",
                yaxis_title="Price",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(forecast_fig, use_container_width=True)

            with st.expander("View backtest: actual vs predicted", expanded=False):
                backtest_fig = go.Figure()
                backtest_fig.add_trace(
                    go.Scatter(
                        x=backtest_df.index,
                        y=backtest_df["Actual"],
                        mode="lines",
                        name="Actual",
                        line=dict(color="#2563eb"),
                    )
                )
                backtest_fig.add_trace(
                    go.Scatter(
                        x=backtest_df.index,
                        y=backtest_df["Predicted"],
                        mode="lines",
                        name="Predicted",
                        line=dict(color="#dc2626", dash="dash"),
                    )
                )
                backtest_fig.update_layout(template="plotly_white", height=420, hovermode="x unified")
                st.plotly_chart(backtest_fig, use_container_width=True)

            st.write("### Forecast Table")
            st.dataframe(forecast_df.style.format({"Predicted Price": "${:,.2f}"}), use_container_width=True)

            csv = forecast_df.to_csv().encode("utf-8")
            st.download_button(
                label="Download forecast as CSV",
                data=csv,
                file_name=f"{ticker}_forecast.csv",
                mime="text/csv",
            )

        except ValueError as exc:
            st.warning(str(exc))

    with tabs[2]:
        st.subheader("Historical Data")
        st.dataframe(data.sort_index(ascending=False), use_container_width=True)

        csv = data.to_csv().encode("utf-8")
        st.download_button(
            label="Download historical data as CSV",
            data=csv,
            file_name=f"{ticker}_historical_data.csv",
            mime="text/csv",
        )

    with tabs[3]:
        st.subheader("About this App")
        st.write(
            """
            This Streamlit app downloads stock market data using Yahoo Finance, displays historical prices,
            calculates moving averages, and creates a simple forecast using regression.

            **Important model note:** This version does **not** need a separate trained model file like `gru_model.h5`.
            It trains a lightweight regression model live whenever the user selects a ticker and prediction target.

            **Features:**
            - Default ticker is TSLA, but any Yahoo Finance ticker can be entered
            - Forecast targets: next minute, next day, next week, next month, next year, or custom business days
            - Historical price and volume charts
            - Moving averages: 20-period, 50-period, and 200-period
            - Linear or polynomial regression forecast
            - Backtest metrics: MAE, RMSE, and R²
            - CSV downloads

            **Disclaimer:** This is an educational project. Stock markets are unpredictable, and this model is intentionally simple.
            Do not use the output as financial advice.
            """
        )

except Exception as exc:
    st.error("Something went wrong while running the app.")
    st.exception(exc)

   
