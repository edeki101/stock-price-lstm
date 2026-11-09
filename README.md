# Stock Price Trend Prediction with LSTM

Predict the next trading day's closing price from past trends using a Keras LSTM, with 20/50-day moving averages and RSI-14 as inputs. Data comes from the Yahoo Finance API (`yfinance`).

## Deliverables

| # | Deliverable | Link |
|---|---|---|
| 1 | GitHub repository | https://github.com/edeki101/stock-price-lstm |
| 2 | Jupyter notebook | [stock_lstm_colab.ipynb](stock_lstm_colab.ipynb) |
| 3 | Model weights | [models/lstm_stock.weights.h5](models/lstm_stock.weights.h5) |
| 4 | Graphs | [outputs/](outputs/) |
| 5 | Streamlit dashboard | Coming soon (deployment steps below) |
| 6 | Project report (2 pages, PDF) | [report/Stock_LSTM_Report.pdf](report/Stock_LSTM_Report.pdf) |

## Results

![Predicted vs actual price on the test period](outputs/03_pred_vs_actual.png)

![Price with MA20, MA50 and RSI-14](outputs/01_price_ma_rsi.png)

![Training and validation loss](outputs/02_loss_curve.png)

The model is evaluated on a held-out test period and compared with a naive baseline (tomorrow's price = today's price). The LSTM tracks the price closely, but its error is essentially level with the baseline, so it shows no clear predictive edge. This is typical for price-only inputs, because markets are close to a random walk. Full metrics (RMSE, MAE, MAPE, directional accuracy) are in [outputs/results.json](outputs/results.json).

Results shown use AAPL daily prices from 2000-03-14 to 2008-10-14 (a public Yahoo-format dataset, used because Yahoo Finance was unreachable during the run). Running the notebook on a working connection pulls live data.

## Repository structure

```
stock-price-lstm/
├── stock_lstm_colab.ipynb    # full pipeline
├── app.py                    # Streamlit dashboard
├── requirements.txt          # dependencies
├── data/                     # prices with MA20, MA50, RSI14 (CSV)
├── models/                   # weights, full model, scalers
├── outputs/                  # graphs and results.json
└── report/                   # PDF report
```

## Quick start

**Google Colab (nothing to install)**

1. Open https://colab.research.google.com and choose **File > Upload notebook**.
2. Select `stock_lstm_colab.ipynb`.
3. Click **Runtime > Run all**. It takes a few minutes on a CPU.

**Your own computer**

```bash
git clone https://github.com/edeki101/stock-price-lstm.git
cd stock-price-lstm
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook stock_lstm_colab.ipynb
```

## How the project is built

1. **Fetch data.** `yf.download("AAPL", start="2015-01-01", auto_adjust=True)`. Change `TICKER` and `START` in the configuration cell to use another stock or period.
2. **Add indicators.** 20-day and 50-day moving averages, and the 14-day RSI, computed from past data only.
3. **Normalize and prepare.** Chronological 80/20 split; features standardized with scalers fit on the training data only; 60-day windows as samples.
4. **Build the LSTM.** `LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16) -> Dense(1)`, Adam optimizer, MSE loss.
5. **Train and validate.** The last 10% of the training windows are used for validation, with early stopping.
6. **Plot predictions vs actual.** Charts are saved to `outputs/`.
7. **Save the model.** Weights, full model and scalers are saved to `models/`.

The model predicts the next-day **log return**, and the price forecast is rebuilt as `previous close x exp(predicted return)`. Predicting raw price levels fails when the price rises beyond the range seen in training, because an LSTM cannot extrapolate.

To reload the saved model:

```python
from tensorflow import keras
model = keras.models.load_model("models/lstm_stock.keras")
```

## Reading the results

| Metric | Meaning |
|---|---|
| RMSE / MAE | Average price error in dollars (lower is better) |
| MAPE | Average error as a percentage of the price |
| Directional accuracy | How often the model gets up or down right |

A chart that hugs the actual line does not prove the model forecasts well, because a baseline that copies yesterday's price also hugs the line. Compare the LSTM's error with the naive baseline, and its directional accuracy with the share of up-days.

## Streamlit dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501`, enter a ticker, choose the look-back window and epochs, then click **Fetch data and train**.

To deploy it free on Streamlit Community Cloud:

1. Sign in at https://share.streamlit.io with GitHub.
2. Click **New app** and choose repository `edeki101/stock-price-lstm`, branch `main`, main file `app.py`.
3. Click **Deploy**, then copy the resulting link into the Deliverables table above.

## Disclaimer

This project is for learning only and is not financial advice. A model that fits past data does not guarantee future results.

## Author

**Ekundayo Monday Edeki**: [GitHub](https://github.com/edeki101) | [LinkedIn](https://linkedin.com/in/edeki-monday-ekundayo)
