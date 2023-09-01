import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from pmdarima.arima import auto_arima
from prophet import Prophet
from keras.models import load_model
import numpy as np 

st.title('Stock Price Prediction App')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')

def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()



@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df



option = st.sidebar.text_input('Enter a Stock Symbol (data taken from yfinance library)', value='SPY')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')




data = download_data(option, start_date, end_date)
scaler = StandardScaler()



def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    
    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Expoenetial Moving Average')
        st.line_chart(ema)


def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))



def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'
                                        , 'LSTM', 'ARIMA', 'Prophet'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        elif model == 'XGBoostRegressor':
            engine = XGBRegressor()
            model_engine(engine, num)
        elif model == 'LSTM':
            lstm_engine(num)
        elif model == 'ARIMA':
            arima_engine(num)
        elif model == 'Prophet':
            prophet_engine(num)


def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1
        
# Create a function to train and predict using the LSTM model
def lstm_engine(num):
    # Prepare the data for LSTM
    model = load_model('keras_model.h5')
    data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
    data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

    data_training_array = scaler.fit_transform(data_training)
    past_100_days = data_training.tail(100)

    final_df = past_100_days.append(data_testing, ignore_index = True)

    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    preds = model.predict(x_test)
    factor = scaler.scale_
    scale_factor = 1/factor[0]
    preds = preds * scale_factor
    y_test = y_test * scale_factor

    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    
    future_input_data = input_data[-num:]  # Using the last 100 days of available data for prediction

    future_predictions = []

    for _ in range(num):
        future_prediction = model.predict(np.array([future_input_data]))
        future_predictions.append(future_prediction[0, 0])
        future_input_data = np.roll(future_input_data, -1, axis=0)
        future_input_data[-1, 0] = future_prediction[0, 0]

    # Scale back the future predictions
    future_predictions = np.array(future_predictions) * scale_factor
    
    day = 1 
    for index, row in future_predictions.iterrows():
        st.text(f'Day {day}: {row["Close"]}')
        day += 1



# Create a function to train and predict using the ARIMA model
def arima_engine(num):
    model_arima= auto_arima(data["Close"],trace=True, error_action='ignore', start_p=1,start_q=1,max_p=3,max_q=3,
                suppress_warnings=True,stepwise=False,seasonal=False)
    model_arima.fit(data["Close"])

    forecast_pred = model_arima.predict(num)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1

# Create a function to train and predict using the Prophet model
def prophet_engine(num):
    # Prepare the data for Prophet
    df = data[['Close']].copy()
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

    # Training the Prophet model
    model = Prophet()
    model.fit(df)

    # Create a dataframe for future dates
    future = model.make_future_dataframe(periods=num)
    forecast = model.predict(future)

    # Forecast using Prophet
    forecast_data = forecast[['ds', 'yhat']].tail(num)
    forecast_data.rename(columns={'ds': 'Date', 'yhat': 'Close'}, inplace=True)
    forecast_data.set_index('Date', inplace=True)

    day = 1
    for index, row in forecast_data.iterrows():
        st.text(f'Day {day}: {row["Close"]}')
        day += 1

if __name__ == '__main__':
    main()
