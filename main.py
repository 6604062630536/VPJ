from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import datetime

def load_data():
    df = pd.read_csv("Weather_chiangmai.csv", parse_dates=["Date"], dayfirst=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.dropna()
    return df

def prepare_data(df, feature_cols, target_col='PM25', lookback=24):
    data_X = df[feature_cols].values  
    data_y = df[[target_col]].values  
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    data_X_scaled = scaler_X.fit_transform(data_X)
    data_y_scaled = scaler_y.fit_transform(data_y)

    X, y = [], []
    for i in range(len(data_X_scaled) - lookback):
        X.append(data_X_scaled[i:i+lookback])  
        y.append(data_y_scaled[i+lookback])  
    
    return np.array(X), np.array(y), scaler_X, scaler_y

def build_lstm_model(input_shape):
    model = Sequential([ 
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ✅ ใช้ @st.cache_resource เพื่อให้โมเดลถูกเทรนแค่ครั้งเดียว
@st.cache_resource
def train_model():
    df = load_data()
    feature_cols = [
        'Pressure_max', 'Pressure_min', 'Pressure_avg', 'Temp_max', 'Temp_min', 'Temp_avg', 
        'Humidity_max', 'Humidity_min', 'Humidity_avg', 'Precipitation', 'Sunshine', 
        'Evaporation', 'Wind_direct', 'Wind_speed'
    ]
    X, y, scaler_X, scaler_y = prepare_data(df, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

    mse = mean_squared_error(y_test, model.predict(X_test))

    return model, scaler_X, scaler_y, feature_cols, df, mse

def PM25():
    st.title("📊 PM2.5 Forecasting using LSTM")

    # ✅ ใช้โมเดลที่เทรนแล้ว
    model, scaler_X, scaler_y, feature_cols, df, mse = train_model()

    # ✅ ย้ายตัวเลือกวันที่มาอยู่ในหน้าหลัก
    selected_date = st.date_input("📅 Select a date", datetime.date(2016, 7, 11))

    # ✅ กรองข้อมูลตามวันที่ที่เลือก
    df_filtered = df[df["Date"] == pd.to_datetime(selected_date)]
    st.write("### Data Preview:")
    st.dataframe(df_filtered.head())

    # ✅ กดปุ่มก่อน Predict
    if st.button("🔮 Predict PM2.5"):
        if not df_filtered.empty:
            X_selected = df_filtered[feature_cols].values  
            X_selected_scaled = scaler_X.transform(X_selected)  
            X_selected_scaled = np.expand_dims(X_selected_scaled, axis=0)

            y_pred_scaled = model.predict(X_selected_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

            # ✅ ลดขนาดตัวหนังสือ
            st.write(f"⛅ Predicted PM2.5 for {selected_date}: ")
            st.write(f"### {y_pred[0][0]:.2f}")
            st.write(f"🎯 Mean Squared Error: {mse:.2f}" )

            st.success("✅ Prediction Completed!")
        else:
            st.error("❌ ไม่มีข้อมูลในวันที่ที่เลือก กรุณาเลือกวันใหม่")

def main(): 
    page = st.sidebar.selectbox("Select a page", ["PM25"])  
    if page == "PM25":
        PM25() 

if __name__ == "__main__":
    main()


