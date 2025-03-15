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

def PM25():
    # 🟢 โหลดข้อมูล PM2.5 (จากไฟล์ CSV ที่มีอยู่)
    @st.cache_data
    def load_data():
        df = pd.read_csv("Weather_chiangmai.csv", parse_dates=["Date"], dayfirst=True)  # แก้ไขตรงนี้
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)   # ระบุ dayfirst=True
        df = df.dropna()  # ลบค่า Missing
        return df

    # 🟢 เตรียมข้อมูลสำหรับ LSTM
    def prepare_data(df, feature_cols=['Pressure_max', 'Pressure_min', 'Pressure_avg', 'Temp_max', 'Temp_min', 'Temp_avg', 
                                      'Humidity_max', 'Humidity_min', 'Humidity_avg', 'Precipitation', 'Sunshine', 
                                      'Evaporation', 'Wind_direct', 'Wind_speed'], target_col='PM25', lookback=24):
        # ตรวจสอบคอลัมน์ใน DataFrame ก่อน
        available_cols = df.columns
        print("Available columns:", available_cols)
        
        # ดึงข้อมูลจากคอลัมน์ที่เลือก
        data = df[feature_cols + [target_col]].values  
        scaler = MinMaxScaler()  # สร้างตัว Scaling
        data_scaled = scaler.fit_transform(data)  # ปรับข้อมูลให้อยู่ในช่วง 0-1
        
        X, y = [], []
        for i in range(len(data_scaled) - lookback):
            X.append(data_scaled[i:i+lookback, :-1])  # ใช้ข้อมูลทั้งหมดยกเว้นค่า PM25
            y.append(data_scaled[i+lookback, -1])  # ใช้ค่าของ PM25 เป็นผลลัพธ์ที่ต้องทำนาย
        
        return np.array(X), np.array(y), scaler  # คืนค่า X, y และตัว scaler

    # 🟢 สร้างโมเดล LSTM
    def build_lstm_model(input_shape):
        model = Sequential([ 
            LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    # 🟢 เริ่มสร้าง Streamlit UI
    st.title("📊 PM2.5 Forecasting using LSTM")

    # 🟡 ให้ผู้ใช้เลือกวัน
    selected_date = st.sidebar.date_input("เลือกวัน", datetime.date(2016, 7, 11))

    # 🟡 โหลดข้อมูล
    df = load_data()

    # 🟡 กรองข้อมูลตามวันที่ผู้ใช้เลือก
    df_filtered = df[df["Date"] == pd.to_datetime(selected_date)]

    # 🟡 แสดงตัวอย่างข้อมูล
    st.write("### ข้อมูล PM2.5 ที่เลือก:")
    st.dataframe(df_filtered.head())

    # 🟡 เตรียมข้อมูลและเทรนโมเดล
    if not df_filtered.empty:
        X, y, scaler = prepare_data(df)  # เทรนข้อมูลทั้งหมด
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write(f"### Data Prepared: {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")

        # 🟡 สร้างและ Train โมเดล LSTM
        model = build_lstm_model((X.shape[1], X.shape[2]))
        st.write("### Training Model...")
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

        # 🟡 ทำนายค่า PM2.5
        st.write("### Making Predictions...")
        X_selected = scaler.transform(df_filtered[[
            'Pressure_max', 'Pressure_min', 'Pressure_avg', 'Temp_max', 'Temp_min', 'Temp_avg', 
            'Humidity_max', 'Humidity_min', 'Humidity_avg', 'Precipitation', 'Sunshine', 'Evaporation', 
            'Wind_direct', 'Wind_speed']].values)  # เตรียมข้อมูลสำหรับทำนาย
        X_selected = np.expand_dims(X_selected, axis=0)  # เพิ่มมิติให้ตรงกับที่โมเดลต้องการ
        y_pred = model.predict(X_selected)
        y_pred_inv = scaler.inverse_transform(np.concatenate((X_selected[0, :, :-1], y_pred), axis=1))[:, -1]

        st.write(f"### Predicted PM2.5 for {selected_date}: {y_pred_inv[0]}")

        st.success("✅ Prediction Completed!")
    else:
        st.error("❌ ไม่มีข้อมูลในวันที่ที่เลือก กรุณาเลือกวันใหม่")

def main(): 
    page = st.selectbox("Select a page", ["PM25"])  # แก้ไขตรงนี้เป็น selectbox แทน sidebar
    if page == "PM25":
        st.title("PM25 Forecasting")
        PM25() 

if __name__ == "__main__":
    main()


