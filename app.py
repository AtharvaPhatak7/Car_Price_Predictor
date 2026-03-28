
import streamlit as st
import pickle
import pandas as pd
import numpy as np

pipe = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
df   = pickle.load(open('df.pkl', 'rb'))

st.title("🚗 Car Price Predictor")

company    = st.selectbox("Car Brand", sorted(df['company'].unique()))
name       = st.selectbox("Car Model", sorted(df[df['company'] == company]['name'].unique()))
year       = st.selectbox("Year", sorted(df['year'].unique(), reverse=True))
fuel_type  = st.selectbox("Fuel Type", df['fuel_type'].unique())
kms_driven = st.number_input("Kilometres Driven", min_value=100, max_value=500000, step=1000, value=30000)

if st.button("🔍 Predict Price"):
    input_df = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )
    prediction = pipe.predict(input_df)
    price = np.round(prediction[0], 2)
    st.success(f"💰 Estimated Resale Price: ₹ {price:,}")
