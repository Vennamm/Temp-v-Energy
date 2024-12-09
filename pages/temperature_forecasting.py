import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.decomposition import PCA
from scipy.stats import chi2, spearmanr
import plotly.express as px
import geopandas as gpd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings

def read_data(state):
    weather = pd.read_csv(f'collated_data/{state}.csv')
    weather['Date'] = pd.to_datetime(weather['Date'].astype(str), format='%Y%m')
    weather['Date'] = weather['Date'] + pd.offsets.MonthEnd(0)
    weather.set_index('Date', inplace=True)
    weather = weather.asfreq('ME')
    return weather, state

def seasonal_dc(weather, column, state, test_size=48):
    train_size = len(weather) - test_size
    train = weather[column][:train_size]
    test = weather[column][train_size:]
    stl = STL(train, seasonal=13, robust=True)
    result = stl.fit()
    
    resid_std = result.resid.std()
    lower_threshold = -3 * resid_std
    upper_threshold = 3 * resid_std
    outliers = (result.resid < lower_threshold) | (result.resid > upper_threshold)
    
    resid_interpolated = result.resid.copy()
    resid_interpolated[outliers] = np.nan
    resid_interpolated = pd.Series(resid_interpolated).interpolate()
    
    cleaned = result.trend + result.seasonal + resid_interpolated
    
    return train, test, cleaned, result

def forecast_weather(train, test, col, steps=None, seasonality=12, p=1, d=0, q=1, P=1, D=0, Q=1):
    sarimax_model = SARIMAX(train, 
                            order=(p, d, q),  # AR, I, MA orders
                            seasonal_order=(P, D, Q, seasonality),  # Seasonal AR, I, MA orders, and seasonality (12 months)
                            enforce_stationarity=False,
                            enforce_invertibility=False)

    sarimax_fitted = sarimax_model.fit(disp=False)
    if steps is None:
        steps = len(test)
    last_date = test.index[-1] if not test.empty else train.index[-1]
    forecast_dates = pd.date_range(start=test.index[0], periods=steps, freq='ME')
    forecast_ = sarimax_fitted.forecast(steps=steps)    

    forecast_series = pd.Series(data=forecast_, index=forecast_dates, name='Forecast')
    
    mae = mean_absolute_error(test, forecast_)
    rmse = np.sqrt(mean_squared_error(test, forecast_))
    
    return sarimax_fitted, forecast_series, mae, rmse

def plot_granger_causality(df, col1, col2, max_lag=12, axes=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_results = grangercausalitytests(df[[col1, col2]], maxlag=max_lag, verbose=False)
        test_results_2 = grangercausalitytests(df[[col2, col1]], maxlag=max_lag, verbose=False)
    lags = []
    f_scores_col1_to_col2 = []
    p_zero_points_col1_to_col2 = []
    f_scores_col2_to_col1 = []
    p_zero_points_col2_to_col1 = []
    for lag, result in test_results.items():
        f_score_col1_to_col2 = result[0]['ssr_ftest'][0]
        p_value_col1_to_col2 = result[0]['ssr_ftest'][1]
        f_scores_col1_to_col2.append(f_score_col1_to_col2)
        if p_value_col1_to_col2 < 0.05:
            p_zero_points_col1_to_col2.append((lag, f_score_col1_to_col2))
    for lag, result in test_results_2.items():
        f_score_col2_to_col1 = result[0]['ssr_ftest'][0]
        p_value_col2_to_col1 = result[0]['ssr_ftest'][1]
        lags.append(lag)
        f_scores_col2_to_col1.append(f_score_col2_to_col1)
        if p_value_col2_to_col1 < 0.05:
            p_zero_points_col2_to_col1.append((lag, f_score_col2_to_col1))
    f_scores_data = pd.DataFrame({
        'Lag': lags,
        f'{col1} → {col2} F-score': f_scores_col1_to_col2,
        f'{col2} → {col1} F-score': f_scores_col2_to_col1
    })
    

    # Create the figure
    fig = go.Figure()
    
    # Line plot for col1 → col2
    fig.add_trace(go.Scatter(
        x=f_scores_data['Lag'],
        y=f_scores_data[f'{col1} → {col2} F-score'],
        mode='lines',
        name=f'{col1} → {col2}',
        line=dict(color='blue')
    ))
    
    # Line plot for col2 → col1
    fig.add_trace(go.Scatter(
        x=f_scores_data['Lag'],
        y=f_scores_data[f'{col2} → {col1} F-score'],
        mode='lines',
        name=f'{col2} → {col1}',
        line=dict(color='orange')
    ))
    
    # Scatter plot for p_zero_points_col1_to_col2
    for lag, f_score in p_zero_points_col1_to_col2:
        fig.add_trace(go.Scatter(
            x=[lag],
            y=[f_score],
            mode='markers',
            name=f'{col1} → {col2} (p<0.05)' if lag == p_zero_points_col1_to_col2[0][0] else "",
            marker=dict(color='red', size=10),
            showlegend=False
        ))
    
    # Scatter plot for p_zero_points_col2_to_col1
    for lag, f_score in p_zero_points_col2_to_col1:
        fig.add_trace(go.Scatter(
            x=[lag],
            y=[f_score],
            mode='markers',
            name=f'{col2} → {col1} (p<0.05)' if lag == p_zero_points_col2_to_col1[0][0] else "",
            marker=dict(color='green', size=10),
            showlegend=False
        ))
    
    # Update the layout
    fig.update_layout(
        title=f'Granger Causality F-scores: {col1} vs {col2}',
        xaxis_title='Lag',
        yaxis_title='F-score',
        legend_title="Causality",
        template="plotly",
        showlegend=True,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    
    # Show the figure
    # fig.show()
    st.plotly_chart(fig, use_container_width=True)



def temperature_forecasting():
    st.title('Temperature Forecasting')

    state_list = [f.replace('.csv', '') for f in os.listdir('collated_data')]
    state = st.selectbox("Select a state:", state_list)
    weather_data, state_name = read_data(state)
    
    st.write(f"Displaying data for {state_name}:")
    

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input('Select Year', min_value=2019, max_value=2024, value=2019, step=1)

    with col2:
        month = st.number_input('Select Month', min_value=1, max_value=12, value=6, step=1)
    forecast_date = pd.to_datetime(f'{year}-{month:02d}-01') + pd.offsets.MonthEnd(0)
    
    test_size = 48
    train_data, test_data, cleaned_data, seasonal_result = seasonal_dc(weather_data, 'tavg', state_name, test_size=test_size)

    sarimax_fitted, forecast_series, mae, rmse = forecast_weather(train_data, test_data, 'tavg', steps=len(test_data))

    st.markdown(f"""
<div style="background-color: #F0F8FF; padding: 20px; border-radius: 15px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);">
    <h3 style="font-size: 22px; font-weight: bold; color: #003366; text-align: center;">Model Performance Metrics for {state_name}</h3>
    <div style="font-size: 18px; color: #003366; text-align: center;">
        <p><strong>Mean Absolute Error (MAE):</strong> <span style="color: #FF6347;">{mae:.2f} °F</span></p>
        <p><strong>Root Mean Squared Error (RMSE):</strong> <span style="color: #32CD32;">{rmse:.2f} °F</span></p>
    </div>
</div>
""", unsafe_allow_html=True)

    # Row 1: Predicted and Actual Temperatures
    st.subheader('Predicted and Actual Temperatures')
    train_end_date = train_data.index[-1]
    steps_ = (forecast_date.year - train_end_date.year) * 12 + (forecast_date.month - train_end_date.month)
    # st.write(steps_)
    output_series = sarimax_fitted.forecast(steps= steps_)

    if steps_ <=48:
        fc = output_series[output_series.index == forecast_date].values[0]
        ac = weather_data['tavg'][weather_data.index == forecast_date].values[0]
        st.markdown(f"""
    <div style="background-color: #D1E8FF; padding: 10px; border-radius: 10px; display: flex; justify-content: space-between;">
        <div style="font-size: 18px; font-weight: bold;">
            <strong>Forecasted Temperature:</strong> {fc:.2f} °F
        </div>
        <div style="font-size: 18px; font-weight: bold;">
            <strong>Actual Temperature:</strong> {ac:.2f} °F
        </div>
    </div>
    """, unsafe_allow_html=True)
    else:
        fc = output_series[output_series.index == forecast_date].values[0]
        st.markdown(f"""
    <div style="background-color: #D1E8FF; padding: 10px; border-radius: 10px; display: flex; justify-content: center;">
        <div style="font-size: 18px; font-weight: bold;">
            <strong>Forecasted Temperature:</strong> {fc:.2f} °F
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Row 2: Actual Data Trend + Forecast
    st.subheader('Actual Data Trend + Forecast')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data, mode='lines', name='Train Data'))
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data, mode='lines', name='Test Data', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=output_series.index, y=output_series, mode='lines', name='Forecast', line=dict(color='green')))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander('Advanced Options'):

        st.subheader("Granger Causality:")

        st.markdown("""Granger Causality checks where one variable causes the other variable in a timeseries. But the issue is that true causality is a hard thing to prove even with such a test.
        But it gives us insight on how each state behaves. The following graph shows if last month's column $\\rho=1$ causes effects in current month's another column.""")
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Will", options=weather_data.columns)
        with col2:
            var2 = st.selectbox("Cause", options=weather_data.columns)
            
        if (var1!=var2):
            plot_granger_causality(weather_data, var1, var2)
        
        
        # Row 3: Seasonal Decomposition
        st.subheader('Seasonal Decomposition')

        st.markdown("""For temperature forecasting, since there is already a very consistent seasonality and very little noise within the temperature data, it becomes less worth it to use the impact of another variable. 
        Feel free to look at the seasonal decomposition.""")
        
        fig_x = go.Figure()
        fig_x.add_trace(go.Scatter(x=seasonal_result.trend.index, y=seasonal_result.trend, mode='lines', name='Trend', line=dict(color='blue')))
        fig_x.add_trace(go.Scatter(x=seasonal_result.seasonal.index, y=seasonal_result.seasonal, mode='lines', name='Seasonal', line=dict(color='orange')))
        fig_x.add_trace(go.Scatter(x=seasonal_result.resid.index, y=seasonal_result.resid, mode='lines', name='Residual', line=dict(color='green')))

        fig_x.update_layout(
            title='Trend, Seasonal, and Residual Components',
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Components',
            template='plotly_dark'  # Optional: Adjust the theme
        )
        # fig.write_image("trend_seasonal_residual.png")
        
        st.plotly_chart(fig_x, use_container_width=True)
        
        # Row 4: ACF and PACF
        st.subheader('ACF and PACF')

        st.markdown("""Earlier, we check if a previous foreign variable had any impact on the current predictor variable. Now, we are looking if previous values of our predictor variable have any impact on our current values. 
        We need to look at the Auto-Correlation and Partial Auto-Correlation Functions to understand the seasonality of the temperature data so that we can train our model on it. 
        The graph shows that the functions repeat themselves after every 12 months approximately. That is to say that last year's temperature have an impact on current year's average temperature.""")
        
        fig_y, ax = plt.subplots(2, 1, figsize=(12, 6))
        plot_acf(train_data, lags=48, ax=ax[0])
        plot_pacf(train_data, lags=48, ax=ax[1])
        st.pyplot(fig_y)

        st.markdown("""We worked with a SARIMAX model with the common values as $p=1, d=0, q=1, P=1, D=0, Q=0,\\text{ and seasonality}=12$. In a complex scenario, the PACF and ACF graphs help us determine more than just the seasonality 
        and help us even reassess the remaining parameters of the SARIMAX model.""")

        st.subheader("LSTM Impact")
        st.markdown("""LSTM was trained to predict CDD and HDD based on lags 1, 3, 6 based on causality inferences from the Granger Causality tests. But the model performed just as decently as not having the causality dependencies. 
        The references to this can be found both on CDD-HDD Forecasting and Temp Forecasting notebooks on my [GitHub](https://github.com/Vennamm/Temp-v-Energy/tree/main).""")
    
    st.subheader('Most Recent Year in the Dataset')
    st.dataframe(weather_data.tail(12), use_container_width=True)
