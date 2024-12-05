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


state_to_region = {
    'Alabama': 'South', 'Alaska': 'West', 'Arizona': 'West', 'Arkansas': 'South',
    'California': 'West', 'Colorado': 'West', 'Connecticut': 'Northeast', 'Delaware': 'South',
    'Florida': 'South', 'Georgia': 'South', 'Hawaii': 'West', 'Idaho': 'West',
    'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Iowa': 'Midwest', 'Kansas': 'Midwest',
    'Kentucky': 'South', 'Louisiana': 'South', 'Maine': 'Northeast', 'Maryland': 'South',
    'Massachusetts': 'Northeast', 'Michigan': 'Midwest', 'Minnesota': 'Midwest', 'Mississippi': 'South',
    'Missouri': 'Midwest', 'Montana': 'West', 'Nebraska': 'Midwest', 'Nevada': 'West',
    'New Hampshire': 'Northeast', 'New Jersey': 'Northeast', 'New Mexico': 'West', 'New York': 'Northeast',
    'North Carolina': 'South', 'North Dakota': 'Midwest', 'Ohio': 'Midwest', 'Oklahoma': 'South',
    'Oregon': 'West', 'Pennsylvania': 'Northeast', 'Rhode Island': 'Northeast', 'South Carolina': 'South',
    'South Dakota': 'Midwest', 'Tennessee': 'South', 'Texas': 'South', 'Utah': 'West',
    'Vermont': 'Northeast', 'Virginia': 'South', 'Washington': 'West', 'West Virginia': 'South',
    'Wisconsin': 'Midwest', 'Wyoming': 'West'
}

season_mapping = {
    1: 'Winter', 2: 'Winter', 12: 'Winter', 
    3: 'Spring', 4: 'Spring', 5: 'Spring', 
    6: 'Summer', 7: 'Summer', 8: 'Summer', 
    9: 'Fall', 10: 'Fall', 11: 'Fall'
}


def create_frame(state):
    weather = pd.read_csv(f'collated_data/{state}.csv')
    energy = pd.read_csv(f'power_consumption/{state}.csv')
    energy.set_index('Year', inplace=True)
    weather['Year'] = weather.Date.astype(str).str[:4].astype(int)
    weatherly = weather.groupby('Year').mean().drop(columns=['Date'])
    weathergy = pd.concat([weatherly[['tavg','cdd','hdd','tmax','tmin']], energy], axis=1)
    return weathergy

# Function to read and preprocess data
def read_data(state):
    weather = pd.read_csv(f'collated_data/{state}.csv')
    weather['Date'] = pd.to_datetime(weather['Date'].astype(str), format='%Y%m')
    weather['Date'] = weather['Date'] + pd.offsets.MonthEnd(0)
    weather.set_index('Date', inplace=True)
    weather = weather.asfreq('ME')
    return weather, state

# Function for seasonal decomposition and outlier removal
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

# SARIMAX forecasting function
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

def compute_spearmanr(weather_stats, consumption_stats):
    results = []
    for state in os.listdir('collated_data'):
        state = state.replace('.csv','')
        weathergy = create_frame(state)
        row = {'state': state}
        for weather in weather_stats:
            for consumption in consumption_stats:
                corr, _ = spearmanr(weathergy[weather], weathergy[consumption])
                row[f'{weather}-{consumption}'] = corr

        results.append(row)
    return pd.DataFrame(results)

def add_ellipse(ax, data, color):
    # Covariance matrix and mean calculation
    cov_matrix = np.cov(data.T)
    mean = data.mean(axis=0)
    
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    
    # Angle of rotation
    angle = np.arctan2(*eigvecs[:, 0][::-1])
    angle = np.degrees(angle)
    
    # Ellipse axes lengths (scaled by chi-squared distribution for 95% confidence)
    chi_square_val = chi2.ppf(0.95, 2)
    axis_length = np.sqrt(eigvals * chi_square_val)
    
    # Create ellipse
    ell = plt.matplotlib.patches.Ellipse(mean, width=axis_length[0]*2, height=axis_length[1]*2, 
                                         angle=angle, color=color, fill=False, linewidth=2)
    ax.add_patch(ell)

def plot_pca_with_ellipses(corr_df, ellipse=0.95):
    corr_matrix = corr_df.set_index('state').transpose()
    corr_matrix_values = corr_matrix.values
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(corr_matrix_values.T)
    
    pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'], index=corr_matrix.columns)
    pca_df['Region'] = pca_df.index.map(state_to_region)
    
    # Create scatter plot using Plotly
    fig = go.Figure()

    # Add scatter points
    for region in pca_df['Region'].unique():
        region_data = pca_df[pca_df['Region'] == region]
        region_color = sns.color_palette('Set1')[list(pca_df['Region'].unique()).index(region)]
        region_color_str = f'rgb({int(region_color[0]*255)},{int(region_color[1]*255)},{int(region_color[2]*255)})'  # Convert to rgb()

        fig.add_trace(go.Scatter(
            x=region_data['PCA1'], 
            y=region_data['PCA2'], 
            mode='markers',
            name=region,
            marker=dict(size=10, color=region_color_str),
        ))

    # Add ellipses for each region
    for region in pca_df['Region'].unique():
        region_data = pca_df[pca_df['Region'] == region]
        
        # Covariance matrix and mean calculation
        cov_matrix = np.cov(region_data[['PCA1', 'PCA2']].T)
        mean = region_data[['PCA1', 'PCA2']].mean(axis=0)
        
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        
        # Angle of rotation
        angle = np.arctan2(*eigvecs[:, 0][::-1])
        angle = np.degrees(angle)
        
        # Ellipse axes lengths (scaled by chi-squared distribution for 95% confidence)
        chi_square_val = chi2.ppf(ellipse, 2)
        axis_length = np.sqrt(eigvals * chi_square_val)

        # Parametric equation for the ellipse path
        t = np.linspace(0, 2 * np.pi, 100)
        x_ellipse = mean[0] + axis_length[0] * np.cos(t) * np.cos(np.radians(angle)) - axis_length[1] * np.sin(t) * np.sin(np.radians(angle))
        y_ellipse = mean[1] + axis_length[0] * np.cos(t) * np.sin(np.radians(angle)) + axis_length[1] * np.sin(t) * np.cos(np.radians(angle))
        
        # Convert color to RGB format
        region_color = sns.color_palette('Set1')[list(pca_df['Region'].unique()).index(region)]
        region_color_str = f'rgb({int(region_color[0]*255)},{int(region_color[1]*255)},{int(region_color[2]*255)})'

        # Create ellipse shape using path
        fig.add_trace(go.Scatter(
            x=x_ellipse,
            y=y_ellipse,
            mode='lines',
            line=dict(color=region_color_str, width=2),
            fill='toself',
            opacity=0.3,
            name=f'Ellipse {region}'
        ))

    # Update layout
    fig.update_layout(
        title="PCA Plot with Clustering (Ellipses by Region)",
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        showlegend=True,
        height=1000
    )

    st.plotly_chart(fig, use_container_width=True)

def visualize_corr(corr_df):
    melt_df = corr_df.melt(id_vars='state', var_name='Pair', value_name='Correlation')

    corr_matrix = corr_df.set_index('state').transpose()
    pair_correlation_strength = corr_matrix.abs().mean(axis=0) 
    sorted_pairs = pair_correlation_strength.sort_values(ascending=False).index

    reordered_corr = corr_matrix[sorted_pairs]

    plt.figure(figsize=(15, 10))
    sns.heatmap(
        reordered_corr,
        cmap=sns.diverging_palette(250, 10, as_cmap=True),
        center=0,
        annot=False,
        cbar_kws={'label': 'Spearman Correlation'},
    )
    plt.title('Weather-Consumption Pairs Reordered by Correlation')
    plt.ylabel('Weather-Consumption Pairs')
    plt.xlabel('State')
    plt.show()
    st.pyplot(plt)

def plot_pca_choropleth_on_map(corr_df, geojson_path, n_clusters=4):

    corr_matrix = corr_df.set_index('state').transpose()  
    corr_matrix_values = corr_matrix.values
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(corr_matrix_values.T) 
    
    pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'], index=corr_matrix.columns)
    
    
    
    if n_clusters==4:
        random_state = 132
    else:
        random_state=133
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        # pca_df['Cluster'] = gmm.fit_predict(pca_df[['PCA1', 'PCA2']])
        # hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        # pca_df['Cluster'] = hierarchical.fit_predict(pca_df[['PCA1', 'PCA2']])
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        pca_df['Cluster'] = kmeans.fit_predict(pca_df[['PCA1', 'PCA2']]) 
    
    state_cluster_mapping = pca_df['Cluster'].to_dict()
    
    geo_df = gpd.read_file(geojson_path)
    
    
    geo_df['Cluster'] = geo_df['name'].map(state_cluster_mapping) 
    
    fig = px.choropleth(geo_df,
                        geojson=geo_df.geometry,
                        locations=geo_df.index,
                        color='Cluster',
                        hover_name='name',
                        color_discrete_sequence=px.colors.qualitative.Set1, 
                        labels={'Cluster': 'Cluster ID'},
                        title="Cluster-Based PCA Visualization on U.S. States")
    
   
    # fig.update_geos(fitbounds="locations", visible=False)
    fig.update_traces(showlegend=False)
    fig.update_layout(
        title="Cluster-Based PCA Visualization on U.S. States", 
        title_x=0.25,
        title_font_size=20,
        showlegend=False,  
        geo=dict(
            visible=False,
            fitbounds="locations"
        ),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},  
        height=600,
        width=1000,
        dragmode='zoom'
        # scrollZoom=False
    )
    # fig.show()
    st.plotly_chart(fig, use_container_width=True)

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

def create_frame2(state, target_column):
    importance_df = pd.read_csv('feature_importance_results.csv')
    importance_df = importance_df[(importance_df['State'] == state) & (importance_df['Target Column'] == target_column)]
    importance_df = importance_df[['Feature','Importance']]
    
    return importance_df, state, target_column

def aggregate_and_rank(df, state, target_column):
    season_map = {
        "Winter": ["cdd_Winter", "hdd_Winter"],
        "Spring": ["cdd_Spring", "hdd_Spring"],
        "Summer": ["cdd_Summer", "hdd_Summer"],
        "Fall": ["cdd_Fall", "hdd_Fall"]
    }
    season_totals = {season: df[df["Feature"].isin(features)]["Importance"].sum() for season, features in season_map.items()}
    season_df = pd.DataFrame(list(season_totals.items()), columns=["Season", "Contribution"]).sort_values(by="Contribution", ascending=False)

    top_contributor = season_df.iloc[0]
    second_contributor = season_df.iloc[1] if len(season_df) > 1 else None

    co_dominance_threshold = 0.05 

    TEXT = ''

    if second_contributor is not None:
        diff = top_contributor["Contribution"] - second_contributor["Contribution"]

        if diff <= co_dominance_threshold:
            TEXT = f"For {state}, {top_contributor['Season']} and {second_contributor['Season']} contribute the most"
        else:
            TEXT = f"For {state}, {top_contributor['Season']} contributes the most to the {target_column.lower()}."

    else:
        TEXT = f"For {state}, {top_contributor['Season']} is the only major contributor to the {target_column.lower()}."

    fig1 = px.pie(season_df, names='Season', values='Contribution', title="Seasonal Energy Consumption Contributions")

    df["Category"] = df["Feature"].apply(lambda x: "Hot " + x.split("_")[1] if "cdd" in x else "Cold " + x.split("_")[1])

    min_importance_blue = df[df['Category'].str.startswith('Cold')]['Importance'].min()
    max_importance_blue = df[df['Category'].str.startswith('Cold')]['Importance'].max()
    min_importance_red = df[df['Category'].str.startswith('Hot')]['Importance'].min()
    max_importance_red = df[df['Category'].str.startswith('Hot')]['Importance'].max()
    min_color_value = 100
    
    df["Color"] = df.apply(
        lambda row: f"rgb({min(255, int((row['Importance'] - min_importance_red) / (max_importance_red - min_importance_red) * (255 - min_color_value) + min_color_value))}, 0, 0)"
        if "Hot" in row["Category"]
        else f"rgb(0, 0, {min(255, int((row['Importance'] - min_importance_blue) / (max_importance_blue - min_importance_blue) * (255 - min_color_value) + min_color_value))})", axis=1
    )
    
    df_sorted = df.sort_values(by="Importance", ascending=False)

    
    fig2 = go.Figure()

    # Add Hot features (Red gradient)
    fig2.add_trace(go.Bar(
        x=df_sorted["Category"],  # Feature names
        y=df_sorted["Importance"],  # Importance values
        # text=df_sorted["Category"],  # Show Category as text
        hoverinfo="text",  # Show category on hover
        marker=dict(
            color=df_sorted["Color"],  # Color based on importance and category
        )
    ))

    fig2.update_layout(
        title="Individual Contributions by Feature",
        xaxis_title="Feature",
        yaxis_title="Importance",
        showlegend=False
    )

    # Show both plots
    # fig1.show()
    # fig2.show()
    st.markdown(TEXT)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    

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
    
    st.subheader('Most Recent Year in the Dataset')
    st.dataframe(weather_data.tail(12), use_container_width=True)

def weather_energy_cluster():
    st.title("Weather Energy Clustering")

    weather_stats = ['tavg', 'cdd', 'hdd']
    consumption_stats = ['Residential sector', 'Commercial sector', 'Industrial sector', 'Transportation sector', 'Total consumption']
    corr_df = compute_spearmanr(weather_stats, consumption_stats)
    
    st.subheader("Cluster-Based PCA Visualization on Map")
    st.markdown("""
There is a lot of process behind this. But let me make it straightforward:
- We analyzed the relationships between 'Average Temperature', 'Cooling Degree Days', and 'Heating Degree Days' to see their impact on all types of energy consumption (Residential, Commercial, Industrial, Transportation, and all combined).
- We captured similar behaviors (energy consumption vs temperature) between the states and projected those behaviors onto a map, by dividing these behaviors into different groups.
- And voila, these behaviors formed these interesting borders, indicating that there is a regional pattern in energy consumption.
- I ideally would love the divided groups to be 4, but I would love for you to play around and see what happens if you split these behaviors into 3, 5, or 6 groups. :)
""")
    geojson_path = 'us-states.json'

    n_clusters = st.selectbox("Select Number of Groups", options=[3, 4, 5, 6], index=0)
    plot_pca_choropleth_on_map(corr_df, geojson_path, n_clusters)

    st.subheader("Season-Energy Contributions")
    st.markdown("""Understanding which seasons contribute the highest to which sector for each state.""")

    state_list = [f.replace('.csv', '') for f in os.listdir('collated_data')]
    
    sectors = ['Residential', 'Commercial', 'Industrial', 'Transportation', 'Total Consumption']
    col1, col2 = st.columns(2)
    with col1:
        state_name = st.selectbox("Select State:", options=state_list)
    with col2:
        col_name = st.selectbox("Select Sector:", options=sectors)

    if col_name in ['Residential', 'Commercial', 'Industrial', 'Transportation']:
        col_name = col_name + ' sector'
    else:
        col_name = 'Total consumption'

    importance_df, state_name, target_column = create_frame2(state_name, col_name)

    aggregate_and_rank(importance_df, state_name, target_column)
    
    st.subheader("Weather-Energy Correlation Matrix")   
    
    visualize_corr(corr_df)
    st.subheader("PCA Clustering Visualization")

    ellipse_rad = st.slider("Select Confidence Level", 0.5, 0.99, 0.95, 0.01)
    plot_pca_with_ellipses(corr_df, ellipse_rad)

    

st.set_page_config(layout="wide")

section = st.sidebar.radio("Select a section", ["Documentation", "Temperature Forecasting", "Weather-Energy Relationship"])

if section == "Documentation":
    st.title("Documentation")
    st.write("This section will contain documentation. (Under construction!)")

elif section == "Temperature Forecasting":
    temperature_forecasting()

elif section == "Weather-Energy Relationship":
    weather_energy_cluster()
