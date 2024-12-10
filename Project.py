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
import requests
import base64
import time
from datetime import datetime, timedelta


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

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def get_season_color(TEXT):
    color = "#f9f9f9"
    if "Spring" in TEXT:
        color = "#e0f7fa"
    elif "Summer" in TEXT:
        color = "#ffeb3b"
    elif "Winter" in TEXT:
        color = "#b3e5fc" 
    elif "Fall" in TEXT:
        color = "#ff7043"
    return color

@st.cache
def create_frame(state):
    weather = pd.read_csv(f'collated_data/{state}.csv')
    energy = pd.read_csv(f'power_consumption/{state}.csv')
    energy.set_index('Year', inplace=True)
    weather['Year'] = weather.Date.astype(str).str[:4].astype(int)
    weatherly = weather.groupby('Year').mean().drop(columns=['Date'])
    weathergy = pd.concat([weatherly[['tavg','cdd','hdd','tmax','tmin']], energy], axis=1)
    return weathergy

@st.cache
def read_data(state):
    weather = pd.read_csv(f'collated_data/{state}.csv')
    weather['Date'] = pd.to_datetime(weather['Date'].astype(str), format='%Y%m')
    weather['Date'] = weather['Date'] + pd.offsets.MonthEnd(0)
    weather.set_index('Date', inplace=True)
    weather = weather.asfreq('ME')
    return weather, state

@st.cache
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

@st.cache
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

@st.cache
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

@st.cache
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

@st.cache
def plot_pca_with_ellipses(corr_df, ellipse=0.95):
    corr_matrix = corr_df.set_index('state').transpose()
    corr_matrix_values = corr_matrix.values
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(corr_matrix_values.T)
    
    pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'], index=corr_matrix.columns)
    pca_df['Region'] = pca_df.index.map(state_to_region)

    explained_variance = pca.explained_variance_ratio_
    
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
        title=f"PCA Plot with Clustering (Ellipses by Region) - Explained Variance: {round(explained_variance.sum(),4)*100}%",
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        showlegend=True,
        height=1000
    )

    st.plotly_chart(fig, use_container_width=True)

@st.cache
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

@st.cache
def plot_pca_choropleth_on_map(corr_df, geojson_path, n_clusters=4):

    corr_matrix = corr_df.set_index('state').transpose()  
    corr_matrix_values = corr_matrix.values
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(corr_matrix_values.T) 
    
    pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'], index=corr_matrix.columns)
    
    explained_variance = pca.explained_variance_ratio_
    
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

@st.cache
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


@st.cache
def create_frame3(state, target_column):
    importance_df_m = pd.read_csv('feature_importances/feature_importance_results_DecisionTree.csv')
    importance_df_m = importance_df_m[(importance_df_m['State'] == state) & (importance_df_m['Target Column'] == target_column)]
    importance_df_m = importance_df_m[['Feature', 'Importance']]

    return importance_df_m, state, target_column

@st.cache
def aggregate_and_rank(df_m, state, target_column):
    # season_map = {
    #     "Winter": ["cdd_Winter", "hdd_Winter"],
    #     "Spring": ["cdd_Spring", "hdd_Spring"],
    #     "Summer": ["cdd_Summer", "hdd_Summer"],
    #     "Fall": ["cdd_Fall", "hdd_Fall"]
    # }

    season_map = {
        "Winter": ["December", "January", "February"],
        "Spring": ["March", "April", "May"],
        "Summer": ["June", "July", "August"],
        "Fall": ["September", "October", "November"],
    }

    energy_rates = pd.read_csv(f'power_consumption/{state}.csv')
    
    fig4 = px.line(
        energy_rates,
        x="Year",
        y=target_column,
        title=f"Energy Consumption over years - {target_column.title()}",
        labels={"Year": "Year", target_column: "Energy Consumption"},
    )

    month_totals = {month: df_m[df_m["Feature"].str.contains(f"_{month}")]["Importance"].sum() for month in df_m["Feature"].str.split("_").str[1].unique()}
    month_df = pd.DataFrame(list(month_totals.items()), columns=["Month", "Contribution"]).sort_values(by="Contribution", ascending=False)

    season_totals = {season: month_df[month_df["Month"].isin(months)]["Contribution"].sum() for season, months in season_map.items()}
    season_df = pd.DataFrame(list(season_totals.items()), columns=["Season", "Contribution"]).sort_values(by="Contribution", ascending=False)
    
    df = df_m.copy()
    df["Type"] = df["Feature"].str.split("_").str[0]
    df["Month"] = df["Feature"].str.split("_").str[1]
    df["Season"] = df["Month"].map({month: season for season, months in season_map.items() for month in months})
    df["Feature"] = df["Type"] + "_" + df["Season"]
    df = df.groupby("Feature", as_index=False)["Importance"].sum()
    
    
    top_contributor = season_df.iloc[0]
    second_contributor = season_df.iloc[1] if len(season_df) > 1 else None

    top_contributor_m = month_df.iloc[0]
    second_contributor_m = month_df.iloc[1] if len(month_df) > 1 else None

    co_dominance_threshold_s = 0.1 
    co_dominance_threshold = 0.05

    TEXT = ''
    TEXT_m = ''

    if second_contributor_m is not None:
        diff_m = top_contributor_m["Contribution"] - second_contributor_m["Contribution"]
        if diff_m <= co_dominance_threshold:
            TEXT_m = f"In a monthly perspective, {top_contributor_m['Month']} and {second_contributor_m['Month']} contribute the most to the {target_column.lower()}."
        else:
            TEXT_m = f"In a monthly perspective, {top_contributor_m['Month']} contributes the most to the {target_column.lower()}."
    else:
        TEXT_m = f"In a monthly perspective, {top_contributor_m['Month']} is the major contributor to the {target_column.lower()}."

    if second_contributor is not None:
        diff = top_contributor["Contribution"] - second_contributor["Contribution"]

        if diff <= co_dominance_threshold_s:
            TEXT = f"For {state}, {top_contributor['Season']} and {second_contributor['Season']} contribute the most to the {target_column.lower()}."
        else:
            TEXT = f"For {state}, {top_contributor['Season']} contributes the most to the {target_column.lower()}."

    else:
        TEXT = f"For {state}, {top_contributor['Season']} is the only major contributor to the {target_column.lower()}."

    fig1 = px.pie(season_df, names='Season', values='Contribution', title=f"Seasonal Contributions to Energy Expenditure - {target_column.title()}")
    
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

    df_m["Month"] = df_m["Feature"].str.split("_").str[1]
    df_m["Category"] = df_m["Feature"].str.split("_").str[0]

    pivot_df = df_m.pivot_table(
        index="Month",
        columns="Category",
        values="Importance",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    months_order = [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ]
    pivot_df["Month"] = pd.Categorical(pivot_df["Month"], categories=months_order, ordered=True)
    pivot_df = pivot_df.sort_values("Month")
    fig3 = go.Figure()

    fig3.add_trace(go.Bar(
        x=pivot_df["Month"],
        y=pivot_df["cdd"],
        name="Hot Days of the Month",
        marker_color="red"
    ))

    fig3.add_trace(go.Bar(
        x=pivot_df["Month"],
        y=pivot_df["hdd"],
        name="Cold Days of the Month",
        marker_color="blue"
    ))

    fig3.update_layout(
        barmode="stack",
        title=f"Monthly Contributions to Energy Expenditure - {target_column.title()}",
        xaxis_title="Month",
        yaxis_title="Contribution",
        legend_title="Category"
    )
    
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
        title=f"Energy Expenditure by Temperature and Season - {target_column.title()}",
        xaxis_title="Season",
        yaxis_title="Importance",
        showlegend=False
    )


    # Show both plots
    # fig1.show()
    # fig2.show()
    season_color = get_season_color(TEXT)
    st.markdown(f"""
    <div style="
        background-color: {season_color}; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #ddd; 
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); 
        margin-bottom: 20px;
        text-align: center; 
        display: block;
        width: 100%; 
        height: auto;
        box-sizing: border-box;
    ">
        <p style="font-size: 16px; line-height: 1.6; color: #333; font-family: Arial, sans-serif; font-weight: bold; margin: 0; padding: 0;">
            {TEXT} {TEXT_m}
        </p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.plotly_chart(fig4, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)

def increment_month():
    if st.session_state.selected_month_index < len(months) - 1:
        st.session_state.selected_month_index += 1
    else:
        st.session_state.selected_month_index = 0 

def decrement_month():
    if st.session_state.selected_month_index > 0:
        st.session_state.selected_month_index -= 1
    else:
        st.session_state.selected_month_index = len(months) - 1

def temperature_forecasting():
    # st.title('Temperature Forecasting')

    

    if "selected_month_index" not in st.session_state:
        st.session_state.selected_month_index = 5
    
    state_list = [f.replace('.csv', '') for f in os.listdir('collated_data')]
    state = st.selectbox("Select a state:", state_list)
    weather_data, state_name = read_data(state)
    
    st.write(f"Displaying data for {state_name}:")
    

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input('Select Year', min_value=2019, max_value=2024, value=2019, step=1)

    with col2:
        with st.container():
            
            col1_, col2_, col3_ = st.columns([1, 4, 1])
            with col1_:
                if st.button("◀"):
                    decrement_month()
            with col2_:
                st.markdown(f"<div style='text-align: center; border: 2px solid #0099ff; padding: 10px; border-radius: 10px;'>
                <h3 style='font-weight: bold;'>{months[st.session_state.selected_month_index]}</h3>
                </div>", unsafe_allow_html=True)
                
            with col3_:
                if st.button("▶"):
                    increment_month()
            
        # month = st.number_input('Select Month', min_value=1, max_value=12, value=6, step=1)
        month = st.session_state.selected_month_index
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
        But it gives us insight on how each state behaves. The following graph shows if last month's column $\\rho=1$ causes effects in current month's another column. The values where you see a blue or red point means 
        that we can say with a 95% confidence (p<0.05) that there is Granger Causality (again, not true causality).
        """)

        st.markdown("                                          ")
        col1, col2, col3, col4, col5 = st.columns([1, 6, 1, 6, 2])
        with col1:
            st.markdown("<div style='text-align: right; font-weight: bold;'>Will</div>", unsafe_allow_html=True)
        
        with col2:
            var1 = st.selectbox("", options=['tavg', 'cdd', 'hdd'], index=0, label_visibility="collapsed")
        
        with col3:
            st.markdown("<div style='text-align: center; font-weight: bold;'>Cause</div>", unsafe_allow_html=True)
        
        with col4:
            var2 = st.selectbox("", options=['tavg', 'cdd', 'hdd'], index=1, label_visibility="collapsed")
        
        with col5:
            st.markdown("<div style='text-align: left; font-weight: bold;'>or vice-versa?</div>", unsafe_allow_html=True)
        
        var3_selected = ['tavg', 'cdd', 'hdd'].index(var1)
        if (var1 == var2):
            st.warning(f"To see if past values of {var1} has impact on its current value, you need to look at the Auto-Correlation and Partial Auto-Correlation Graphs (they are below)")
        if (var1 != var2):
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
        For example, we need to look at the Auto-Correlation and Partial Auto-Correlation Functions to understand the seasonality of the temperature data so that we can train our model on it. 
        And the graph shows that the functions repeat themselves after every 12 months approximately. That is to say that last year's temperature has an impact on current year's average temperature.""")

        var3 = st.selectbox("Will the variable cause itself?", options=['tavg', 'cdd', 'hdd'], index=var3_selected)
        
        fig_y, ax = plt.subplots(2, 1, figsize=(12, 6))
        plot_acf(weather_data[var3], lags=48, ax=ax[0])
        plot_pacf(weather_data[var3], lags=48, ax=ax[1])
        st.pyplot(fig_y)

        st.markdown("""We worked with a SARIMAX model with the common values as $p=1, d=0, q=1, P=1, D=0, Q=0,\\text{ and seasonality}=12$. In a complex scenario, the PACF and ACF graphs help us determine more than just the seasonality 
        and help us even reassess the remaining parameters of the SARIMAX model.""")

        st.subheader("LSTM Impact")
        st.markdown("""LSTM was trained to predict CDD and HDD based on lags 1, 3, 6 based on causality inferences from the Granger Causality tests. But the model performed just as decently as not having the causality dependencies. 
        The references to this can be found both on CDD-HDD Forecasting and Temp Forecasting notebooks on my [GitHub](https://github.com/Vennamm/Temp-v-Energy/tree/main).""")
    
    st.subheader('Most Recent Year in the Dataset')
    st.dataframe(weather_data.tail(12), use_container_width=True)


def weather_energy_cluster():
    # st.title("Weather Energy Clustering")

    weather_stats = ['tavg', 'cdd', 'hdd']
    consumption_stats = ['Residential sector', 'Commercial sector', 'Industrial sector', 'Transportation sector', 'Total consumption']
    corr_df = compute_spearmanr(weather_stats, consumption_stats)

    st.subheader("Season-Energy Contributions")
    st.markdown("""Understanding which seasons contribute the highest to which sector for each state.""")

    state_list = [f.replace('.csv', '') for f in os.listdir('collated_data')]
    state_list.sort()
    
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

    importance_df_m, state_name, target_column = create_frame3(state_name, col_name)
    
    aggregate_and_rank(importance_df_m, state_name, target_column)
    
    st.subheader("Cluster-Based PCA Visualization on Map")
    st.markdown("""
As you can see from the above visualization, each state behaves differently. But there is a pattern within the states. I collated all of these patterns and put them on a map. But how?

There is a lot of process behind this. But let me make it straightforward:
- I analyzed the relationships between 'Average Temperature', 'Cooling Degree Days', and 'Heating Degree Days' to see their impact on all types of energy consumption (Residential, Commercial, Industrial, Transportation, and all combined).
- And then, I captured similar behaviors (energy consumption vs temperature) between the states and projected those behaviors onto a map, by dividing these behaviors into different groups.
- And voila, these behaviors formed these interesting borders, indicating a similar pattern in energy consumption, as I said earlier.
- I ideally would love the divided groups to be 4, but I would love for you to play around and see what happens if you split these behaviors into 3, 5, or 6 groups. :)
""")
    geojson_path = 'us-states.json'

    n_clusters = st.selectbox("Select Number of Groups", options=[3, 4, 5, 6], index=1)
    plot_pca_choropleth_on_map(corr_df, geojson_path, n_clusters)

    
    
    with st.expander('Advanced Options'):
        st.subheader("Weather-Energy Correlation Matrix")   
        st.markdown("""
    Well, how do I explain this? You see the really cool map at the beginning? So for that to happen, it was a two-step process. 
    - **First step**: I checked the Spearman Correlation for each state between each weather stat (Average Temperature, CDD, and HDD) and all the sectors. Now, that is a lot of columns versus a lot of states, and I did not want to manually read through each one of them to understand the relationships.
    - **Second step**: I created a pretty heatmap of all the correlations, but I also sorted them by their unusual relationships. That is why you see all the reds and blues sticking to each other. Sometimes, the correlation heatmap does 
    something stupid and does not load, please reload. I am still trying to figure out what is wrong with this app.
        """)
        visualize_corr(corr_df)
        st.subheader("PCA Clustering Visualization")

        st.markdown("""
    Now that we had our pretty heatmap, there's still a lot to collect and create similar behaviors out of.
    - **First step**: I reduced the dimensions using our good old PCA (Principal Component Analysis) dimension reduction. And it worked like a charm because most of the columns do not explain any variance for a certain state (for example, CDD_Winter usually does not do much for some states, as you won’t require cooling in Winters for most states).
    - **Second step**: Then we started seeing these pretty little patterns in our principal components. Now, how do I put this all on a map? The answer: kNN (k-Nearest Neighbors).
    - **Third step**: We ran a kNN on the resulting PCA components to assign a cluster to each state, and then we mapped them all on, well, the map. This helped us see patterns on how each state behaved.
        """)
        ellipse_rad = st.slider("Select Confidence Level", 0.5, 0.99, 0.95, 0.01)
        plot_pca_with_ellipses(corr_df, ellipse_rad)

    

st.set_page_config(page_title="Weathergy", page_icon='https://media.istockphoto.com/id/1337173750/photo/solar-and-wind-power.jpg?s=612x612&w=0&k=20&c=krNUQVFMq4DDPDvhKhW4SwL06NlmZ7dcHWWGDsxZzKI=', layout="wide")

st.markdown(
    """
    <style>
    .centered-title {
        font-size: 40px;
        font-weight: bold;
        color: #003366;
        text-align: center;
        margin-bottom: 20px;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="centered-title">Weather-Energy Relationship Analysis</div>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    div.stTabs button {
        flex-grow: 1; 
        margin: 0 5px;
    }
    div.stTabs button div {
        justify-content: center; 
    }
    </style>
    """,
    unsafe_allow_html=True
)
# section = st.sidebar.radio("Select a section", ["Documentation", "Temperature Forecasting", "Weather-Energy Relationship"])
t1, t2, t3, t4 = st.tabs(["Documentation", "Temperature Forecasting", "Weather-Energy Relationship", "Future Goals"])
with t1:
    st.markdown("""
    # **Weather-Energy Relationship Analysis and Forecasting**
    This project explores the relationships between weather patterns (Average Temperature, Cooling Degree Days (CDD), Heating Degree Days (HDD)) and energy consumption across different sectors (Residential, Commercial, Industrial, Transportation, and Total Consumption). It uses various data science techniques, such as Principal Component Analysis (PCA), Random Forests, and XGBoost, to create insights into regional energy consumption patterns based on weather factors. The analysis is visualized using interactive maps, graphs, and charts on a Streamlit-based app.
    """)
    
    # Sources section in an expander
    with st.expander("### **Sources**"):
        st.markdown("""
        - [**Weather Data**](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/): This is official weather data from National Centers for Environmental Information. **Granularity:** Monthly, State. 
        - [**Energy Data**](https://www.eia.gov/state/seds/seds-data-complete.php?sid=MI#Keystatisticsrankings): This is the energy consumption data sourced from U.S. Energy Information Administration. **Granularity:** Yearly, State.
        - [**GeoJSON for Mapping**](https://github.com/PublicaMundi/MappingAPI/blob/master/data/geojson/us-states.json): MappingAPI's GitHub Repository.
        - There are a few other sources that have not been used but have potential to explore: [**Monthly Energy Consumption - Nationwide**](https://www.eia.gov/totalenergy/data/browser/index.php?tbl=T01.01#/?f=M&start=199701&end=202407&charted=6-7-14-1-2-3-4-8-11-12-13), [**Wildfire Data**](https://www.nifc.gov/fire-information/statistics/wildfires)
        """)
    
    # Section 1: Temperature Forecasting in an expander
    with st.expander("### **Section 1: Temperature Forecasting**"):
        st.markdown("""
        - **Objective**: Forecast average temperatures using time-series analysis. **Forecastable Years**: 2019 to 2024.
        - **Instructions**: Select Month, Year and State. The forecast temperature and actual temperature should show up. 
        - **Model**: SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors).
        - **Process**: 
            - **Data Preparation**: Loading temperature data for different states.
            - **Granger Causality**: Checking dependencies between temperature and other weather factors.
            - **Seasonal Decomposition**: Breaking down the temperature data into trend, seasonal, and residual components.
            - **Forecasting**: Using SARIMAX to predict future temperatures.
            - **Visualization**: Graphs displaying predicted and actual temperatures, model performance metrics, and seasonal components.
        - This model does not predict further than 2024 for two reasons:
            - **Error Propagation**: Since the training data is limited to dates until 2018, the model cannot perform well on data in the longer future. Six years is a stretch, as the error propagates with no new information.
            - **Lack of Validation**: This app was created in November, 2024. The predictive power of this app cannot be validated without future data.
            - **Room for Improvement**: Our source is constantly updated, but currently, the data is being fed from my [GitHub](https://github.com/Vennamm/Temp-v-Energy/tree/main/collated_data). We can get information straight from the source while simulatenously updating our GitHub.
        """)
    
    # Section 2: Weather-Energy Relationships in an expander
    with st.expander("### **Section 2: Weather-Energy Relationships**"):
        st.markdown("""
        - **Objective**: Investigate how temperature (CDD, HDD, and Average Temperature) influences energy consumption in various sectors across different states.
        - **Instructions**: 
            - **Season-Energy Contributions**: Select State and Consumption Sector to see the seasonal and monthly contribution to the energy consumption.
            - **Cluster-Based PCA Visualization**: Select the number of groups to separate states based on their energy consumption behaviors in different weathers.
        - **Techniques**:
            - **Principal Component Analysis (PCA)**: Used to reduce the dimensionality of the data and identify key components driving energy consumption.
            - **Clustering**: k-Nearest Neighbors (kNN) clustering algorithm used to assign states to different groups based on their behavior in energy consumption related to weather patterns.
            - **Feature Importance**: Random Forest, XGBoost and Decision Tree Regression models were trained to determine which weather-related features (e.g., CDD, HDD) are the most important predictors for energy consumption in each sector.
            - **Model Selection**: Decision Tree was chosen for three reasons - **deterministic** (not using random state), **robust** (less sensitive to noise when addressed with cross-validation) and **interpretable** (Feature Importances are extracted).
    
        - **Process**:
            1. **Data Aggregation**: Collating weather data (CDD, HDD) and energy consumption data (Residential, Commercial, Industrial, etc.) for each state.
            2. **PCA and Clustering**: Reducing dimensionality and grouping states based on similar energy behaviors.
            3. **Visualization**: 
                - Cluster-based PCA visualizations are shown on an interactive map with option to edit the number of clusters.
                - Pie Chart showing the seasonal contributions for a selected sector and state.
                - Line Graph showing the energy consumption of the sector over the years.
                - Two Bar Charts: 
                    - Seasonal Contribution like the pie chart, but also which parts of the season (Hot and Cold days of each season).
                    - Monthly Contributions of both the hot and cold seasons within the season.
                - Scatter Plot of PCA Clusters (PC1 vs. PC2) grouped by U.S. Regions.
                - Feature importance analysis identifies the key weather factors influencing energy consumption.
            4. **Correlation Analysis**: Checking the relationship between weather variables (Temperature, CDD, HDD) and energy consumption sectors for each state.
            5. **Modeling**: Using regression models to predict energy consumption and determine feature importance.
        """)
    
    # For data scientists (hidden section)
    with st.expander("### **For the Data Scientist: Advanced Options**"):
        st.markdown("""
        - **Description**: This window is hidden under an expander. This window holds the data science process behind what the normal user sees on the widgets. 
        """)

    st.markdown("""
    ---
    
    ### **Contact Information**:
    
    If you have any questions, need assistance, or would like to discuss this project further, feel free to reach out to me. I am also looking for additional datasets that can work together 
    with weather and energy consumption data, and it would be great if you have or know datasets that I can add to my application and could mail it to me.
    
    - **Email**: [vennamva@msu.edu](mailto:vennamva@msu.edu)
    - **GitHub**: [https://github.com/Vennamm](https://github.com/Vennamm)
    - **LinkedIn**: [https://www.linkedin.com/in/vaibhavvennam](https://www.linkedin.com/in/vaibhavvennam)
    
    I am happy to connect and assist with any inquiries! You are free to leave any anonymous feedback below as well.
    """)

    token = st.secrets["github"]["GITHUB_TOKEN"]
    repo_owner = st.secrets["github"]["REPO_OWNER"]
    repo_name = st.secrets["github"]["REPO_NAME"]

    file_path = "feedback.txt"
    branch = "main"
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"

    headers = {"Authorization": f"token {token}"}
    
    st.markdown("""
    ---
    
    ### **Leave your feedback here:**
    """)

    
    feedback = st.text_area("", placeholder="Enter your feedback here", height=100, key="text_input")
    
    if st.button('Submit Feedback'):
        if feedback.strip():
            utc_now = datetime.utcnow()
            current_month = utc_now.month
            if current_month > 3 and current_month < 11:
                michigan_time = utc_now - timedelta(hours=4)
            else:
                michigan_time = utc_now - timedelta(hours=5)
            michigan_time_str = michigan_time.strftime("%Y-%m-%d %H:%M:%S")
            feedback = michigan_time_str + ' - ' + feedback
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                response_json = response.json()
                sha = response_json["sha"]
                current_content = base64.b64decode(response_json["content"]).decode("utf-8")
                updated_content = f"{current_content}\n{feedback}"
                content_encoded = base64.b64encode(updated_content.encode("utf-8")).decode("utf-8")
                data = {
                    "message": "User feedback submission",
                    "content": content_encoded,
                    "branch": branch,
                    "sha": sha
                }
                update_response = requests.put(url, headers=headers, json=data)
                if update_response.status_code == 200:
                    st.success("Thank you! Your feedback has been submitted.")
                    # st.session_state["text_input"] = ""
                    time.sleep(3)
                    
                else:
                    st.error("Failed to submit feedback. Please try again later.")
            else:
                st.error("Could not retrieve feedback file. Ensure it exists in the repository.")
        else:
            st.warning("Feedback cannot be empty. Please write something before submitting.")
    
with t2:
# elif section == "Temperature Forecasting":
    temperature_forecasting()
with t3:
# elif section == "Weather-Energy Relationship":
    weather_energy_cluster()
with t4:
    st.markdown('Future Goals')
