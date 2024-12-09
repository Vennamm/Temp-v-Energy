# st.set_page_config(page_title="Documentation", layout='wide')

st.title("Documentation")
st.markdown("""
# **Weather-Energy Relationship Analysis and Forecasting**

This project explores the relationships between weather patterns (Average Temperature, Cooling Degree Days (CDD), Heating Degree Days (HDD)) and energy consumption across different sectors (Residential, Commercial, Industrial, Transportation, and Total Consumption). It uses various data science techniques, such as Principal Component Analysis (PCA), Random Forests, and XGBoost, to create insights into regional energy consumption patterns based on weather factors. The analysis is visualized using interactive maps, graphs, and charts on a Streamlit-based app.

### **Sources**
- [**Weather Data**](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/): This is official weather data from NCEI. **Granularity:** Monthly, State. 
- [**Energy Data**](https://www.eia.gov/state/seds/seds-data-complete.php?sid=MI#Keystatisticsrankings): This is the energy consumption data sourced from U.S. Energy Information Administration. **Granularity:** Yearly, State.
- [GeoJSON for Mapping](https://github.com/PublicaMundi/MappingAPI/blob/master/data/geojson/us-states.json): MappingAPI's GitHub Repository.
- There are a few other sources that have not been used but have potential to explore: [Monthly Energy Consumption - Nationwide](https://www.eia.gov/totalenergy/data/browser/index.php?tbl=T01.01#/?f=M&start=199701&end=202407&charted=6-7-14-1-2-3-4-8-11-12-13), [Wildfire Data](https://www.nifc.gov/fire-information/statistics/wildfires)
## **Project Overview**

The project is divided into three main sections:

### **Section 1: Temperature Forecasting**
- **Objective**: Forecast average temperatures using time-series analysis to help predict energy consumption based on temperature trends.
- **Model**: SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors).
- **Process**: 
    - **Data Preparation**: Loading temperature data for different states.
    - **Granger Causality**: Checking dependencies between temperature and other weather factors.
    - **Seasonal Decomposition**: Breaking down the temperature data into trend, seasonal, and residual components.
    - **Forecasting**: Using SARIMAX to predict future temperatures.
    - **Visualization**: Graphs displaying predicted and actual temperatures, model performance metrics, and seasonal components.

- **Streamlit Features**:
    - Temperature forecasting for selected states, years, and months.
    - Performance metrics and visualizations.

### **Section 2: Weather-Energy Relationships**
- **Objective**: Investigate how temperature (CDD, HDD, and Average Temperature) influences energy consumption in various sectors across different states.
- **Techniques**:
    - **Principal Component Analysis (PCA)**: Used to reduce the dimensionality of the data and identify key components driving energy consumption.
    - **Clustering**: k-Nearest Neighbors (kNN) clustering algorithm used to assign states to different groups based on their behavior in energy consumption related to weather patterns.
    - **Feature Importance**: Random Forest and XGBoost regression models are trained to determine which weather-related features (e.g., CDD, HDD) are the most important predictors for energy consumption in each sector.

- **Process**:
    1. **Data Aggregation**: Collating weather data (CDD, HDD) and energy consumption data (Residential, Commercial, Industrial, etc.) for each state.
    2. **PCA and Clustering**: Reducing dimensionality and grouping states based on similar energy behaviors.
    3. **Visualization**: 
        - Cluster-based PCA visualizations are shown on an interactive map.
        - Two side-by-side bar graphs display energy consumption across seasons.
        - Feature importance analysis identifies the key weather factors influencing energy consumption.
    4. **Correlation Analysis**: Checking the relationship between weather variables (Temperature, CDD, HDD) and energy consumption sectors for each state.
    5. **Modeling**: Using regression models to predict energy consumption and determine feature importance.

- **Streamlit Features**:
    - Interactive selection of states and energy sectors.
    - Visualize cluster maps based on PCA.
    - Seasonal energy consumption comparison through bar graphs.
    - Relationship visualizations between weather variables and energy consumption.

### **Section 3: Advanced Features**
- **Objective**: Provide more granular insights into how specific weather patterns (seasonal or monthly) influence energy consumption.
- **Process**:
    1. **Energy Consumption Forecasting**: Using machine learning models to predict future energy consumption based on weather data.
    2. **Advanced Visualizations**: Creating detailed plots to explore how different weather factors impact energy consumption on a more granular level (e.g., by month, by state).
    3. **Customization**: Allowing the user to tweak the number of clusters or groups when examining seasonal or monthly energy consumption patterns.
""")
