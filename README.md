# Weather and Energy Consumption Forecasting

This project aims to provide insights into how weather patterns, particularly temperature and seasonal variations, influence energy consumption across different states in the U.S. 

Using historical weather data (temperature, Cooling Degree Days (CDD), and Heating Degree Days (HDD)), along with energy consumption data for different sectors (Residential, Commercial, Industrial, and Transport), this app helps users understand the correlations between weather patterns and energy consumption.

## Features:
- **Temperature Forecasting**: Predicts the average temperature for the next 60 months using historical weather data.
- **Weather-Energy Clustering**: Analyzes correlations between weather patterns and energy consumption across U.S. states.
- **Seasonal Energy Relationships**: Helps users understand how seasonal temperature variations impact energy consumption for different sectors.
- **Feedback System**: Allows users to submit feedback directly within the app to the `feedback.txt` file on the repo.

---
## Installation Instructions

To run this project locally, follow these steps:

### Prerequisites:
Make sure you have the following installed:
- **Python** (3.8 or later)
- **Streamlit** (For the app interface)
- **Git** (For cloning the repository)

### Steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Vennamm/Temp-v-Energy.git
   cd your-repository

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the app**:
   ```bash
   streamlit run Project.py

**GitHub Token for Feedback System:**

   To enable the feedback system in the app, you’ll need to provide a **GitHub Token**. This token is required for reading and writing to your `feedback.txt` file in a GitHub repository. Here's how you can set it up securely in **Streamlit Cloud**:
   
   **Setting Up on Streamlit Cloud:**
   
   1. **Navigate to Your App Settings**:
      - Go to **My Apps** and find your deployed app.
      - Click on the gear icon in the top-right corner to access the settings.
   
   2. **Access Secrets**:
      - In the settings menu, find the **Secrets** section.
      - Add the following code in the **Secrets** section:
        ```toml
        [github]
        GITHUB_TOKEN = "your-github-token"
        REPO_OWNER = "your_username"
        REPO_NAME = "your_reponame"
        ```
   
   3. **Why Use Secrets?**  
      Streamlit ensures that your GitHub token is securely stored and is not exposed, preventing misuse. You can now use these secrets in your app.
   
   **Accessing the GitHub Token in Your Code:**
   
   In your app code, you can retrieve the token, owner, and repository name like this:
   ```python
   token = st.secrets["github"]["GITHUB_TOKEN"]
   owner = st.secrets["github"]["REPO_OWNER"]
   repo = st.secrets["github"]["REPO_NAME"]
   ```
---

# Usage Instructions

This section provides a step-by-step guide on how users can interact with the app. I will break it down by key features and their usage:

## 1. Accessing the App
- To use the app, open it in a web browser. You will be presented with the Documentation Page.
- The app is hosted on [this link](https://weathergy.streamlit.app/), so you can simply visit it to start using the app.

## 2. Temperature Forecasting (SARIMAX Model)
- On this page, you can select the **state**, **year**, and **month** for which you want to forecast the average temperature.
- **Step 1**: Select the state from the dropdown list.
- **Step 2**: Choose the year and month for which you want the temperature forecast.
- The forecast will be shown, including the actual values if available (until 2022).

## 3. Weather-Energy Relationship
- This section visualizes weather-energy consumption clusters based on Cooling Degree Days (CDD), Heating Degree Days (HDD), and energy consumption data.
- **Step 1**: Choose a state and sector to see the specific energy consumption by sector and the relationship to seasonal weather patterns.
- **Step 2**: View the clustering map, which shows how different states are grouped based on their weather-energy consumption behaviors.

## 4. Feedback System
- Users can leave feedback regarding the app through the **Feedback** section. You can find it on the Documentation tab.
- **Step 1**: Type your feedback into the text box.
- **Step 2**: Press the **Submit Feedback** button to send your comments.
- **Note**: Your feedback is stored securely on GitHub, and the app will show a success message once the feedback has been successfully submitted.

---
## Project File Map

Below is the list of files and folders within this project, organized by relevance:

### Main Project Files

- **[Project.py](./Project.py)** - The main Streamlit application script.
- **[requirements.txt](./requirements.txt)** - Lists all dependencies required for the app to run smoothly.
- **[feedback.txt](./feedback.txt)** - Stores feedback received from users within the app.
- **[Rubric Match.md](./Rubric%20Match.md)** - Contains validations against the requirements of the project.
- **[us-states.json](./us-states.json)** - The GeoMappingAPI's GeoJSON file used for mapping in the Streamlit app.

### Data Folders

- **`collated_data/`** - Contains weather data for each state for each month.
  - Includes weather-related datasets for all U.S. states.

- **`power_consumption/`** - Contains energy consumption data for each state over a year.
  - Energy consumption data for Residential, Commercial, Industrial, and Transport sectors.

- **`feature_importances/`** - Contains feature importances derived from three models: Decision Tree Regressor, XGB Regressor, and Random Forest Regressor.
  - Includes monthly energy contributions for each state and sector.
 
- **`power_relationships/`** - Contains manually defined valid power relationships for each state.
  - Includes energy consumption patterns and correlations with weather data.

### Additional Sources

- **`Notebooks/`** - Contains all the analysis and work done before they were imported onto Streamlit.
  - Includes Jupyter Notebooks with exploratory data analysis (EDA) and model building.

- **`Additional Sources/`** - Contains raw data from additional sources, which have the potential to be used in future versions of the app.

- **`additional_power_data/`** - Contains cleaned additional sources that are currently unused in the app.

---




