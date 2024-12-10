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
   repo = st.secrets["github"]["REPO_NAME

---

