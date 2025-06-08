# Fundamental Analysis Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit)
![Heroku](https://img.shields.io/badge/Heroku-deployed-430098?style=for-the-badge&logo=heroku)

A comprehensive web application designed for the fundamental analysis of public companies. This platform streamlines financial data aggregation, ratio calculation, and valuation to derive a company's intrinsic value.

**[View Live Demo on Heroku](https://your-app-name.herokuapp.com)** ![App Screenshot](placeholder.png) ---

## 🚀 Overview

Built for finance students and professionals, this platform serves as an all-in-one dashboard for making informed financial decisions. It automates the tedious aspects of financial analysis, allowing users to focus on interpretation and strategy.

## ✨ Key Features

* **Dynamic Data Integration:** Fetches real-time and historical financial data from the Financial Modeling Prep (FMP) API.
* **Interactive Financial Statements:** Displays clean, readable versions of the Income Statement, Balance Sheet, and Cash Flow Statement.
* **Automated Ratio Analysis:** Calculates and visualizes over 20 key financial ratios, including liquidity, profitability, leverage, and efficiency metrics.
* **DCF Valuation Model:** Features a detailed, interactive Discounted Cash Flow (DCF) model to estimate a stock's intrinsic value based on user-defined assumptions.
* **AI-Powered Summaries:** Leverages the Google Gemini API to generate concise, AI-driven summaries of a company's financial health.
* **Scenario Modeling:** Includes a loan impact simulator to assess how new debt would affect a company's interest coverage ratio.

## 🛠️ Tech Stack

| Category            | Technology                                                                                                                                                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backend & Core** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)                              |
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)                       |
| **APIs & Services** | ![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white) ![Financial Modeling Prep](https://img.shields.io/badge/FMP%20API-0077B5?style=for-the-badge)                      |
| **Deployment** | ![Heroku](https://img.shields.io/badge/Heroku-430098?style=for-the-badge&logo=heroku&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)                                         |

## ⚙️ Local Setup & Installation

To run this application locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/fundamental-analysis-platform.git](https://github.com/your-username/fundamental-analysis-platform.git)
    cd fundamental-analysis-platform
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    This project requires API keys. Create a `.env` file in the root directory and add your keys in the following format:
    ```
    FMP_API_KEY="YOUR_FMP_API_KEY"
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ALPHA_VANTAGE_API_KEY="YOUR_ALPHA_VANTAGE_KEY"
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run financial_toolkit_app_fmp.py
    ```
