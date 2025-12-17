
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Crypto Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Title and Introduction ---
st.title("Cryptocurrency Trading Strategy Analysis")
st.write("Explore the results of our cryptocurrency analysis, including clustering, model performance, and backtesting.")

# --- Load Data ---
# Assuming your CSV files are saved in the current directory (or a known path)
@st.cache_data # Cache data loading to improve performance
def load_data():
    try:
        summary_df = pd.read_csv('backtesting_summary.csv')
        cleaned_df = pd.read_csv('cleaned_data.csv')
        weekly_df_with_ma = pd.read_csv('weekly_ma_data.csv')
        monthly_df_with_ma = pd.read_csv('monthly_ma_data.csv')
        quarterly_df_with_ma = pd.read_csv('quarterly_ma_data.csv')
        return summary_df, cleaned_df, weekly_df_with_ma, monthly_df_with_ma, quarterly_df_with_ma
    except FileNotFoundError:
        st.error("One or more data files not found. Please ensure 'backtesting_summary.csv', 'cleaned_data.csv', 'weekly_ma_data.csv', 'monthly_ma_data.csv', and 'quarterly_ma_data.csv' are in the same directory as the app.py.")
        return None, None, None, None, None

summary_df, cleaned_df, weekly_df_with_ma, monthly_df_with_ma, quarterly_df_with_ma = load_data()

if summary_df is not None:
    # --- Sidebar for Navigation/Filters ---
    st.sidebar.header("Dashboard Navigation")
    section = st.sidebar.radio(
        "Go to",
        ("Overview", "Backtesting Summary", "EDA Visualizations", "Correlation Analysis", "Model Performance")
    )

    # --- Overview Section ---
    if section == "Overview":
        st.header("Project Overview")
        st.markdown("""
        This dashboard presents a multi-stage analysis of cryptocurrency market data.
        We started with data acquisition and cleaning, followed by clustering to identify representative assets.
        Various machine learning models were trained for price prediction, and a simple trading strategy was backtested.
        """)
        st.subheader("Representative Cryptocurrencies (from Clustering)")
        # Assuming 'representative_cryptos' is available or you hardcode them
        representative_symbols = ['DASH-USD', 'BTC-USD', 'ETH-USD', 'BCH-USD'] # Example, adapt as needed
        st.write(representative_symbols)

    # --- Backtesting Summary Section ---
    elif section == "Backtesting Summary":
        st.header("Trading Strategy Backtesting Results")

        st.subheader("Overall Backtesting Performance")
        st.dataframe(summary_df)

        st.subheader("Filter by Cryptocurrency and Interval")
        selected_symbol_bt = st.selectbox("Select Cryptocurrency", ['All'] + sorted(summary_df['Symbol'].unique().tolist()), key='select_symbol_bt')
        selected_interval_bt = st.selectbox("Select Interval", ['All'] + summary_df['Interval'].unique().tolist(), key='select_interval_bt')

        filtered_summary_df = summary_df.copy()
        if selected_symbol_bt != 'All':
            filtered_summary_df = filtered_summary_df[filtered_summary_df['Symbol'] == selected_symbol_bt]
        if selected_interval_bt != 'All':
            filtered_summary_df = filtered_summary_df[filtered_summary_df['Interval'] == selected_interval_bt]

        st.dataframe(filtered_summary_df)

        # Basic visualization of total profit
        if not filtered_summary_df.empty:
            fig_profit = plt.figure(figsize=(10, 5))
            sns.barplot(x='Symbol', y='Total Profit ($)', hue='Interval', data=filtered_summary_df)
            plt.title('Total Profit by Cryptocurrency and Interval')
            plt.xticks(rotation=45)
            st.pyplot(fig_profit)

    # --- EDA Visualizations Section ---
    elif section == "EDA Visualizations":
        st.header("Exploratory Data Analysis")

        if cleaned_df is not None:
            # Assuming `representative_symbols_list` is known or derived
            representative_symbols = ['DASH-USD', 'BTC-USD', 'ETH-USD', 'BCH-USD'] # Example, adapt as needed
            selected_crypto_eda = st.selectbox("Select Cryptocurrency for EDA", sorted(representative_symbols))

            daily_crypto_df_eda = cleaned_df[cleaned_df['symbol'] == selected_crypto_eda].sort_values('Date').copy()

            if not daily_crypto_df_eda.empty:
                st.subheader(f"{selected_crypto_eda} Daily Close Price Over Time")
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                sns.lineplot(data=daily_crypto_df_eda, x='Date', y='Close', ax=ax1)
                ax1.set_title(f'{selected_crypto_eda} Daily Close Price Over Time')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Close Price (USD)')
                ax1.grid(True)
                st.pyplot(fig1)

                st.subheader(f"Distribution of {selected_crypto_eda} Daily Close Prices")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.histplot(daily_crypto_df_eda['Close'], kde=True, bins=50, ax=ax2)
                ax2.set_title(f'Distribution of {selected_crypto_eda} Daily Close Prices')
                ax2.set_xlabel('Close Price (USD)')
                ax2.set_ylabel('Frequency')
                st.pyplot(fig2)
            else:
                st.warning(f"No daily data available for {selected_crypto_eda} for EDA.")
        else:
            st.warning("Daily data not loaded. Cannot perform EDA.")

    # --- Correlation Analysis Section ---
    elif section == "Correlation Analysis":
        st.header("Correlation Analysis")
        st.write("Examine the correlation between representative cryptocurrencies across different time intervals.")

        selected_interval_corr = st.selectbox("Select Interval for Correlation", ['Weekly', 'Monthly', 'Quarterly'])

        # Re-using the calculate_and_present_correlations logic in a Streamlit-friendly way
        df_for_corr = None
        if selected_interval_corr == 'Weekly':
            df_for_corr = weekly_df_with_ma
        elif selected_interval_corr == 'Monthly':
            df_for_corr = monthly_df_with_ma
        elif selected_interval_corr == 'Quarterly':
            df_for_corr = quarterly_df_with_ma

        if df_for_corr is not None and not df_for_corr.empty:
            representative_symbols = ['DASH-USD', 'BTC-USD', 'ETH-USD', 'BCH-USD'] # Example, adapt as needed
            df_filtered_corr = df_for_corr[df_for_corr['symbol'].isin(representative_symbols)].copy()

            if not df_filtered_corr.empty:
                correlation_pivot = df_filtered_corr.pivot(index='Date', columns='symbol', values='Close')
                correlation_pivot.dropna(axis=1, how='all', inplace=True)
                correlation_pivot.fillna(method='ffill', inplace=True)
                correlation_pivot.fillna(method='bfill', inplace=True)
                correlation_pivot.fillna(correlation_pivot.mean(), inplace=True)

                if correlation_pivot.shape[1] >= 2:
                    correlation_matrix = correlation_pivot.corr()

                    st.subheader(f"Correlation Matrix ({selected_interval_corr})")
                    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
                    st.pyplot(fig_corr)

                    corr_pairs = correlation_matrix.stack().reset_index()
                    corr_pairs.columns = ['Crypto1', 'Crypto2', 'Correlation']
                    corr_pairs = corr_pairs[corr_pairs['Crypto1'] != corr_pairs['Crypto2']]
                    corr_pairs['Pair'] = corr_pairs.apply(lambda x: tuple(sorted((x['Crypto1'], x['Crypto2']))), axis=1)
                    corr_pairs.drop_duplicates(subset='Pair', inplace=True)

                    st.subheader(f"Top Most Positively Correlated Pairs ({selected_interval_corr})")
                    st.dataframe(corr_pairs.sort_values(by='Correlation', ascending=False).head(4).drop(columns='Pair'))

                    st.subheader(f"Top Most Negatively Correlated Pairs ({selected_interval_corr})")
                    st.dataframe(corr_pairs.sort_values(by='Correlation', ascending=True).head(4).drop(columns='Pair'))
                else:
                    st.warning(f"Not enough data points or cryptocurrencies (at least 2) for correlation in {selected_interval_corr} interval.")
            else:
                st.warning(f"No data for representative cryptocurrencies in {selected_interval_corr} interval for correlation analysis.")
        else:
            st.warning(f"Data for {selected_interval_corr} interval not loaded. Cannot perform correlation analysis.")

    # --- Model Performance Section ---
    elif section == "Model Performance":
        st.header("Machine Learning Model Performance")
        st.write("Compare the RMSE and MAE of different models across cryptocurrencies and intervals.")

        # Assuming you have `arima_results`, `lstm_results`, etc. loaded or accessible
        # For simplicity, we'll just display a consolidated view if available.
        # In a real app, you'd iterate through results to create a display.

        st.subheader("Consolidated Model Evaluation (Sample)")
        # Create a sample DataFrame for demonstration. In a full app, you'd build this from your actual results dictionaries.
        sample_model_eval_data = {
            'Symbol': ['BTC-USD', 'BTC-USD', 'ETH-USD', 'ETH-USD'],
            'Interval': ['Weekly', 'Monthly', 'Weekly', 'Monthly'],
            'ARIMA_RMSE': [35083.77, 36599.59, 963.64, 988.21],
            'LSTM_RMSE': [10726.57, 14519.53, 399.39, 892.24],
            'LightGBM_RMSE': [35319.91, 39906.65, 248.21, 633.68],
            'XGBoost_RMSE': [36203.02, 33857.21, 195.57, 274.75],
            'RandomForest_RMSE': [33772.30, 31920.08, 172.93, 237.93]
        }
        sample_model_eval_df = pd.DataFrame(sample_model_eval_data)

        st.dataframe(sample_model_eval_df)

        st.info("Note: In a full Streamlit app, you would dynamically load and display 'arima_results', 'lstm_results', etc., for all cryptocurrencies and intervals, allowing interactive filtering and comparison.")

