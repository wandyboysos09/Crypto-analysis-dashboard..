
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
# Explicitly define the directory where CSV files are located
DATA_DIR = "/content/"

@st.cache_data # Cache data loading to improve performance
def load_data():
    try:
        file_names = {
            'summary_df': 'backtesting_summary.csv',
            'cleaned_df': 'cleaned_data.csv',
            'weekly_df_with_ma': 'weekly_ma_data.csv',
            'monthly_df_with_ma': 'monthly_ma_data.csv',
            'quarterly_df_with_ma': 'quarterly_ma_data.csv',
            'model_eval_df': 'model_evaluation_results.csv' # Added model_eval_df
        }

        loaded_data = {}
        for df_name, file_name in file_names.items():
            full_path = os.path.join(DATA_DIR, file_name)
            # Adding debug prints to Colab's stdout
            print(f"Debug: Attempting to load {file_name} from {full_path}. Exists: {os.path.exists(full_path)}") 
            loaded_data[df_name] = pd.read_csv(full_path)

        return (
            loaded_data['summary_df'],
            loaded_data['cleaned_df'],
            loaded_data['weekly_df_with_ma'],
            loaded_data['monthly_df_with_ma'],
            loaded_data['quarterly_df_with_ma'],
            loaded_data['model_eval_df'] # Added model_eval_df
        )
    except FileNotFoundError as e:
        print(f"Error: FileNotFoundError caught in load_data: {e}") # Print the actual exception
        st.error(f"One or more data files not found. Please ensure all CSV files are in the '{DATA_DIR}' directory.")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Error: An unexpected error occurred during data loading: {e}")
        st.error(f"An unexpected error occurred during data loading: {e}")
        return None, None, None, None, None, None

summary_df, cleaned_df, weekly_df_with_ma, monthly_df_with_ma, quarterly_df_with_ma, model_eval_df = load_data()

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

        if model_eval_df is not None and not model_eval_df.empty:
            st.subheader("Overall Model Evaluation Results")
            st.dataframe(model_eval_df)

            st.subheader("Filter by Cryptocurrency and Interval")
            
            # Dynamically get unique symbols and intervals from model_eval_df
            unique_symbols = sorted(model_eval_df['Symbol'].unique().tolist())
            unique_intervals = sorted(model_eval_df['Interval'].unique().tolist())

            selected_symbol_model = st.selectbox("Select Cryptocurrency", ['All'] + unique_symbols, key='select_symbol_model')
            selected_interval_model = st.selectbox("Select Interval", ['All'] + unique_intervals, key='select_interval_model')

            filtered_model_eval_df = model_eval_df.copy()
            if selected_symbol_model != 'All':
                filtered_model_eval_df = filtered_model_eval_df[filtered_model_eval_df['Symbol'] == selected_symbol_model]
            if selected_interval_model != 'All':
                filtered_model_eval_df = filtered_model_eval_df[filtered_model_eval_df['Interval'] == selected_interval_model]

            st.dataframe(filtered_model_eval_df)

            # Visualization of Comparative Model Performance (e.g., RMSE)
            if not filtered_model_eval_df.empty:
                metric_choice = st.radio("Select Metric to Visualize", ['RMSE', 'MAE'], key='metric_choice')
                
                # Prepare data for plotting
                plot_df = filtered_model_eval_df.melt(id_vars=['Symbol', 'Interval'], 
                                                       value_vars=[col for col in filtered_model_eval_df.columns if metric_choice in col], 
                                                       var_name='Model_Metric', 
                                                       value_name=metric_choice)
                
                # Clean up Model_Metric names for better readability
                plot_df['Model'] = plot_df['Model_Metric'].apply(lambda x: x.replace(f'_{metric_choice}', ''))
                
                # Remove rows where the metric value is NaN (for models that might not have run)
                plot_df.dropna(subset=[metric_choice], inplace=True)

                if not plot_df.empty:
                    fig_model_perf = plt.figure(figsize=(12, 6))
                    sns.barplot(data=plot_df, x='Model', y=metric_choice, hue='Symbol', palette='viridis')
                    plt.title(f'Comparative Model {metric_choice} for Selected Cryptocurrencies and Intervals')
                    plt.xlabel('Model')
                    plt.ylabel(f'{metric_choice}')
                    plt.xticks(rotation=45)
                    plt.legend(title='Cryptocurrency')
                    st.pyplot(fig_model_perf)
                else:
                    st.warning(f"No data to plot for the selected filters and metric ({metric_choice}).")
            else:
                st.info("Filter the table above to see comparative model performance visualization.")

        else:
            st.warning("Model evaluation results not loaded or empty. Cannot display model performance.")
