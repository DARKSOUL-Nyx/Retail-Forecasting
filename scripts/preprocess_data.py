import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)

# These are the paths INSIDE the container
RAW_DATA_PATH = "/opt/airflow/data/raw/train.csv"
PROCESSED_DATA_DIR = "/opt/airflow/data/processed"
INITIAL_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "initial_train.csv")

def preprocess_and_split_data():
    """
    Reads the raw data, creates an initial training set, and splits the rest
    into monthly files to simulate new data arrival.
    """
    logging.info(f"Starting preprocessing of {RAW_DATA_PATH}...")

    # Create the output directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    try:
        df = pd.read_csv(RAW_DATA_PATH, parse_dates=['date'])
        df = df.sort_values('date')
        
        # Define the split date for the initial training set (e.g., end of 2015)
        split_date = pd.to_datetime('2015-12-31')
        
        # Create the initial training set
        initial_train_df = df[df['date'] <= split_date]
        initial_train_df.to_csv(INITIAL_TRAIN_PATH, index=False)
        logging.info(f"Saved initial training data ({len(initial_train_df)} rows) to {INITIAL_TRAIN_PATH}")

        # Get the remaining data for incremental batches
        incremental_df = df[df['date'] > split_date]
        
        # Group by year and month, then save each group to a separate CSV
        incremental_df['year_month'] = incremental_df['date'].dt.to_period('M')
        
        for period, group in incremental_df.groupby('year_month'):
            file_name = f"sales_{period}.csv"
            output_path = os.path.join(PROCESSED_DATA_DIR, file_name)
            group.drop(columns=['year_month']).to_csv(output_path, index=False)
            logging.info(f"Saved incremental data for {period} to {output_path}")

        logging.info("Preprocessing complete!")

    except FileNotFoundError:
        logging.error(f"Error: The raw data file was not found at {RAW_DATA_PATH}. "
                      "Make sure 'train.csv' is in the 'data/raw' directory.")
    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")


if __name__ == "__main__":
    preprocess_and_split_data()