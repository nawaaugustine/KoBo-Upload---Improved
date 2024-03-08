import json
import uuid
import pandas as pd
import requests
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import numpy as np 

# Configuration file for default values and API tokens.
CONFIG_FILE = 'config.json'

def configure_logging():
    """
    Configures logging for the application.
    Logs are written to './error/app.log' with a specific format including the timestamp.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='./error/app.log',
                        filemode='w')

def load_config(config_file):
    """
    Loads configuration from a JSON file.
    
    :param config_file: Path to the configuration file.
    :return: Dictionary with configuration values.
    """
    with open(config_file, 'r') as f:
        return json.load(f)

def get_api_token(config, user_value):
    """
    Retrieves the API token based on a user value from the configuration.
    
    :param config: Configuration dictionary.
    :param user_value: A key to lookup the API token in the configuration.
    :return: API token as a string.
    """
    token_key = config.get("api_token_map", {}).get(user_value)
    if not token_key:
        logging.error(f'Unknown user value: {user_value}')
        raise ValueError(f'Unknown user value: {user_value}')
    return token_key

def send_request_with_retry(endpoint, headers, payload, max_retries=3, backoff_factor=0.3):
    """
    Sends a POST request with retry mechanism on failure.
    
    :param endpoint: URL endpoint for the POST request.
    :param headers: Request headers.
    :param payload: JSON payload for the POST request.
    :param max_retries: Maximum number of retries on failure.
    :param backoff_factor: Backoff factor for retries.
    :return: Response object or None if request fails.
    """
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=frozenset(['POST'])  
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    try:
        response = session.post(endpoint, headers=headers, json=payload)
        return response
    except requests.RequestException as e:
        logging.error(f'Request failed: {e}')
        return None

def safe_str(value):
    """
    Converts a value to a string, handling NaN values.
    
    :param value: The value to convert.
    :return: String representation of the value or empty string if NaN.
    """
    if pd.isna(value):
        return ''
    return str(value)

def create_payload(parent_row, project_uuid):
    """
    Creates a payload for the API request from a row of data.
    
    :param parent_row: A pandas Series representing a row of the dataframe.
    :param project_uuid: UUID for the project, included in the payload.
    :return: A dictionary representing the JSON payload for the request.
    """
    payload = {
        "id": project_uuid,
        "submission": {
            "formhub": {"uuid": "6c18862e8a0442f5b04e957541bb223d"}, #To update
            "Process_Status": safe_str(parent_row['Process_Status']),
            "Reception_ID": safe_str(parent_row['Reception_ID']),
            "Full_Name": safe_str(parent_row['Full_Name']),
            "Sex": safe_str(parent_row['Sex']),
            "Date_of_Birth": safe_str(parent_row['Date_of_Birth']),
            "Arrival_Date": safe_str(parent_row['Arrival_Date']),
            "Ethnicity": safe_str(parent_row['Ethnicity']),
            "Group_Size": safe_str(parent_row['Group_Size']),
            "Reception_Location": safe_str(parent_row['Reception_Location']),
            
            "__version__": "vHgTnHiEdARTknHYRTLR2x",#To update
            
            "meta": {"instanceID": f"uuid:{safe_str(uuid.uuid4())}"}
        }
    }
    return payload

def process_batch(batch_df, config, endpoint, headers):
    """
    Processes a batch of data by sending parallel requests to the API.
    
    :param batch_df: DataFrame slice representing the batch to process.
    :param config: Configuration dictionary.
    :param endpoint: API endpoint for the data submission.
    :param headers: Headers to include in the request.
    """
    with ThreadPoolExecutor(max_workers=config['concurrency_level']) as executor:
        future_to_row = {
            executor.submit(send_request_with_retry, endpoint, headers, create_payload(row, config['project_uuid'])): row
            for _, row in batch_df.iterrows()
        }
        for future in as_completed(future_to_row):
            response = future.result()
            if response and response.status_code == 201:
                logging.info('Submission success')
            else:
                logging.error('Submission failed')

def main():
    """
    Main function to load configuration, prepare data, and process submissions in batches.
    """
    configure_logging()
    config = load_config(CONFIG_FILE)
    
    df_root = pd.read_excel(config['parent_data_path'])
    endpoint = 'https://kobocat.unhcr.org/api/v1/submissions'
    api_token = get_api_token(config, 'unhcr_prod')
    headers = {'Authorization': f"Token {api_token}", 'Content-Type': 'application/json'}
    
    for _, batch_df in tqdm(df_root.groupby(np.arange(len(df_root)) // config['batch_size']), desc="Processing batches"):
        process_batch(batch_df, config, endpoint, headers)
        time.sleep(config.get('dynamic_sleep_interval', 5))  # Adjust dynamically based on feedback

if __name__ == '__main__':
    main()
