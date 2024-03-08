import os
import json
import logging
import pandas as pd
import requests
import uuid
from typing import Optional, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import time

# Configuration file for default values. Sensitive data like API tokens should be stored in environment variables.
CONFIG_FILE = 'config.json'
API_TOKEN = os.getenv("API_TOKEN")  # API token stored as an environment variable

def configure_logging():
    """
    Configures logging for the application.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='./error/app.log',
                        filemode='w')

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file.
    """
    with open(config_file, 'r') as f:
        return json.load(f)

def get_headers(api_token: str) -> Dict[str, str]:
    """
    Returns the headers needed for the API request, including the Authorization token.
    """
    return {'Authorization': f"Token {api_token}", 'Content-Type': 'application/json'}

def send_request_with_retry(endpoint: str, headers: Dict[str, str], payload: Dict, max_retries: int = 3, backoff_factor: float = 0.3) -> Optional[requests.Response]:
    """
    Sends a POST request with a retry mechanism on failure.
    """
    session = requests.Session()
    retries = Retry(total=max_retries, backoff_factor=backoff_factor, status_forcelist=[500, 502, 503, 504], allowed_methods=frozenset(['POST']))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    try:
        response = session.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logging.error(f'Request failed: {e}')
        return None

def safe_str(value) -> str:
    """
    Converts a value to a string, handling NaN values.
    """
    if pd.isna(value):
        return ''
    return str(value)

def create_payload(row: pd.Series, project_uuid: str) -> Dict[str, Any]:
    """
    Creates a payload for the API request from a row of data.
    """
    return {
        "id": project_uuid,
        "submission": {
            "formhub": {"uuid": "ba58ebec6e0948788e3b6069a1e2f19f"},
            "Reception_ID": safe_str(row['Reception_ID']),
            "Type": safe_str(row['Type']),
            "Group_Size": safe_str(row['Group_Size']),
            "__version__": "v6aBj5Nyn7GUydpo5kXjsv",
            "meta": {"instanceID": f"uuid:{safe_str(uuid.uuid4())}"}
        }
    }

def process_batch(batch_df: pd.DataFrame, endpoint: str, headers: Dict[str, str], project_uuid: str, max_workers: int):
    """
    Processes a batch of data by sending parallel requests to the API.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_request_with_retry, endpoint, headers, create_payload(row, project_uuid)) for _, row in batch_df.iterrows()]

        for future in as_completed(futures):
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
    endpoint = 'https://kobocat.unhcr.org/api/v1/submissions'
    headers = get_headers(API_TOKEN)
    project_uuid = config['project_uuid']
    max_workers = config.get('concurrency_level', 5)

    df_root = pd.read_excel(config['parent_data_path'])

    for _, batch_df in tqdm(df_root.groupby(np.arange(len(df_root)) // config['batch_size']), desc="Processing batches"):
        process_batch(batch_df, endpoint, headers, project_uuid, max_workers)
        time.sleep(config.get('dynamic_sleep_interval', 5))  # Adjust dynamically based on feedback

if __name__ == '__main__':
    main()
