import requests
import regex as re
import os
import time
import tempfile

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, WebDriverException

import pandas as pd

from google.cloud import storage
from dotenv import load_dotenv

def setup_chrome_options():
    """Configures and returns Chrome options for headless Browse."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    return chrome_options

def initialize_driver(chrome_options):
    """Initializes and returns a Chrome WebDriver instance."""
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def navigate_to_page(driver, url):
    """Navigates the WebDriver to the specified URL."""
    print(f"\n[INFO] NAVIGATING TO: {url}")
    driver.get(url)

def find_and_get_pdf_url(driver, pdf_url_fragment):
    """
    Finds the PDF link on the page and returns its full, absolute URL.
    Waits up to 10 seconds for the link to be clickable.
    """
    pdf_filename_from_url = pdf_url_fragment.split('/')[-1]
    correct_relative_href = f"pdf/{pdf_filename_from_url}"
    pdf_link_xpath = f"//a[@href='{correct_relative_href}' and @data-ga-label='pdf_download_desktop']"

    wait = WebDriverWait(driver, 10)
    pdf_link_element = wait.until(
        EC.element_to_be_clickable((By.XPATH, pdf_link_xpath))
    )
    
    relative_url = pdf_link_element.get_attribute('href')
    # Resolve the relative URL to an absolute one
    base_url = driver.current_url
    absolute_url = urljoin(base_url, relative_url)
    print(f"[INFO] Found PDF download link: {absolute_url}")
    return absolute_url

def download_file_content(download_url):
    """Downloads file content from a URL using requests."""
    print(f"[INFO] Starting download from: {download_url}")
    response = requests.get(download_url, stream=True)
    response.raise_for_status()  # Will raise an HTTPError for bad responses
    return response

def save_content_to_file(response, save_path):
    """Saves the response content to a local file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[INFO] Successfully downloaded PDF to: {save_path}")
    return save_path

def download_pdf(pdf_url, article_page_url, output_directory, desired_filename):
    """
    Main pipeline function to download a PDF from PMC.
    It sets up a browser, navigates to the page, finds the link, downloads, and saves the file.
    
    Args:
        pdf_url (str): A URL fragment used to identify the correct download link.
        article_page_url (str): The URL of the article page to navigate to.
        output_directory (str): The directory to save the downloaded PDF in.
        desired_filename (str): The name for the saved PDF file.

    Returns:
        str: The full path to the saved file if successful, None otherwise.
    """
    driver = None
    save_path = os.path.join(output_directory, desired_filename)

    try:
        # Action 1: Configure browser
        options = setup_chrome_options()
        
        # Action 2: Initialize WebDriver
        driver = initialize_driver(options)
        
        # Action 3: Navigate to the article page
        navigate_to_page(driver, article_page_url)
        
        # Action 4: Find the actual PDF download URL from the page
        actual_download_url = find_and_get_pdf_url(driver, pdf_url)
        
        # Action 5: Download the file content using requests
        response = download_file_content(actual_download_url)
        
        # Action 6: Save the downloaded content to the specified file
        final_path = save_content_to_file(response, save_path)
        
        return final_path

    except (requests.exceptions.HTTPError, TimeoutException) as e:
        print(f"[ERROR] Failed to download during web interaction or request: {e}")
        if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
            print(f"  Status Code: {e.response.status_code}")
            print(f"  Response content (first 300 bytes): {e.response.content[:300]}")
        return None
    except WebDriverException as e:
        print(f"[ERROR] A WebDriverException occurred: {e}")
        if driver:
            print(f"  Current URL at WebDriverException: {driver.current_url}")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in the download pipeline: {e}")
        return None
    finally:
        # Final Action: Clean up and close the browser
        if driver:
            driver.quit()

def sanitize_filename(filename):
    """
    Removes characters that are invalid for most file systems and GCS object names.
    If the input is None or not a string, it returns a default name.
    """
    if not isinstance(filename, str) or not filename.strip():
        return "untitled_article"
    # Remove invalid characters
    s = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Replace whitespace with underscores and limit length
    s = re.sub(r'\s+', '_', s)
    return s[:150] # Limit filename length for safety

def load_configuration():
    """
    Loads environment variables and sets up GCS configuration.
    
    Returns:
        tuple: A tuple containing (GCS_BUCKET_NAME, GCS_FOLDER_PREFIX).
    """
    load_dotenv()
    GCP_CREDENTIAL_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if GCP_CREDENTIAL_JSON:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIAL_JSON
        print(f"[INFO] Using GCS credentials from: {GCP_CREDENTIAL_JSON}")
    else:
        print("[WARN] GOOGLE_APPLICATION_CREDENTIALS not found in .env. SDK might use default ADC.")

    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    GCS_FOLDER_PREFIX = os.getenv("GCS_FOLDER_PREFIX")
    
    return GCS_BUCKET_NAME, GCS_FOLDER_PREFIX

def initialize_gcs_bucket(bucket_name):
    """
    Initializes the connection to the Google Cloud Storage client and bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.

    Returns:
        google.cloud.storage.Bucket: The bucket object if successful, otherwise None.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        print(f"[INFO] Successfully connected to GCS bucket: {bucket_name}")
        return bucket
    except Exception as e:
        print(f"[ERROR] Failed to initialize GCS client or bucket: {e}. File uploads will be skipped.")
        return None

def load_source_data(filepath):
    """
    Loads the source data from a CSV file into a pandas DataFrame.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if loading fails.
    """
    try:
        database = pd.read_csv(filepath)
        print(f"[INFO] Successfully loaded data from '{filepath}'.")
        return database
    except FileNotFoundError:
        print(f"[ERROR] The CSV file '{filepath}' was not found. Exiting.")
        return None
    except Exception as e_csv:
        print(f"[ERROR] Failed to read or parse '{filepath}': {e_csv}. Exiting.")
        return None

def upload_to_gcs(bucket, source_file_path, gcs_blob_name):
    """
    Uploads a single file to the specified Google Cloud Storage bucket.

    Args:
        bucket (google.cloud.storage.Bucket): The GCS bucket object.
        source_file_path (str): The local path of the file to upload.
        gcs_blob_name (str): The destination path (blob name) in GCS.
    """
    if not bucket:
        print(f"[WARN] Skipping GCS upload for {os.path.basename(source_file_path)} as GCS bucket is not available.")
        return

    blob = bucket.blob(gcs_blob_name)
    try:
        print(f"[INFO] Uploading {os.path.basename(source_file_path)} to GCS bucket '{bucket.name}' as '{gcs_blob_name}'...")
        blob.upload_from_filename(source_file_path)
        print(f"[SUCCESS] Successfully uploaded to {gcs_blob_name}")
    except Exception as e_upload:
        print(f"[ERROR] Failed to upload {os.path.basename(source_file_path)} to GCS: {e_upload}")

def process_articles(database, bucket, gcs_folder_prefix):
    """
    Iterates through articles in the DataFrame, downloads them, and uploads them to GCS.
    
    Args:
        database (pandas.DataFrame): DataFrame containing article information.
        bucket (google.cloud.storage.Bucket): The initialized GCS bucket object.
        gcs_folder_prefix (str): The folder prefix to use for GCS uploads.
    """
    with tempfile.TemporaryDirectory() as temp_dir_name:
        print(f"[INFO] Created temporary directory: {temp_dir_name}")

        for index, row in database.iterrows():
            try:
                article_title = row.get('title')
                pmc_url = row.get('pmc_link')
                download_url = row.get('download_link')
                
                display_title = str(article_title) if pd.notna(article_title) else f"untitled_row_{index}"
                print(f"\n--- Processing article: {display_title[:70]}... (Row {index}) ---")

                if not (pmc_url and pd.notna(pmc_url) and download_url and pd.notna(download_url)):
                    print(f"[WARN] Missing 'pmc_link' or 'download_link'. Skipping.")
                    continue

                pdf_filename = sanitize_filename(article_title) + ".pdf"
                
                downloaded_pdf_path = download_pdf(download_url, pmc_url, temp_dir_name, pdf_filename)

                if downloaded_pdf_path:
                    gcs_blob_name = f"{gcs_folder_prefix}{pdf_filename}"
                    upload_to_gcs(bucket, downloaded_pdf_path, gcs_blob_name)
                else:
                    print(f"[FAIL] Failed to download PDF for article: {display_title}")

            except Exception as e_row:
                title_for_error = row.get('title', f"unavailable_title_at_row_{index}")
                print(f"[ERROR] An critical error occurred processing row {index} (Title: {title_for_error}): {e_row}")

    print(f"\n[INFO] Processing complete. Temporary directory {temp_dir_name} has been removed.")

def main():
    """
    Main execution function to run the entire article download and upload pipeline.
    """
    gcs_bucket_name, gcs_folder_prefix = load_configuration()
    bucket = initialize_gcs_bucket(gcs_bucket_name)
    database = load_source_data("pasc_pubmed.csv")

    if database is None:
        return # Exit if data loading failed

    # --- Processing Phase ---
    process_articles(database, bucket, gcs_folder_prefix)

if __name__ == "__main__":
    main()