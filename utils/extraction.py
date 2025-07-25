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
import pypdf # Added for PDF metadata editing

from google.cloud import storage
from dotenv import load_dotenv

# --- download_pmc_pdf function remains the same ---
def download_pmc_pdf(pdf_url, article_page_url, output_directory, desired_filename):
    """
    Downloads a PDF from PMC.
    Saves it to the specified output_directory with desired_filename.
    Returns the full path to the saved file if successful, None otherwise.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    # This experimental option helps in some environments to prevent crashes
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": output_directory,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True
    })

    driver = None
    save_path = os.path.join(output_directory, desired_filename)

    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        print(f"\n[INFO] NAVIGATING TO: {article_page_url}")
        driver.get(article_page_url)

        pdf_filename_from_url = pdf_url.split('/')[-1]
        correct_relative_href = f"pdf/{pdf_filename_from_url}"
        pdf_link_xpath = f"//a[@href='{correct_relative_href}' and @data-ga-label='pdf_download_desktop']"

        pdf_article_link_element = None

        try:
            # Wait for the primary PDF link
            pdf_article_link_element = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, pdf_link_xpath))
            )
        except TimeoutException:
            # If the primary link isn't found, try a fallback
            print(f"[ERROR] Timeout finding clickable PDF link on article page with primary XPath: {pdf_link_xpath}.")
            print(f"[INFO] Trying fallback XPath: //a[@href='{correct_relative_href}']")
            try:
                pdf_link_xpath_fallback = f"//a[@href='{correct_relative_href}']"
                pdf_article_link_element = WebDriverWait(driver, 15).until(
                    EC.element_to_be_clickable((By.XPATH, pdf_link_xpath_fallback))
                )
                print(f"[INFO] PDF link on article page found with fallback XPath!")
            except TimeoutException:
                print(f"[ERROR] Timeout finding clickable PDF link with fallback XPath either.")
                return None

        try:
            # Click the link to ensure any necessary cookies or session state are set
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'auto', block: 'center', inline: 'center'});",
                                  pdf_article_link_element)
            time.sleep(0.5)
            actions = ActionChains(driver)
            actions.move_to_element(pdf_article_link_element).click().perform()
            print(f"[INFO] Clicked PDF link on article page.")

        except Exception as click_err:
            print(f"[ERROR] An error occurred while clicking PDF link on article page: {click_err}")
            return None

        # Give a moment for any redirects or new tabs to process before grabbing cookies
        time.sleep(2)
        
        selenium_cookies = driver.get_cookies()
        request_cookies_dict = {cookie['name']: cookie['value'] for cookie in selenium_cookies}

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': driver.current_url # Use the final URL from the browser as the referer
        }

        print(f"[INFO] DOWNLOADING FROM: {pdf_url} to {save_path}")
        response = requests.get(pdf_url, headers=headers, cookies=request_cookies_dict, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' not in content_type:
            print(f"  [WARN] Content-Type is '{content_type}', not 'application/pdf'. Expected PDF at {pdf_url}")
            # Even if content-type is not PDF, we can check the file signature
            content_start = response.content[:4]
            if content_start != b'%PDF':
                 print(f"  [ERROR] File from {pdf_url} does not appear to be a PDF. Skipping.")
                 return None
            print("  [INFO] File signature is PDF, proceeding with download.")


        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[INFO] Successfully downloaded PDF to: {save_path}")
        return save_path

    except requests.exceptions.HTTPError as http_err:
        print(f"[ERROR] HTTP error during requests download: {http_err}")
        if hasattr(http_err, 'response') and http_err.response is not None:
            print(f"  Status Code: {http_err.response.status_code}")
            print(f"  Response content (first 300 bytes): {http_err.response.content[:300]}")
        return None
    except WebDriverException as e:
        print(f"[ERROR] A WebDriverException occurred: {e}")
        if driver:
            print(f"  Current URL at WebDriverException: {driver.current_url}")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in download_pmc_pdf: {e}")
        return None
    finally:
        if driver:
            driver.quit()

# --- sanitize_filename function remains the same ---
def sanitize_filename(name_str: str | None) -> str:
    """Sanitizes a string to be a valid filename."""
    if not name_str or pd.isna(name_str):
        name_str = f"untitled_article_{time.strftime('%Y%m%d%H%M%S')}"
    # Replace invalid characters with an underscore
    name_str = re.sub(r'[^\w\-_. ]', '_', name_str)
    # Replace one or more spaces with a single underscore
    name_str = re.sub(r'\s+', '_', name_str)
    # Limit filename length
    return name_str[:150]

# --- NEW function to edit PDF metadata ---
def edit_pdf_metadata(input_path, output_path, metadata):
    """
    Adds custom metadata to a PDF file using the pypdf library.

    Args:
        input_path (str): Path to the original PDF.
        output_path (str): Path to save the modified PDF.
        metadata (dict): A dictionary of metadata to add.
                         Keys should be PDF metadata keys (e.g., '/Title').

    Returns:
        str: The path to the new PDF with metadata, or None on failure.
    """
    try:
        print(f"[INFO] Reading PDF for metadata editing: {os.path.basename(input_path)}")
        reader = pypdf.PdfReader(input_path)
        writer = pypdf.PdfWriter()

        # Copy all pages from the original PDF to the writer object
        writer.clone_document_from_reader(reader)
        
        # Add or update metadata fields
        print(f"[INFO] Adding/updating metadata: {metadata}")
        writer.add_metadata(metadata)

        # Write the new PDF with updated metadata to the output file
        with open(output_path, "wb") as f_out:
            writer.write(f_out)

        print(f"[SUCCESS] Metadata successfully written to: {os.path.basename(output_path)}")
        return output_path
    except Exception as e:
        print(f"[ERROR] Failed to edit PDF metadata for {os.path.basename(input_path)}: {e}")
        return None

if __name__ == "__main__":
    load_dotenv()
    GCP_CREDENTIAL_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if GCP_CREDENTIAL_JSON and os.path.exists(GCP_CREDENTIAL_JSON):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIAL_JSON
        print(f"[INFO] Using GCS credentials from: {GCP_CREDENTIAL_JSON}")
    else:
        print("[WARN] GOOGLE_APPLICATION_CREDENTIALS not found or path is invalid. SDK might use default ADC.")

    GCS_BUCKET_NAME = "med_article"
    GCS_FOLDER_PREFIX = "27062025/"

    storage_client = None
    bucket = None
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        print(f"[INFO] Successfully connected to GCS bucket: {GCS_BUCKET_NAME}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize GCS client or bucket: {e}. File uploads will be skipped.")

    try:
        database = pd.read_csv("utils/pasc_pubmed.csv")
    except FileNotFoundError:
        print(f"[ERROR] The CSV file 'utils/pasc_pubmed.csv' was not found. Exiting.")
        exit()
    except Exception as e_csv:
        print(f"[ERROR] Failed to read or parse 'utils/pasc_pubmed.csv': {e_csv}. Exiting.")
        exit()

    with tempfile.TemporaryDirectory() as temp_dir_name:
        print(f"[INFO] Created temporary directory: {temp_dir_name}")

        for index, row in database.iterrows():
            try:
                article_title_original = row.get('title')
                pmc_url = row.get('pmc_link', '') # Default to empty string if missing
                download_url = row.get('download_link')

                display_title = str(article_title_original) if pd.notna(article_title_original) else f"untitled_article_row_{index}"
                print(f"\n--- Processing article: {display_title[:70]}... (Row {index}) ---")

                if not pmc_url or pd.isna(pmc_url) or not download_url or pd.isna(download_url):
                    print(f"[WARN] Row {index}: Missing 'pmc_link' or 'download_link'. Skipping.")
                    continue

                base_filename = sanitize_filename(article_title_original)
                pdf_filename = base_filename + ".pdf"
                
                # STEP 1: DOWNLOAD PDF
                downloaded_pdf_path = download_pmc_pdf(download_url, pmc_url, temp_dir_name, pdf_filename)

                if downloaded_pdf_path:
                    print(f"[SUCCESS] PDF downloaded to: {os.path.basename(downloaded_pdf_path)}")
                    
                    if storage_client and bucket:
                        # STEP 2: EDIT PDF METADATA
                        # Prepare the metadata values
                        pmcid = "N/A"
                        if 'PMC' in pmc_url.upper():
                            # Extracts the ID like 'PMC12345' from '.../articles/PMC12345/'
                            pmcid = pmc_url.strip().strip('/').split('/')[-1]

                        gcs_blob_name = f"{GCS_FOLDER_PREFIX}{pdf_filename}"
                        gcs_public_link = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{gcs_blob_name}"

                        metadata_to_add = {
                            '/Title': str(article_title_original) if pd.notna(article_title_original) else "N/A",
                            '/PMCID': pmcid,
                            '/GCS_Public_Link': gcs_public_link
                        }
                        
                        # Define path for the new PDF with metadata
                        edited_pdf_path = os.path.join(temp_dir_name, f"meta_{pdf_filename}")
                        
                        # Run the metadata editing function
                        final_pdf_to_upload = edit_pdf_metadata(downloaded_pdf_path, edited_pdf_path, metadata_to_add)

                        # STEP 3: UPLOAD MODIFIED PDF TO GCS
                        if final_pdf_to_upload:
                            blob = bucket.blob(gcs_blob_name)
                            try:
                                print(f"[INFO] Uploading {os.path.basename(final_pdf_to_upload)} to GCS as '{gcs_blob_name}'...")
                                blob.upload_from_filename(final_pdf_to_upload)
                                # You can make the blob public if needed, but be aware of security implications
                                # blob.make_public()
                                print(f"[SUCCESS] Upload complete. Public link (if bucket is public): {gcs_public_link}")
                            except Exception as e_upload:
                                print(f"[ERROR] Failed to upload {os.path.basename(final_pdf_to_upload)} to GCS: {e_upload}")
                        else:
                            print(f"[WARN] Could not edit metadata for {pdf_filename}. Aborting upload.")
                    else:
                        print(f"[WARN] Skipping metadata/upload for {pdf_filename} as GCS is not available.")
                else:
                    print(f"[FAIL] Failed to download PDF for article: {display_title}")

            except Exception as e_row_processing:
                current_title_for_error = row.get('title', f"unavailable_title_at_row_{index}")
                print(f"[ERROR] A critical unexpected error occurred while processing row {index} (Title: {current_title_for_error}): {e_row_processing}")
                print(f"[INFO] Skipping this item and continuing with the next.")

    print(f"\n[INFO] Processing complete. Temporary directory {temp_dir_name} and its contents have been removed.")