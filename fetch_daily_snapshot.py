import os
import requests
import pandas as pd
import json
import re
from datetime import datetime, date
import numpy as np
import glob
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
import asyncio

# --- Configuration ---
LMSYS_LEADERBOARD_URL = "https://lmarena.ai/leaderboard/text"
# Directory to store daily snapshot CSV files
DATA_DIR = 'data'
# Filename template for daily snapshots
FILENAME_TEMPLATE = os.path.join(DATA_DIR, 'lmsys_snapshot_{}.csv')
SNAPSHOT_FILE_PATTERN = os.path.join(DATA_DIR, 'lmsys_snapshot_*.csv')

# --- Helper Functions ---

def load_latest_snapshot(data_dir, snapshot_file_pattern):
    """
    Finds the latest dated snapshot file and loads it into a DataFrame.
    Returns the DataFrame and its date, or None, None if no snapshots are found.
    """
    try:
        snapshot_files = glob.glob(snapshot_file_pattern)
        if not snapshot_files:
            print("--- Snapshot Load: No existing snapshots found.")
            return None, None

        sorted_files = []
        for f in snapshot_files:
            try:
                date_str = os.path.basename(f).replace('lmsys_snapshot_', '').replace('.csv', '')
                file_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                sorted_files.append((file_date, f))
            except ValueError:
                print(f"--- Snapshot Load Warning: Could not parse date from filename: {f}")
                continue

        if not sorted_files:
             print("--- Snapshot Load: No files with valid dates found.")
             return None, None

        latest_file_date, latest_file_path = max(sorted_files)

        print(f"--- Snapshot Load: Loading latest snapshot from: {latest_file_path} (Date: {latest_file_date})")

        df_latest = pd.read_csv(latest_file_path)
        print(f"--- Snapshot Load: Successfully loaded {len(df_latest)} rows from latest snapshot.")

        return df_latest, latest_file_date

    except Exception as e:
        print(f"--- Snapshot Load Error: Failed to load latest snapshot: {e}")
        return None, None

def extract_last_updated_date(html_content):
    """
    Extracts the 'Last Updated' date from the HTML content.
    Returns the date as a datetime.date object or None if not found.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Look for text containing "Last Updated" followed by date
    try:
        # Search for various patterns where "Last Updated" might appear
        patterns = [
            r"Last Updated\s*(?:\n|\r\n?|\s)*([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})",  # "Last Updated\nAug 4, 2025"
            r"Last Updated[:\s]*([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})",  # "Last Updated: Aug 4, 2025"
            r"Last Updated[:\s]*(\d{4}-\d{2}-\d{2})",  # "Last Updated: 2025-08-04"
        ]
        
        text_content = soup.get_text()
        
        for pattern in patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                print(f"Found Last Updated date string: '{date_str}'")
                
                # Try to parse the date
                try:
                    # Handle format like "Aug 4, 2025"
                    if re.match(r"[A-Za-z]{3}\s+\d{1,2},?\s+\d{4}", date_str):
                        # Remove comma if present
                        date_str_clean = date_str.replace(',', '')
                        parsed_date = datetime.strptime(date_str_clean, '%b %d %Y').date()
                        print(f"Successfully parsed Last Updated date: {parsed_date}")
                        return parsed_date
                    # Handle format like "2025-08-04"
                    elif re.match(r"\d{4}-\d{2}-\d{2}", date_str):
                        parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        print(f"Successfully parsed Last Updated date: {parsed_date}")
                        return parsed_date
                except ValueError as e:
                    print(f"Failed to parse date '{date_str}': {e}")
                    continue
        
        print("Warning: Could not find 'Last Updated' date in HTML content")
        return None
        
    except Exception as e:
        print(f"Error extracting last updated date: {e}")
        return None

def should_fetch_new_snapshot(last_updated_date, latest_snapshot_date):
    """
    Determines whether to fetch a new snapshot based on the last updated date
    from the website and our latest snapshot date.
    
    Logic:
    - If last_updated_date is yesterday, we should get a new snapshot
    - If last_updated_date is older than yesterday, check if it's newer than our latest snapshot
    - If we have no latest snapshot, always fetch
    
    Args:
        last_updated_date (datetime.date): Date when the leaderboard was last updated
        latest_snapshot_date (datetime.date or None): Date of our latest snapshot
    
    Returns:
        tuple: (should_fetch: bool, reason: str)
    """
    today = date.today()
    yesterday = date.fromordinal(today.toordinal() - 1)
    
    if last_updated_date is None:
        return True, "Could not determine last updated date from website, fetching to be safe"
    
    if latest_snapshot_date is None:
        return True, "No existing snapshots found, fetching new snapshot"
    
    print(f"Date comparison - Today: {today}, Yesterday: {yesterday}")
    print(f"Website last updated: {last_updated_date}, Latest snapshot: {latest_snapshot_date}")
    
    # If the website was updated yesterday, we should fetch
    if last_updated_date == yesterday:
        return True, f"Website was updated yesterday ({yesterday}), fetching new snapshot"
    
    # If the website was updated earlier than yesterday, check if it's newer than our latest snapshot
    if last_updated_date < yesterday:
        if last_updated_date > latest_snapshot_date:
            return True, f"Website last updated ({last_updated_date}) is newer than our latest snapshot ({latest_snapshot_date})"
        else:
            return False, f"Website last updated ({last_updated_date}) is not newer than our latest snapshot ({latest_snapshot_date})"
    
    # If the website was updated today, it might be too fresh - let's fetch anyway
    if last_updated_date == today:
        return True, f"Website was updated today ({today}), fetching new snapshot"
    
    # If the website was updated in the future (shouldn't happen, but just in case)
    if last_updated_date > today:
        return True, f"Website shows future update date ({last_updated_date}), fetching to be safe"
    
    return False, "No clear reason to fetch new snapshot"

def parse_leaderboard_from_html(html_content):
    """
    Parses the HTML content to extract the leaderboard table using BeautifulSoup.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')

    if not table:
        print("Error: Could not find the leaderboard table in the HTML.")
        return None

    headers = []
    for th in table.find_all('th'):
        headers.append(th.get_text(strip=True))

    data = []
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        cols = [ele.get_text(strip=True) for ele in cols]
        data.append(cols)

    if not headers or not data:
        print("Error: No headers or data found in the leaderboard table.")
        return None

    df = pd.DataFrame(data, columns=headers)
    print(f"Successfully parsed {len(df)} rows from HTML table.")
    return df

def process_lmsys_snapshot(df_raw):
    """
    Processes the raw DataFrame snapshot.
    """
    if df_raw is None or df_raw.empty:
        print("No raw data provided to process.")
        return None
    df = df_raw.copy()

    # --- Identify and Standardize Column Names ---
    column_mapping = {}
    original_headers = df.columns.tolist()
    print(f"Original headers for processing: {original_headers}")

    # Map specific columns (case-insensitive, partial match)
    header_map = {col: str(col).lower() for col in original_headers}

    for original_col, lower_col_name in header_map.items():
        if 'arena score' in lower_col_name or 'score' == lower_col_name:
             column_mapping[original_col] = 'ELO_Score'
        elif 'organization' in lower_col_name:
             column_mapping[original_col] = 'Provider'
        elif 'licence' in lower_col_name or 'license' in lower_col_name:
             column_mapping[original_col] = 'License'
        elif 'model' in lower_col_name:
             column_mapping[original_col] = 'Model_Name'

    print(f"Applying column mapping: {column_mapping}")
    df.rename(columns=column_mapping, inplace=True)
    print("Renamed columns (step 1):", df.columns.tolist())

    required_cols = ['Model_Name', 'ELO_Score', 'Provider']
    if 'License' in df.columns:
        required_cols.append('License')

    cols_to_keep = [col for col in required_cols if col in df.columns]

    missing_critical = [col for col in ['Model_Name', 'ELO_Score', 'Provider'] if col not in cols_to_keep]
    if missing_critical:
         print(f"Error: Missing critical columns after processing: {missing_critical}. Cannot proceed.")
         return None

    df = df[cols_to_keep].copy()
    print(f"Keeping and ordering columns: {cols_to_keep}")

    print("Cleaning data types...")
    df['ELO_Score'] = pd.to_numeric(df['ELO_Score'], errors='coerce')

    for col in ['Model_Name', 'Provider']:
         if col in df.columns:
              df[col] = df[col].astype(str).fillna('Unknown').str.strip()
    if 'License' in df.columns:
        df['License'] = df['License'].astype(str).fillna('Unknown').str.strip()

    initial_rows = len(df)
    df.dropna(subset=['Model_Name', 'ELO_Score', 'Provider'], inplace=True)

    df = df[df['Model_Name'] != 'Unknown']
    df = df[df['Provider'] != 'Unknown']

    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows due to missing critical information or 'Unknown' values.")

    print("Final DataFrame columns:", df.columns.tolist())
    print("Final DataFrame head:\n", df.head())

    return df

async def fetch_with_crawl4ai(url):
    """
    Fetch HTML content using crawl4ai in stealth mode.
    """
    try:
        print("Attempting to fetch data using crawl4ai in stealth mode...")
        crawler = AsyncWebCrawler()
        result = await crawler.arun(
            url=url,
            stealth=True,
            wait_for_js=True,
            timeout=30
        )
        
        if result.success:
            print("Successfully fetched HTML content using crawl4ai.")
            return result.html
        else:
            print(f"crawl4ai failed: {result.error_message}")
            return None
    except Exception as e:
        print(f"Error using crawl4ai: {e}")
        return None

async def main():
    print(f"Attempting to fetch data from {LMSYS_LEADERBOARD_URL}")
    
    # First, load our latest snapshot to check dates
    print("--- Loading latest snapshot for comparison ---")
    df_latest, latest_snapshot_date = load_latest_snapshot(DATA_DIR, SNAPSHOT_FILE_PATTERN)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    html_content = None
    
    try:
        print("Trying with requests and fake user agent...")
        response = requests.get(LMSYS_LEADERBOARD_URL, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text
        print("Successfully fetched HTML content with requests.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data with requests: {e}")
        print("Falling back to crawl4ai stealth mode...")
        html_content = await fetch_with_crawl4ai(LMSYS_LEADERBOARD_URL)
    
    if html_content:
        # Save the fetched HTML content for inspection
        with open("crawled_content.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("Saved fetched HTML content to crawled_content.html")

        # Extract the last updated date from the website
        print("--- Extracting last updated date from website ---")
        last_updated_date = extract_last_updated_date(html_content)
        
        # Determine if we should fetch a new snapshot
        should_fetch, reason = should_fetch_new_snapshot(last_updated_date, latest_snapshot_date)
        print(f"--- Decision: {'FETCH' if should_fetch else 'SKIP'} - {reason} ---")
        
        if not should_fetch:
            print("Skipping snapshot creation based on date analysis.")
            return
        
        # Proceed with parsing and saving the snapshot
        df_current = parse_leaderboard_from_html(html_content)
        if df_current is not None and not df_current.empty:
            df_processed = process_lmsys_snapshot(df_current)
            if df_processed is not None and not df_processed.empty:
                os.makedirs(DATA_DIR, exist_ok=True)
                today_str = date.today().strftime('%Y-%m-%d')
                output_filename = FILENAME_TEMPLATE.format(today_str)
                df_processed.to_csv(output_filename, index=False)
                print(f"Successfully saved today's snapshot to {output_filename}")
            else:
                print("Processed DataFrame is empty or None. Not saving snapshot.")
        else:
            print("Current DataFrame is empty or None. Not saving snapshot.")
    else:
        print("Failed to fetch HTML content with all methods.")

if __name__ == "__main__":
    asyncio.run(main())

