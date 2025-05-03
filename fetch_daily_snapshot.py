import os
import requests
import pandas as pd
import json
import re
from datetime import datetime, date # Import date
import numpy as np
import glob # To find snapshot files
from bs4 import BeautifulSoup # To parse HTML for last updated date

# --- Configuration ---
LMSYS_LEADERBOARD_URL = "https://lmarena-ai-chatbot-arena-leaderboard.hf.space/"
# Directory to store daily snapshot CSV files
DATA_DIR = 'data'
# Filename template for daily snapshots
FILENAME_TEMPLATE = os.path.join(DATA_DIR, 'lmsys_snapshot_{}.csv')
SNAPSHOT_FILE_PATTERN = os.path.join(DATA_DIR, 'lmsys_snapshot_*.csv') # Pattern for glob

# --- Helper Functions ---

def get_latest_snapshot_date(data_dir):
    """Finds the latest date from snapshot filenames in the data directory."""
    latest_date = None
    try:
        snapshot_files = glob.glob(SNAPSHOT_FILE_PATTERN)
        if not snapshot_files:
            print("--- Update Check: No existing snapshots found.")
            return None

        dates = []
        for f in snapshot_files:
            try:
                date_str = os.path.basename(f).replace('lmsys_snapshot_', '').replace('.csv', '')
                dates.append(datetime.strptime(date_str, '%Y-%m-%d').date())
            except ValueError:
                print(f"--- Update Check Warning: Could not parse date from filename: {f}")
                continue # Skip files with invalid date format

        if dates:
            latest_date = max(dates)
            print(f"--- Update Check: Latest snapshot date found: {latest_date}")
        else:
            print("--- Update Check: No valid dates found in snapshot filenames.")

    except Exception as e:
        print(f"--- Update Check Error: Failed to get latest snapshot date: {e}")

    return latest_date

def extract_last_updated_date(html_content):
    """Extracts the 'Last updated' date from the HTML content."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser') # Use html.parser (built-in) or 'lxml' if installed
        # Common pattern for Gradio update time: look for text containing "Last updated"
        # This might need adjustment if the website structure changes.
        # Search in footer or specific divs/spans
        possible_elements = soup.find_all(string=re.compile(r"Last updated", re.IGNORECASE))

        if not possible_elements:
             # Fallback: Sometimes it's in a timestamp element within the config JSON script
             # This is less reliable as it might be the page load time, not data update time
             script_tag = soup.find('script', string=re.compile(r'window\.gradio_config'))
             if script_tag:
                  # Very rough extraction, might need refinement
                  match = re.search(r'"version":\s*".*?",\s*"updated_at":\s*"(\d{4}-\d{2}-\d{2})', script_tag.string)
                  if match:
                       date_str = match.group(1)
                       print(f"--- Update Check: Found 'updated_at' date in script: {date_str}")
                       return datetime.strptime(date_str, '%Y-%m-%d').date()

             print("--- Update Check Warning: Could not find 'Last updated' text on the page.")
             return None

        # Iterate through found elements to find a parsable date
        for element in possible_elements:
             # Try to find a date string like YYYY-MM-DD near the "Last updated" text
             # This regex looks for YYYY-MM-DD possibly preceded/followed by other text
             match = re.search(r'(\d{4}-\d{2}-\d{2})', element.parent.get_text())
             if match:
                 date_str = match.group(1)
                 print(f"--- Update Check: Found 'Last updated' date string: {date_str}")
                 try:
                     return datetime.strptime(date_str, '%Y-%m-%d').date()
                 except ValueError:
                     print(f"--- Update Check Warning: Failed to parse date string: {date_str}")
                     continue # Try next potential element

        print("--- Update Check Warning: Found 'Last updated' text but couldn't parse a date.")
        return None

    except Exception as e:
        print(f"--- Update Check Error: Failed to parse HTML for last updated date: {e}")
        return None


# --- standardize_provider function is removed or commented out ---
# def standardize_provider(provider_name):
#    """Standardizes provider names based on common variations."""
#    # ... (function content) ...
#    pass # Or remove the function entirely

def extract_json_from_html(html_content):
    """Extracts the Gradio config JSON embedded in the HTML script tag."""
    match = re.search(r'window\.gradio_config\s*=\s*(\{.*?\});?\s*</script>', html_content, re.DOTALL)
    if match:
        json_string = match.group(1)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Extracted string snippet: {json_string[:500]}...")
            return None
    else:
        print("Error: Could not find Gradio config JSON in HTML.")
        return None

def find_leaderboard_component(config_json):
    """Finds the component index and headers for the leaderboard data."""
    if not config_json or 'components' not in config_json:
        print("Error: Invalid or missing 'components' in config JSON.")
        return None, None
    for i, component in enumerate(config_json.get('components', [])):
        props = component.get('props', {})
        # Check based on headers containing 'Arena Score' now
        if isinstance(props.get('value'), dict) and isinstance(props.get('headers'), list):
            headers = props['headers']
            if any("arena score" in str(h).lower() for h in headers): # Check for 'Arena Score'
                print(f"Found leaderboard component at index {i} with headers: {headers}")
                if 'data' in props['value']:
                    return i, headers
                else:
                    print(f"Warning: Found component with 'Arena Score' header at index {i}, but 'data' key is missing.")
        elif isinstance(props.get('value'), list) and isinstance(props.get('headers'), list):
             headers = props['headers']
             if any("arena score" in str(h).lower() for h in headers): # Check for 'Arena Score'
                 print(f"Found leaderboard component (list type) at index {i} with headers: {headers}")
                 if props['value']:
                     return i, headers
                 else:
                     print(f"Warning: Found component (list type) with 'Arena Score' header at index {i}, but list is empty.")
    print("Error: Could not find a component with 'Arena Score' in headers containing data.") # Updated error message
    return None, None

def fetch_and_parse_lmsys_data(config_json):
    """Parses the LMSYS leaderboard data from the pre-fetched config JSON."""
    if not config_json:
        print("Error: No config JSON provided to parse.")
        return None, None # Return None for both df and headers

    component_index, headers = find_leaderboard_component(config_json)
    if component_index is None: return None, None

    try:
        component_data = config_json['components'][component_index]['props']['value']
        if isinstance(component_data, dict) and 'data' in component_data:
             raw_data = component_data['data']
        elif isinstance(component_data, list):
             raw_data = component_data
        else:
            print("Error: Unexpected data structure.")
            return None, None
        if not raw_data:
             print("Error: Extracted data list is empty.")
             return None, None

        print(f"Successfully extracted raw data with {len(raw_data)} rows.")
        df = pd.DataFrame(raw_data, columns=headers) # Use headers found
        print("Raw DataFrame columns:", df.columns.tolist())
        print("Raw DataFrame head:\n", df.head())
        return df # Return only the dataframe now
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error accessing data in JSON structure: {e}")
        return None, None


def process_lmsys_snapshot(df_raw):
    """Processes the raw DataFrame snapshot."""
    if df_raw is None or df_raw.empty:
        return None
    df = df_raw.copy()

    # --- Store original model column name if it exists ---
    original_model_col_name = None
    for col in df.columns:
        col_lower = str(col).lower()
        if 'model' in col_lower and 'votes' not in col_lower and 'stylectrl' not in col_lower and 'rank' not in col_lower:
            original_model_col_name = col
            print(f"Identified potential original model column: '{original_model_col_name}'")
            break

    # --- Standardize Column Names ---
    column_mapping = {}
    original_headers = df.columns.tolist()
    print(f"Original headers found: {original_headers}")

    for col in original_headers:
        col_lower = str(col).lower()
        if 'arena score' in col_lower:
             column_mapping[col] = 'ELO_Score'
        elif col == original_model_col_name:
             continue
        elif 'model' in col_lower and 'votes' not in col_lower and 'stylectrl' not in col_lower:
             if 'Model_Name' not in column_mapping.values():
                 column_mapping[col] = 'Model_Name'
        elif 'organization' in col_lower:
             column_mapping[col] = 'Provider'
        elif 'licence' in col_lower or 'license' in col_lower:
             column_mapping[col] = 'License'

    print(f"Applying column mapping (excluding original model col): {column_mapping}")
    df.rename(columns=column_mapping, inplace=True)
    print("Renamed columns (step 1):", df.columns.tolist())

    # --- Extract Anchor Text for Model Name ---
    if original_model_col_name and original_model_col_name in df.columns:
        print(f"Extracting anchor text from '{original_model_col_name}' into 'Model_Name'...")
        df['Model_Name'] = df[original_model_col_name].astype(str).str.extract(r'<a[^>]*>(.*?)</a>', expand=False).str.strip()
        df['Model_Name'].fillna('Unknown', inplace=True)
        print("Extraction complete. Sample Model Names:", df['Model_Name'].head().tolist())
    elif 'Model_Name' not in df.columns:
         print("Error: Could not find or create 'Model_Name' column.")
         return None

    # --- Select Required Columns ---
    required_cols = ['Model_Name', 'ELO_Score', 'Provider']
    if 'License' in df.columns:
        required_cols.append('License')

    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
         print(f"Error: Missing critical columns after processing: {missing_required}")
         if any(col in missing_required for col in ['ELO_Score', 'Provider']):
             print(f"Cannot proceed without columns: {missing_required}")
             return None

    cols_to_keep = [col for col in required_cols if col in df.columns]
    df = df[cols_to_keep].copy()
    print(f"Keeping columns: {cols_to_keep}")

    # --- Clean Data Types ---
    print("Cleaning data types...")
    df['ELO_Score'] = pd.to_numeric(df['ELO_Score'], errors='coerce')
    df['Provider'] = df['Provider'].astype(str).fillna('Unknown').str.strip()
    if 'License' in df.columns:
        df['License'] = df['License'].astype(str).fillna('Unknown').str.strip()

    # --- Handle Missing/Invalid Data ---
    initial_rows = len(df)
    df.dropna(subset=['ELO_Score', 'Model_Name', 'Provider'], inplace=True)
    df = df[df['Model_Name'].str.lower() != 'unknown']
    df = df[df['Model_Name'] != '']
    df = df[df['Provider'].str.lower() != 'unknown']
    df = df[df['Provider'] != '']
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to missing/invalid ELO/Model Name/Provider or empty strings.")

    # --- Standardize Provider Names ---
    # (Commented out as per user request)
    # print("Standardizing provider names...")
    # df['Provider'] = df['Provider'].apply(standardize_provider)

    # Sort snapshot by ELO score descending
    df.sort_values(by='ELO_Score', ascending=False, inplace=True)

    print(f"Processed snapshot has {len(df)} rows.")
    print("Processed DataFrame head:\n", df.head())
    return df

# --- Main Execution ---

def main():
    print("--- Starting Daily Snapshot Fetch Script ---")
    today_date = date.today() # Use date object for comparisons
    print(f"Today's Date: {today_date}")

    # --- Create data directory if it doesn't exist ---
    if not os.path.exists(DATA_DIR):
        print(f"Creating data directory: {DATA_DIR}")
        os.makedirs(DATA_DIR)

    # --- Get latest snapshot date ---
    latest_snapshot_date = get_latest_snapshot_date(DATA_DIR)

    # --- Fetch HTML and get 'Last Updated' date ---
    print(f"Fetching HTML from: {LMSYS_LEADERBOARD_URL}")
    html_content = None
    try:
        response = requests.get(LMSYS_LEADERBOARD_URL, timeout=30)
        response.raise_for_status()
        html_content = response.text
        print("Successfully fetched HTML content.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {LMSYS_LEADERBOARD_URL}: {e}")
        # Decide whether to exit or proceed without check
        print("Proceeding without update check due to fetch error.")
        # Optionally exit here:
        # return

    last_updated_date = None
    if html_content:
        last_updated_date = extract_last_updated_date(html_content)

    # --- Check if update is needed ---
    if latest_snapshot_date is not None:
        if last_updated_date is not None:
            if last_updated_date <= latest_snapshot_date:
                print(f"--- Update Check: No new data found. Website last updated ({last_updated_date}) on or before the latest snapshot ({latest_snapshot_date}). Skipping snapshot creation.")
                print("--- Daily Snapshot Fetch Script Finished (No Update Needed) ---")
                return # Exit script
            else:
                print(f"--- Update Check: New data potentially available. Website last updated ({last_updated_date}) after latest snapshot ({latest_snapshot_date}).")
        else:
            # Couldn't find update date on website, proceed with caution
            print("--- Update Check Warning: Could not determine website update date. Proceeding to create snapshot.")
            # Optionally, compare today's date with latest snapshot date as a fallback
            if today_date <= latest_snapshot_date:
                 print(f"--- Update Check Fallback: Today's date ({today_date}) is not after latest snapshot ({latest_snapshot_date}). Skipping snapshot creation.")
                 print("--- Daily Snapshot Fetch Script Finished (No Update Needed based on Date) ---")
                 return # Exit script

    else:
        # No snapshots exist yet, always proceed
        print("--- Update Check: No previous snapshots found. Proceeding to create first snapshot.")


    # --- Proceed with processing and saving ---
    if not html_content:
         print("Error: Cannot proceed without HTML content.")
         return

    config_json = extract_json_from_html(html_content)
    df_raw_new = fetch_and_parse_lmsys_data(config_json) # Pass config_json now
    df_processed_new = process_lmsys_snapshot(df_raw_new)

    if df_processed_new is None or df_processed_new.empty:
        print("Failed to parse or process new data from fetched HTML. No snapshot saved.")
        return # Exit if no data processed

    # --- Save Daily Snapshot ---
    snapshot_filename = FILENAME_TEMPLATE.format(today_date.strftime('%Y-%m-%d')) # Use today's date for filename
    print(f"Saving today's snapshot ({len(df_processed_new)} rows) to: {snapshot_filename}")
    try:
        df_processed_new.to_csv(snapshot_filename, index=False)
        print(f"Successfully saved snapshot: {snapshot_filename}")
    except Exception as e:
        print(f"Error saving snapshot data: {e}")

    print("--- Daily Snapshot Fetch Script Finished ---")

if __name__ == "__main__":
    main()
