import os
import requests
import pandas as pd
import json
import re
from datetime import datetime
import numpy as np

# --- Configuration ---
LMSYS_LEADERBOARD_URL = "https://lmarena-ai-chatbot-arena-leaderboard.hf.space/"
# Directory to store daily snapshot CSV files
DATA_DIR = 'data'
# Filename template for daily snapshots
FILENAME_TEMPLATE = os.path.join(DATA_DIR, 'lmsys_snapshot_{}.csv')

# --- Helper Functions ---

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

def fetch_lmsys_data(url):
    """Fetches and processes the LMSYS leaderboard data."""
    print(f"Fetching data from: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        print("Successfully fetched HTML content.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

    config_json = extract_json_from_html(response.text)
    if not config_json: return None

    component_index, headers = find_leaderboard_component(config_json)
    if component_index is None: return None

    try:
        component_data = config_json['components'][component_index]['props']['value']
        if isinstance(component_data, dict) and 'data' in component_data:
             raw_data = component_data['data']
        elif isinstance(component_data, list):
             raw_data = component_data
        else:
            print("Error: Unexpected data structure.")
            return None
        if not raw_data:
             print("Error: Extracted data list is empty.")
             return None
        print(f"Successfully extracted raw data with {len(raw_data)} rows.")
        df = pd.DataFrame(raw_data, columns=headers)
        print("Raw DataFrame columns:", df.columns.tolist())
        print("Raw DataFrame head:\n", df.head())
        return df
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error accessing data in JSON structure: {e}")
        return None

def process_lmsys_snapshot(df_raw):
    """Processes the raw DataFrame snapshot."""
    if df_raw is None or df_raw.empty:
        return None
    df = df_raw.copy()

    # --- Store original model column name if it exists ---
    # Find the original column that likely contains the model HTML link
    original_model_col_name = None
    for col in df.columns:
        col_lower = str(col).lower()
        # Heuristic: Look for 'model' but exclude columns clearly not the primary model name
        if 'model' in col_lower and 'votes' not in col_lower and 'stylectrl' not in col_lower and 'rank' not in col_lower:
            original_model_col_name = col
            print(f"Identified potential original model column: '{original_model_col_name}'")
            break # Assume the first match is the correct one

    # --- Standardize Column Names ---
    column_mapping = {}
    original_headers = df.columns.tolist()
    print(f"Original headers found: {original_headers}")

    for col in original_headers:
        col_lower = str(col).lower()
        if 'arena score' in col_lower:
             column_mapping[col] = 'ELO_Score'
        # Don't map the original model column here yet if we found it
        elif col == original_model_col_name:
             continue # Skip mapping the original model column for now
        elif 'model' in col_lower and 'votes' not in col_lower and 'stylectrl' not in col_lower:
             # This handles cases where the heuristic above might miss, but map it only if needed
             if 'Model_Name' not in column_mapping.values(): # Avoid overwriting if already mapped somehow
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
        # Use regex to extract text between > and </a>, strip whitespace
        # Handle potential errors if the cell doesn't contain the expected HTML
        df['Model_Name'] = df[original_model_col_name].astype(str).str.extract(r'<a[^>]*>(.*?)</a>', expand=False).str.strip()
        # Fill any NaNs resulting from failed extraction (e.g., if some rows didn't have links)
        df['Model_Name'].fillna('Unknown', inplace=True)
        print("Extraction complete. Sample Model Names:", df['Model_Name'].head().tolist())
        # Optionally drop the original HTML column if desired
        # df.drop(columns=[original_model_col_name], inplace=True)
        # print(f"Dropped original model column: '{original_model_col_name}'")
    elif 'Model_Name' not in df.columns:
         print("Error: Could not find or create 'Model_Name' column.")
         return None # Cannot proceed without model name


    # --- Select Required Columns ---
    # Now Model_Name should exist from the extraction step
    required_cols = ['Model_Name', 'ELO_Score', 'Provider']
    if 'License' in df.columns:
        required_cols.append('License')

    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
         # This check should ideally not be needed now for Model_Name, but kept defensively
         print(f"Error: Missing critical columns after processing: {missing_required}")
         if any(col in missing_required for col in ['ELO_Score', 'Provider']):
             print(f"Cannot proceed without columns: {missing_required}")
             return None

    # Keep only the columns we need/have mapped
    cols_to_keep = [col for col in required_cols if col in df.columns]
    df = df[cols_to_keep].copy()
    print(f"Keeping columns: {cols_to_keep}")

    # --- Clean Data Types ---
    print("Cleaning data types...")
    df['ELO_Score'] = pd.to_numeric(df['ELO_Score'], errors='coerce')
    # Model_Name is already string from extraction/fillna
    # Provider needs cleaning
    df['Provider'] = df['Provider'].astype(str).fillna('Unknown').str.strip()
    if 'License' in df.columns:
        df['License'] = df['License'].astype(str).fillna('Unknown').str.strip()

    # --- Handle Missing/Invalid Data ---
    initial_rows = len(df)
    df.dropna(subset=['ELO_Score', 'Model_Name', 'Provider'], inplace=True)
    df = df[df['Model_Name'].str.lower() != 'unknown'] # Remove rows where extraction failed -> 'Unknown'
    df = df[df['Model_Name'] != '']
    df = df[df['Provider'].str.lower() != 'unknown']
    df = df[df['Provider'] != '']
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to missing/invalid ELO/Model Name/Provider or empty strings.")

    # --- Standardize Provider Names ---
    # print("Standardizing provider names...") # No longer standardizing
    # df['Provider'] = df['Provider'].apply(standardize_provider) # *** THIS LINE IS REMOVED/COMMENTED OUT ***

    # Sort snapshot by ELO score descending
    df.sort_values(by='ELO_Score', ascending=False, inplace=True)

    print(f"Processed snapshot has {len(df)} rows.")
    print("Processed DataFrame head:\n", df.head())
    return df

# --- Main Execution ---

def main():
    print("--- Starting Daily Snapshot Fetch Script ---")
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"Snapshot Date: {today_str}")

    if not os.path.exists(DATA_DIR):
        print(f"Creating data directory: {DATA_DIR}")
        os.makedirs(DATA_DIR)

    df_raw_new = fetch_lmsys_data(LMSYS_LEADERBOARD_URL)
    df_processed_new = process_lmsys_snapshot(df_raw_new)

    if df_processed_new is None or df_processed_new.empty:
        print("Failed to fetch or process new data. No snapshot saved.")
        return

    snapshot_filename = FILENAME_TEMPLATE.format(today_str)
    print(f"Saving today's snapshot ({len(df_processed_new)} rows) to: {snapshot_filename}")
    try:
        df_processed_new.to_csv(snapshot_filename, index=False)
        print(f"Successfully saved snapshot: {snapshot_filename}")
    except Exception as e:
        print(f"Error saving snapshot data: {e}")

    print("--- Daily Snapshot Fetch Script Finished ---")

if __name__ == "__main__":
    main()
