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

# --- Helper Functions (Standardize Provider, Extract JSON, Find Component - Copied from previous script) ---

def standardize_provider(provider_name):
    """Standardizes provider names based on common variations."""
    if pd.isna(provider_name) or not isinstance(provider_name, str): return 'Unknown'
    name = provider_name.strip().lower()
    # Add more specific mappings first (expand as needed, based on your previous script)
    if name in ['openai', 'openai baseline', 'openai-internal']: return 'OpenAI'
    if name in ['google', 'google research', 'google-research', 'google-internal', 'google deepmind']: return 'Google'
    if name in ['meta', 'metaai', 'facebook', 'meta-llama', 'meta platforms inc.', 'meta platforms inc', 'llama team (meta)', 'meta/llama']: return 'Meta'
    if name in ['anthropic', 'anthropic-internal']: return 'Anthropic'
    if name in ['mistralai', 'mistral ai', 'mistral', 'mistral-ai']: return 'Mistral'
    if name in ['alibaba', 'alibaba group', 'qwen', 'tongyi qianwen', 'modelscope', 'alibaba cloud']: return 'Alibaba'
    if name in ['xai', 'x.ai', 'grok', 'xai corp.', 'xai-internal']: return 'xAI'
    if name in ['deepseek', 'deepseek ai', 'deepseek-ai']: return 'DeepSeek'
    if name in ['tencent', 'hunyuan', 'tencent ai lab', 'tencent research']: return 'Tencent'
    if name in ['amazon', 'aws', 'titan', 'amazon-internal']: return 'Amazon'
    if name in ['cohere']: return 'Cohere'
    if name in ['01.ai', '01 ai', 'yi technologies', 'lingyi万物', '01-ai', 'yi', 'yi large', '01ai']: return '01 AI'
    if name in ['databricks', 'dbrx']: return 'Databricks'
    if name in ['zhipuai', 'zhipu.ai', 'zhipu', 'zhipu ai', 'chatglm', 'glm']: return 'Zhipu AI'
    if name in ['microsoft', 'msft', 'phi', 'microsoft research', 'microsoft-internal']: return 'Microsoft'
    if name in ['reka ai', 'reka']: return 'Reka AI'
    if name in ['nvidia', 'nemotron', 'nvidia research']: return 'Nvidia'
    if name in ['together ai', 'together computer', 'together']: return 'Together AI'
    if name in ['upstage', 'solar']: return 'Upstage'
    if name in ['intel']: return 'Intel'
    if name in ['snowflake', 'arctic']: return 'Snowflake'
    if name in ['baichuan', 'baichuan intelligent technology']: return 'Baichuan'
    if name in ['adept', 'adept ai']: return 'Adept'
    if name in ['teknium inc.', 'teknium']: return 'Teknium'
    if name in ['kaist', 'korea advanced institute of science and technology']: return 'KAIST'
    if name in ['eleutherai', 'eleuther ai']: return 'EleutherAI'
    if name in ['bigcode project', 'bigcode', 'big code']: return 'BigCode'
    if name in ['salesforce', 'xgen']: return 'Salesforce'
    if name in ['allen institute for ai', 'ai2', 'allenai']: return 'Allen Institute for AI'
    if name in ['hugging face', 'huggingface', 'huggingfaceh4']: return 'Hugging Face'
    if name in ['lmsys', 'vicuna', 'lmsys.org']: return 'LMSys' # Treat Vicuna as LMSys project
    if name in ['uc berkeley', 'berkeley', 'sky lab, uc berkeley']: return 'UC Berkeley'
    if name in ['stanford', 'stanford university']: return 'Stanford'
    # Fallback to original, cleaned name, Title Cased
    return provider_name.strip().title()

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
        if isinstance(props.get('value'), dict) and isinstance(props.get('headers'), list):
            headers = props['headers']
            if any("elo" in str(h).lower() for h in headers):
                print(f"Found leaderboard component at index {i} with headers: {headers}")
                if 'data' in props['value']:
                    return i, headers
                else:
                    print(f"Warning: Found component with 'Elo' header at index {i}, but 'data' key is missing.")
        elif isinstance(props.get('value'), list) and isinstance(props.get('headers'), list):
             headers = props['headers']
             if any("elo" in str(h).lower() for h in headers):
                 print(f"Found leaderboard component (list type) at index {i} with headers: {headers}")
                 if props['value']:
                     return i, headers
                 else:
                     print(f"Warning: Found component (list type) with 'Elo' header at index {i}, but list is empty.")
    print("Error: Could not find a component with 'Elo' in headers containing data.")
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
    column_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if 'model' in col_lower and 'votes' not in col_lower:
             column_mapping[col] = 'Model_Name'
        elif 'elo' in col_lower and 'rating' in col_lower:
             column_mapping[col] = 'ELO_Score'
        elif 'organization' in col_lower:
             column_mapping[col] = 'Provider'
        elif 'licence' in col_lower or 'license' in col_lower:
             column_mapping[col] = 'License'
    print(f"Applying column mapping: {column_mapping}")
    df.rename(columns=column_mapping, inplace=True)
    print("Renamed columns:", df.columns.tolist())

    required_cols = ['Model_Name', 'ELO_Score', 'Provider']
    if 'License' in df.columns: required_cols.append('License')

    missing_required = [col for col in required_cols if col not in df.columns and col != 'License']
    if missing_required:
        print(f"Error: Missing critical columns after renaming: {missing_required}")
        html_model_col = next((c for c in df_raw.columns if 'model' in str(c).lower() and '<a href' in str(df_raw[c].iloc[0]).lower()), None)
        if 'Model_Name' in missing_required and html_model_col:
             print(f"Attempting to extract Model_Name from HTML in column '{html_model_col}'")
             try:
                  df['Model_Name'] = df_raw[html_model_col].str.extract(r'<a[^>]*>(.*?)</a>', expand=False).str.strip()
                  print("Successfully extracted Model_Name from HTML.")
                  required_cols.insert(0, 'Model_Name')
                  missing_required.remove('Model_Name')
             except Exception as e:
                  print(f"Failed to extract Model_Name from HTML: {e}")
                  return None
        else:
             return None

    cols_to_keep = [col for col in required_cols if col in df.columns]
    df = df[cols_to_keep].copy()
    print(f"Keeping columns: {cols_to_keep}")

    print("Cleaning data types...")
    df['ELO_Score'] = pd.to_numeric(df['ELO_Score'], errors='coerce')
    df['Model_Name'] = df['Model_Name'].astype(str).fillna('Unknown').str.strip()
    df['Provider'] = df['Provider'].astype(str).fillna('Unknown').str.strip()
    if 'License' in df.columns:
        df['License'] = df['License'].astype(str).fillna('Unknown').str.strip()

    initial_rows = len(df)
    df.dropna(subset=['ELO_Score', 'Model_Name'], inplace=True)
    df = df[df['Model_Name'].str.lower() != 'unknown']
    df = df[df['Model_Name'] != '']
    df = df[df['Provider'] != '']
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to missing/invalid ELO/Model Name or empty strings.")

    print("Standardizing provider names...")
    df['Provider'] = df['Provider'].apply(standardize_provider)

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

    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        print(f"Creating data directory: {DATA_DIR}")
        os.makedirs(DATA_DIR)

    # --- Fetch and Process New Data ---
    df_raw_new = fetch_lmsys_data(LMSYS_LEADERBOARD_URL)
    df_processed_new = process_lmsys_snapshot(df_raw_new)

    if df_processed_new is None or df_processed_new.empty:
        print("Failed to fetch or process new data. No snapshot saved.")
        return # Exit if no new data

    # --- Save Daily Snapshot ---
    snapshot_filename = FILENAME_TEMPLATE.format(today_str)
    print(f"Saving today's snapshot ({len(df_processed_new)} rows) to: {snapshot_filename}")
    try:
        # Save without the Snapshot_Date column, as the date is in the filename
        df_processed_new.to_csv(snapshot_filename, index=False)
        print(f"Successfully saved snapshot: {snapshot_filename}")
    except Exception as e:
        print(f"Error saving snapshot data: {e}")

    print("--- Daily Snapshot Fetch Script Finished ---")

if __name__ == "__main__":
    main()
