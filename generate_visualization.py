import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob # To find all snapshot files
from datetime import datetime

# --- Configuration ---
DATA_DIR = 'data' # Directory where snapshots are stored
SNAPSHOT_FILE_PATTERN = os.path.join(DATA_DIR, 'lmsys_snapshot_*.csv')
OLD_COMBINED_FILE = 'combined_raw_lmsys_data.csv' # Path to the old combined file
OUTPUT_HTML_FILE = 'index.html'
# Columns expected in the snapshot CSVs
EXPECTED_SNAPSHOT_COLUMNS = ['Model_Name', 'ELO_Score', 'Provider'] # Add 'License' if saved
# Columns expected/needed from the old combined file
EXPECTED_OLD_COLUMNS = ['Model_Name', 'ELO_Score', 'Provider', 'snapshot_date_extracted']

# --- Helper Function (Standardize Provider - Copied for consistency) ---
# (Mappings omitted for brevity - ensure they match fetch script if re-enabled)
def standardize_provider(provider_name):
    """Standardizes provider names based on common variations."""
    if pd.isna(provider_name) or not isinstance(provider_name, str): return 'Unknown'
    name = provider_name.strip().lower()
    # ... include all mappings from fetch_daily_snapshot.py ...
    if name in ['stanford', 'stanford university']: return 'Stanford'
    return provider_name.strip().title()

# --- Load and Combine Data ---

def load_data_sources(snapshot_pattern, old_file_path):
    """Loads data from daily snapshots and optionally the old combined CSV."""
    all_data = []
    processed_files = 0
    skipped_files = 0

    # 1. Load Daily Snapshots
    print(f"--- Debug: Looking for files matching pattern: {snapshot_pattern}")
    all_snapshot_files = glob.glob(snapshot_pattern)
    print(f"--- Debug: Found snapshot files: {all_snapshot_files}")

    if not all_snapshot_files:
        print("No daily snapshot files found.")
    else:
        print(f"Found {len(all_snapshot_files)} potential snapshot files.")
        for f in sorted(all_snapshot_files):
            print(f"\n--- Debug: Processing snapshot file: {f}")
            try:
                base_name = os.path.basename(f)
                date_str = base_name.replace('lmsys_snapshot_', '').replace('.csv', '')
                print(f"--- Debug: Extracted date string: '{date_str}'")
                snapshot_date = pd.to_datetime(date_str).date()
                print(f"--- Debug: Parsed snapshot date: {snapshot_date}")

                df_snap = pd.read_csv(f)
                print(f"--- Debug: Loaded Snapshot DataFrame shape: {df_snap.shape}")
                print(f"--- Debug: Loaded Snapshot DataFrame columns: {df_snap.columns.tolist()}")

                if df_snap.empty:
                    print(f"--- Debug: Skipping empty snapshot file: {f}")
                    skipped_files += 1
                    continue

                missing_cols = [col for col in EXPECTED_SNAPSHOT_COLUMNS if col not in df_snap.columns]
                if missing_cols:
                     print(f"--- Debug: Skipping snapshot file {f} due to missing columns: {missing_cols}. Expected: {EXPECTED_SNAPSHOT_COLUMNS}")
                     skipped_files += 1
                     continue

                df_snap['Snapshot_Date'] = snapshot_date
                all_data.append(df_snap)
                processed_files += 1
                print(f"--- Debug: Successfully processed snapshot data for {snapshot_date}")

            except Exception as e:
                print(f"--- Debug: Error processing snapshot file {f}: {e}")
                skipped_files += 1

        print(f"\n--- Debug: Finished processing snapshots ---")
        print(f"--- Debug: Loaded data from {processed_files} snapshot files.")
        if skipped_files > 0: print(f"--- Debug: Skipped {skipped_files} snapshot files.")

    # 2. Load Old Combined File (if exists)
    print(f"\n--- Debug: Checking for old combined file: {old_file_path}")
    if os.path.exists(old_file_path):
        print(f"--- Debug: Found old combined file. Attempting to load...")
        try:
            df_old = pd.read_csv(old_file_path)
            print(f"--- Debug: Loaded Old Combined DataFrame shape: {df_old.shape}")
            print(f"--- Debug: Loaded Old Combined DataFrame columns: {df_old.columns.tolist()}")

            if not df_old.empty:
                # --- Rename and Select Columns from Old File ---
                # Check for the date column (might be named differently)
                date_col_found = None
                if 'snapshot_date_extracted' in df_old.columns:
                    date_col_found = 'snapshot_date_extracted'
                elif 'Snapshot_Date' in df_old.columns: # Or maybe it was already named correctly
                    date_col_found = 'Snapshot_Date'
                # Add other potential old date column names if necessary

                if not date_col_found:
                    print(f"--- Debug: Skipping old file {old_file_path} - Could not find a suitable date column (e.g., 'snapshot_date_extracted' or 'Snapshot_Date').")
                else:
                    # Rename the found date column to the standard 'Snapshot_Date'
                    if date_col_found != 'Snapshot_Date':
                        df_old.rename(columns={date_col_found: 'Snapshot_Date'}, inplace=True)
                        print(f"--- Debug: Renamed column '{date_col_found}' to 'Snapshot_Date'.")

                    # Check for other essential columns (Model, ELO, Provider) - adapt names if needed
                    # Example: If old file used 'model' instead of 'Model_Name'
                    rename_map_old = {}
                    if 'model' in df_old.columns and 'Model_Name' not in df_old.columns:
                        rename_map_old['model'] = 'Model_Name'
                    if 'arena_score' in df_old.columns and 'ELO_Score' not in df_old.columns:
                         rename_map_old['arena_score'] = 'ELO_Score'
                    # Add other potential renames based on the old file's structure

                    if rename_map_old:
                         print(f"--- Debug: Applying renames to old data: {rename_map_old}")
                         df_old.rename(columns=rename_map_old, inplace=True)

                    # Verify essential columns are present after potential renames
                    old_essential_cols = ['Model_Name', 'ELO_Score', 'Provider', 'Snapshot_Date']
                    missing_old_cols = [col for col in old_essential_cols if col not in df_old.columns]
                    if missing_old_cols:
                        print(f"--- Debug: Skipping old file {old_file_path} - Missing essential columns after renaming: {missing_old_cols}")
                    else:
                        # Select only the necessary columns
                        cols_to_keep_old = [col for col in old_essential_cols if col in df_old.columns]
                        # Include License if it exists
                        if 'License' in df_old.columns: cols_to_keep_old.append('License')

                        df_old_processed = df_old[cols_to_keep_old].copy()

                        # Parse the date column
                        df_old_processed['Snapshot_Date'] = pd.to_datetime(df_old_processed['Snapshot_Date'], errors='coerce').dt.date
                        df_old_processed.dropna(subset=['Snapshot_Date'], inplace=True) # Drop rows where date parsing failed

                        print(f"--- Debug: Successfully processed old combined file. Shape: {df_old_processed.shape}")
                        all_data.append(df_old_processed)
            else:
                print(f"--- Debug: Old combined file {old_file_path} is empty.")

        except Exception as e:
            print(f"--- Debug: Error processing old combined file {old_file_path}: {e}")
    else:
        print(f"--- Debug: Old combined file not found at {old_file_path}.")


    # 3. Combine and Final Processing
    if not all_data:
        print("--- Debug: No data loaded from any source.")
        return pd.DataFrame()

    print("\n--- Debug: Concatenating all loaded dataframes...")
    # Use outer join to keep all columns from all sources initially
    df_combined = pd.concat(all_data, ignore_index=True, join='outer', sort=False)

    print(f"--- Debug: Combined DataFrame shape before cleaning: {df_combined.shape}")
    print(f"--- Debug: Combined DataFrame columns: {df_combined.columns.tolist()}")
    print(f"--- Debug: Combined DataFrame head before cleaning:\n{df_combined.head()}")

    # Ensure correct types after combining
    print("--- Debug: Converting data types...")
    df_combined['Snapshot_Date'] = pd.to_datetime(df_combined['Snapshot_Date'], errors='coerce')
    df_combined['ELO_Score'] = pd.to_numeric(df_combined['ELO_Score'], errors='coerce')
    # Fill NaNs in string columns that might result from outer join or loading issues
    df_combined['Model_Name'] = df_combined['Model_Name'].astype(str).fillna('Unknown')
    df_combined['Provider'] = df_combined['Provider'].astype(str).fillna('Unknown')
    if 'License' in df_combined.columns:
         df_combined['License'] = df_combined['License'].astype(str).fillna('Unknown')

    # Optional: Re-apply standardization defensively
    print("--- Debug: Re-applying provider standardization...")
    df_combined['Provider'] = df_combined['Provider'].apply(standardize_provider)

    # Drop rows with invalid essential data AFTER type conversion
    initial_rows_before_dropna = len(df_combined)
    df_combined.dropna(subset=['ELO_Score', 'Snapshot_Date', 'Model_Name', 'Provider'], inplace=True)
    # Also drop rows where key identifiers became empty strings or 'Unknown' inappropriately
    df_combined = df_combined[df_combined['Model_Name'].str.strip().str.lower() != 'unknown']
    df_combined = df_combined[df_combined['Model_Name'].str.strip() != '']
    df_combined = df_combined[df_combined['Provider'].str.strip().str.lower() != 'unknown']
    df_combined = df_combined[df_combined['Provider'].str.strip() != '']

    rows_dropped = initial_rows_before_dropna - len(df_combined)
    if rows_dropped > 0:
        print(f"--- Debug: Dropped {rows_dropped} rows due to NaN/Invalid ELO, Date, Model, or Provider.")

    # 4. Deduplicate across all sources
    print(f"--- Debug: Deduplicating combined data ({len(df_combined)} rows)...")
    # Sort by date ascending, then ELO descending to keep highest score for a given day
    df_combined.sort_values(by=['Snapshot_Date', 'Model_Name', 'Provider', 'ELO_Score'],
                            ascending=[True, True, True, False],
                            inplace=True)
    # Keep the FIRST entry after sorting, which has the highest ELO for that day/model/provider
    df_deduplicated = df_combined.drop_duplicates(subset=['Snapshot_Date', 'Model_Name', 'Provider'], keep='first')
    deduplicated_rows_count = len(df_combined) - len(df_deduplicated)
    if deduplicated_rows_count > 0:
        print(f"--- Debug: Removed {deduplicated_rows_count} duplicate entries (same day, model, provider), keeping highest ELO.")


    # Final sort for visualization
    df_final = df_deduplicated.sort_values(by=['Snapshot_Date', 'ELO_Score'], ascending=[True, False])


    print(f"--- Debug: Final combined DataFrame shape: {df_final.shape}")
    if not df_final.empty:
        print(f"--- Debug: Final combined DataFrame head:\n{df_final.head()}")
        print(f"--- Debug: Final combined DataFrame date range: {df_final['Snapshot_Date'].min()} to {df_final['Snapshot_Date'].max()}")
    else:
        print("--- Debug: Final combined DataFrame is empty.")

    return df_final


# --- Generate Plotly Visualization (Code remains the same as before) ---
# ... (generate_plot function remains exactly the same as the previous version) ...
def generate_plot(df_processed):
    """Generates the Plotly bar chart race visualization."""
    if df_processed is None or df_processed.empty:
        print("No processed data available to generate visualization.")
        # Create an empty placeholder HTML or return None
        try:
            with open(OUTPUT_HTML_FILE, 'w') as f:
                f.write("<html><body>No data available to generate visualization. Check logs for errors loading snapshot data.</body></html>")
            print(f"Created empty placeholder file: {OUTPUT_HTML_FILE}")
        except Exception as e:
            print(f"Error writing placeholder HTML file: {e}")
        return None

    print("\n--- Generating Plotly Visualization ---")

    # Config with REVISED Corporate Identity Colors (Same as before)
    COLOR_MAPPING = {
        'OpenAI': '#10A37F', 'Google': '#4285F4', 'Meta': '#8A2BE2', 'Anthropic': '#D97706',
        'Mistral': '#FF4B4B', 'Alibaba': '#FF8800', 'xAI': '#333333', 'DeepSeek': '#00BFFF',
        'Tencent': '#0052D9', 'Amazon': '#FF9900', 'Cohere': '#8E44AD', '01 AI': '#2ECC71',
        'Zhipu AI': '#2962FF', 'Databricks': '#FF3621', 'Microsoft': '#00A4EF', 'Reka AI': '#6B5B95',
        'Nvidia': '#76B900', 'Together AI': '#FF6B6B', 'Upstage': '#4ECDC4', 'Snowflake': '#29B5E8',
        'Baichuan': '#A0522D', 'Adept': '#FF69B4', 'Teknium': '#4682B4', 'KAIST': '#00427E',
        'EleutherAI': '#000000', 'BigCode': '#FFD700', 'Salesforce': '#00A1E0',
        'Allen Institute for AI': '#00B398', 'Hugging Face': '#FFD700', 'LMSys': '#FBBD01',
        'UC Berkeley': '#003262', 'Stanford': '#8C1515',
        'Unknown': '#B0B0B0', 'Other': '#CCCCCC'
    }

    # Prepare dates
    df_processed['Snapshot_Date'] = pd.to_datetime(df_processed['Snapshot_Date'])
    unique_dates = sorted(df_processed['Snapshot_Date'].unique())
    if not unique_dates:
        print("Error: No unique dates found in processed data. Cannot generate plot.")
        with open(OUTPUT_HTML_FILE, 'w') as f:
            f.write("<html><body>No unique dates found in data. Cannot generate visualization.</body></html>")
        print(f"Created empty placeholder file: {OUTPUT_HTML_FILE}")
        return None

    date_strings = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in unique_dates]
    max_elo_overall = df_processed['ELO_Score'].max()
    min_elo_for_annotation_calc = df_processed['ELO_Score'].min() * 0.9 if pd.notna(df_processed['ELO_Score'].min()) else 0
    min_elo_for_annotation = max(min_elo_for_annotation_calc, 10)

    fig = go.Figure()
    frames = []
    slider_steps = []
    initial_top_df = pd.DataFrame()

    hovertemplate = (
        "<b>Provider:</b> %{customdata[0]}<br>" +
        "<b>Model:</b> %{customdata[1]}<br>" +
        "<b>ELO Score:</b> %{customdata[2]:.1f}<extra></extra>"
    )

    print(f"Creating {len(unique_dates)} frames...")
    for idx, date in enumerate(unique_dates):
        date_str = date_strings[idx]
        snapshot_df = df_processed[df_processed['Snapshot_Date'] == date].copy()
        snapshot_sorted_all = snapshot_df.sort_values(by='ELO_Score', ascending=False)
        top_df = snapshot_sorted_all.head(15).copy() # Top 15 models per frame

        if top_df.empty:
            print(f"Warning: No data for date {date_str}, skipping frame.")
            continue # Skip frame if no data for this date

        # Create unique Y-category including score for sorting/identification
        top_df['Y_Category'] = top_df['Provider'] + ' | ' + top_df['Model_Name'] + ' | ' + top_df['ELO_Score'].round(1).astype(str)

        # Store the data for the last valid frame found
        initial_top_df = top_df # Keep updating until the last loop iteration

        # --- Create Bar Trace for the Frame ---
        frame_plotly_data = go.Bar(
            x=top_df['ELO_Score'], y=top_df['Y_Category'], orientation='h',
            text=top_df['ELO_Score'].round(1).astype(str), textposition='outside',
            hoverinfo='skip', customdata=top_df[['Provider', 'Model_Name', 'ELO_Score']],
            hovertemplate=hovertemplate,
            marker_color=[COLOR_MAPPING.get(p, COLOR_MAPPING['Other']) if p != 'Unknown' else COLOR_MAPPING['Unknown'] for p in top_df['Provider']],
            name=date_str
        )

        # --- Annotations for Frame ---
        frame_annotations = []
        frame_annotations.append(dict(x=0.5, y=1.10, xref='paper', yref='paper', text=f"<b>{date_str}</b>", showarrow=False, font=dict(size=18, color="#333333"), align='center'))
        for i in range(len(top_df)):
            row = top_df.iloc[i]
            frame_annotations.append(dict(
                x=min_elo_for_annotation, y=row['Y_Category'], xref='x', yref='y',
                text=f"<i>{row['Model_Name']}</i>", showarrow=False,
                font=dict(size=8, color='white'), align='left', xanchor='left', xshift=5
            ))

        # --- Y-axis Layout for Frame ---
        frame_yaxis_layout = dict(
            tickmode='array', tickvals=top_df['Y_Category'].tolist(), ticktext=top_df['Provider'].tolist(),
            autorange="reversed", tickfont=dict(size=10), automargin=True, ticklabelstandoff=10
        )

        frame_layout = go.Layout(annotations=frame_annotations, yaxis=frame_yaxis_layout)
        frames.append(go.Frame(data=[frame_plotly_data], name=date_str, layout=frame_layout))

        # --- Slider Step ---
        slider_step = {"args": [[date_str], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}], "label": date_str, "method": "animate"}
        slider_steps.append(slider_step)

    # --- Add Initial Data Trace (based on the data from the *last successful frame*) ---
    initial_annotations = []
    initial_yaxis_layout = {}
    last_valid_date_str = date_strings[unique_dates.index(initial_top_df['Snapshot_Date'].iloc[0])] if not initial_top_df.empty else "Latest"

    if not initial_top_df.empty:
        initial_annotations.append(dict(x=0.5, y=1.10, xref='paper', yref='paper', text=f"<b>{last_valid_date_str}</b>", showarrow=False, font=dict(size=18, color="#333333"), align='center'))
        for i in range(len(initial_top_df)):
            row = initial_top_df.iloc[i]
            initial_annotations.append(dict(x=min_elo_for_annotation, y=row['Y_Category'], xref='x', yref='y', text=f"<i>{row['Model_Name']}</i>", showarrow=False, font=dict(size=8, color='white'), align='left', xanchor='left', xshift=5))

        initial_yaxis_layout = dict(
            tickmode='array', tickvals=initial_top_df['Y_Category'].tolist(), ticktext=initial_top_df['Provider'].tolist(),
            autorange="reversed", tickfont=dict(size=10), automargin=True, ticklabelstandoff=10
        )

        fig.add_trace(go.Bar(
            x=initial_top_df['ELO_Score'], y=initial_top_df['Y_Category'], orientation='h',
            text=initial_top_df['ELO_Score'].round(1).astype(str), textposition='outside',
            hoverinfo='skip', customdata=initial_top_df[['Provider', 'Model_Name', 'ELO_Score']],
            hovertemplate=hovertemplate,
            marker_color=[COLOR_MAPPING.get(p, COLOR_MAPPING['Other']) if p != 'Unknown' else COLOR_MAPPING['Unknown'] for p in initial_top_df['Provider']],
            name=last_valid_date_str
        ))
    else:
        print("Warning: Could not determine initial frame data (initial_top_df is empty). Plot might be empty initially.")
        if not frames:
             fig.add_annotation(text="No data available for visualization.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))


    # --- Configure Overall Layout ---
    fig.frames = frames
    fig.update_layout(
        title=dict(
            text=f"LMSYS Chatbot Arena - Top {len(initial_top_df) if not initial_top_df.empty else 'N'} Models by ELO Score<br><sup>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</sup>",
            font=dict(size=20), x=0.5, xanchor='center'
        ),
        xaxis_title="ELO Score", yaxis_title="Provider",
        xaxis=dict(range=[min_elo_for_annotation * 0.98, max_elo_overall * 1.05 if pd.notna(max_elo_overall) else 1500]),
        yaxis=initial_yaxis_layout, # Apply layout from the last valid frame
        height=800, margin=dict(l=150, r=50, t=150, b=100),
        hovermode='closest', hoverlabel=dict(bgcolor="white", font_size=12),
        annotations=initial_annotations,
        sliders=[{"active": len(slider_steps) - 1 if slider_steps else 0, "currentvalue": {"prefix": "Date: ", "visible": True, "xanchor": "right"}, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.05, "y": 0, "steps": slider_steps}],
        updatemenus=[{"buttons": [{"args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300, "easing": "linear"}}], "label": "► Play", "method": "animate"}, {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}], "label": "❚❚ Pause", "method": "animate"}], "direction": "left", "pad": {"r": 10, "t": 80}, "showactive": False, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0.08, "yanchor": "top"}]
    )

    # --- Save HTML ---
    try:
        fig.write_html(OUTPUT_HTML_FILE)
        print(f"Interactive HTML saved: {OUTPUT_HTML_FILE}")
    except Exception as e:
        print(f"Error saving HTML: {e}")

    return fig

# --- Main Execution ---

def main():
    print("--- Starting Visualization Generation Script ---")
    # Pass both the snapshot pattern and the old file path to the loading function
    df_historical = load_data_sources(SNAPSHOT_FILE_PATTERN, OLD_COMBINED_FILE)
    generate_plot(df_historical)
    print("--- Visualization Generation Script Finished ---")


if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob # To find all snapshot files
from datetime import datetime

# --- Configuration ---
DATA_DIR = 'data' # Directory where snapshots are stored
SNAPSHOT_FILE_PATTERN = os.path.join(DATA_DIR, 'lmsys_snapshot_*.csv')
OLD_COMBINED_FILE = 'combined_raw_lmsys_data.csv' # Path to the old combined file
OUTPUT_HTML_FILE = 'index.html'
# Columns expected in the snapshot CSVs
EXPECTED_SNAPSHOT_COLUMNS = ['Model_Name', 'ELO_Score', 'Provider'] # Add 'License' if saved
# Columns expected/needed from the old combined file
EXPECTED_OLD_COLUMNS = ['Model_Name', 'ELO_Score', 'Provider', 'snapshot_date_extracted']

# --- Helper Function (Standardize Provider - Copied for consistency) ---
# (Mappings omitted for brevity - ensure they match fetch script if re-enabled)
def standardize_provider(provider_name):
    """Standardizes provider names based on common variations."""
    if pd.isna(provider_name) or not isinstance(provider_name, str): return 'Unknown'
    name = provider_name.strip().lower()
    # ... include all mappings from fetch_daily_snapshot.py ...
    if name in ['stanford', 'stanford university']: return 'Stanford'
    return provider_name.strip().title()

# --- Load and Combine Data ---

def load_data_sources(snapshot_pattern, old_file_path):
    """Loads data from daily snapshots and optionally the old combined CSV."""
    all_data = []
    processed_files = 0
    skipped_files = 0

    # 1. Load Daily Snapshots
    print(f"--- Debug: Looking for files matching pattern: {snapshot_pattern}")
    all_snapshot_files = glob.glob(snapshot_pattern)
    print(f"--- Debug: Found snapshot files: {all_snapshot_files}")

    if not all_snapshot_files:
        print("No daily snapshot files found.")
    else:
        print(f"Found {len(all_snapshot_files)} potential snapshot files.")
        for f in sorted(all_snapshot_files):
            print(f"\n--- Debug: Processing snapshot file: {f}")
            try:
                base_name = os.path.basename(f)
                date_str = base_name.replace('lmsys_snapshot_', '').replace('.csv', '')
                print(f"--- Debug: Extracted date string: '{date_str}'")
                snapshot_date = pd.to_datetime(date_str).date()
                print(f"--- Debug: Parsed snapshot date: {snapshot_date}")

                df_snap = pd.read_csv(f)
                print(f"--- Debug: Loaded Snapshot DataFrame shape: {df_snap.shape}")
                print(f"--- Debug: Loaded Snapshot DataFrame columns: {df_snap.columns.tolist()}")

                if df_snap.empty:
                    print(f"--- Debug: Skipping empty snapshot file: {f}")
                    skipped_files += 1
                    continue

                missing_cols = [col for col in EXPECTED_SNAPSHOT_COLUMNS if col not in df_snap.columns]
                if missing_cols:
                     print(f"--- Debug: Skipping snapshot file {f} due to missing columns: {missing_cols}. Expected: {EXPECTED_SNAPSHOT_COLUMNS}")
                     skipped_files += 1
                     continue

                df_snap['Snapshot_Date'] = snapshot_date
                all_data.append(df_snap)
                processed_files += 1
                print(f"--- Debug: Successfully processed snapshot data for {snapshot_date}")

            except Exception as e:
                print(f"--- Debug: Error processing snapshot file {f}: {e}")
                skipped_files += 1

        print(f"\n--- Debug: Finished processing snapshots ---")
        print(f"--- Debug: Loaded data from {processed_files} snapshot files.")
        if skipped_files > 0: print(f"--- Debug: Skipped {skipped_files} snapshot files.")

    # 2. Load Old Combined File (if exists)
    print(f"\n--- Debug: Checking for old combined file: {old_file_path}")
    if os.path.exists(old_file_path):
        print(f"--- Debug: Found old combined file. Attempting to load...")
        try:
            df_old = pd.read_csv(old_file_path)
            print(f"--- Debug: Loaded Old Combined DataFrame shape: {df_old.shape}")
            print(f"--- Debug: Loaded Old Combined DataFrame columns: {df_old.columns.tolist()}")

            if not df_old.empty:
                # --- Rename and Select Columns from Old File ---
                # Check for the date column (might be named differently)
                date_col_found = None
                if 'snapshot_date_extracted' in df_old.columns:
                    date_col_found = 'snapshot_date_extracted'
                elif 'Snapshot_Date' in df_old.columns: # Or maybe it was already named correctly
                    date_col_found = 'Snapshot_Date'
                # Add other potential old date column names if necessary

                if not date_col_found:
                    print(f"--- Debug: Skipping old file {old_file_path} - Could not find a suitable date column (e.g., 'snapshot_date_extracted' or 'Snapshot_Date').")
                else:
                    # Rename the found date column to the standard 'Snapshot_Date'
                    if date_col_found != 'Snapshot_Date':
                        df_old.rename(columns={date_col_found: 'Snapshot_Date'}, inplace=True)
                        print(f"--- Debug: Renamed column '{date_col_found}' to 'Snapshot_Date'.")

                    # Check for other essential columns (Model, ELO, Provider) - adapt names if needed
                    # Example: If old file used 'model' instead of 'Model_Name'
                    rename_map_old = {}
                    if 'model' in df_old.columns and 'Model_Name' not in df_old.columns:
                        rename_map_old['model'] = 'Model_Name'
                    if 'arena_score' in df_old.columns and 'ELO_Score' not in df_old.columns:
                         rename_map_old['arena_score'] = 'ELO_Score'
                    # Add other potential renames based on the old file's structure

                    if rename_map_old:
                         print(f"--- Debug: Applying renames to old data: {rename_map_old}")
                         df_old.rename(columns=rename_map_old, inplace=True)

                    # Verify essential columns are present after potential renames
                    old_essential_cols = ['Model_Name', 'ELO_Score', 'Provider', 'Snapshot_Date']
                    missing_old_cols = [col for col in old_essential_cols if col not in df_old.columns]
                    if missing_old_cols:
                        print(f"--- Debug: Skipping old file {old_file_path} - Missing essential columns after renaming: {missing_old_cols}")
                    else:
                        # Select only the necessary columns
                        cols_to_keep_old = [col for col in old_essential_cols if col in df_old.columns]
                        # Include License if it exists
                        if 'License' in df_old.columns: cols_to_keep_old.append('License')

                        df_old_processed = df_old[cols_to_keep_old].copy()

                        # Parse the date column
                        df_old_processed['Snapshot_Date'] = pd.to_datetime(df_old_processed['Snapshot_Date'], errors='coerce').dt.date
                        df_old_processed.dropna(subset=['Snapshot_Date'], inplace=True) # Drop rows where date parsing failed

                        print(f"--- Debug: Successfully processed old combined file. Shape: {df_old_processed.shape}")
                        all_data.append(df_old_processed)
            else:
                print(f"--- Debug: Old combined file {old_file_path} is empty.")

        except Exception as e:
            print(f"--- Debug: Error processing old combined file {old_file_path}: {e}")
    else:
        print(f"--- Debug: Old combined file not found at {old_file_path}.")


    # 3. Combine and Final Processing
    if not all_data:
        print("--- Debug: No data loaded from any source.")
        return pd.DataFrame()

    print("\n--- Debug: Concatenating all loaded dataframes...")
    # Use outer join to keep all columns from all sources initially
    df_combined = pd.concat(all_data, ignore_index=True, join='outer', sort=False)

    print(f"--- Debug: Combined DataFrame shape before cleaning: {df_combined.shape}")
    print(f"--- Debug: Combined DataFrame columns: {df_combined.columns.tolist()}")
    print(f"--- Debug: Combined DataFrame head before cleaning:\n{df_combined.head()}")

    # Ensure correct types after combining
    print("--- Debug: Converting data types...")
    df_combined['Snapshot_Date'] = pd.to_datetime(df_combined['Snapshot_Date'], errors='coerce')
    df_combined['ELO_Score'] = pd.to_numeric(df_combined['ELO_Score'], errors='coerce')
    # Fill NaNs in string columns that might result from outer join or loading issues
    df_combined['Model_Name'] = df_combined['Model_Name'].astype(str).fillna('Unknown')
    df_combined['Provider'] = df_combined['Provider'].astype(str).fillna('Unknown')
    if 'License' in df_combined.columns:
         df_combined['License'] = df_combined['License'].astype(str).fillna('Unknown')

    # Optional: Re-apply standardization defensively
    print("--- Debug: Re-applying provider standardization...")
    df_combined['Provider'] = df_combined['Provider'].apply(standardize_provider)

    # Drop rows with invalid essential data AFTER type conversion
    initial_rows_before_dropna = len(df_combined)
    df_combined.dropna(subset=['ELO_Score', 'Snapshot_Date', 'Model_Name', 'Provider'], inplace=True)
    # Also drop rows where key identifiers became empty strings or 'Unknown' inappropriately
    df_combined = df_combined[df_combined['Model_Name'].str.strip().str.lower() != 'unknown']
    df_combined = df_combined[df_combined['Model_Name'].str.strip() != '']
    df_combined = df_combined[df_combined['Provider'].str.strip().str.lower() != 'unknown']
    df_combined = df_combined[df_combined['Provider'].str.strip() != '']

    rows_dropped = initial_rows_before_dropna - len(df_combined)
    if rows_dropped > 0:
        print(f"--- Debug: Dropped {rows_dropped} rows due to NaN/Invalid ELO, Date, Model, or Provider.")

    # 4. Deduplicate across all sources
    print(f"--- Debug: Deduplicating combined data ({len(df_combined)} rows)...")
    # Sort by date ascending, then ELO descending to keep highest score for a given day
    df_combined.sort_values(by=['Snapshot_Date', 'Model_Name', 'Provider', 'ELO_Score'],
                            ascending=[True, True, True, False],
                            inplace=True)
    # Keep the FIRST entry after sorting, which has the highest ELO for that day/model/provider
    df_deduplicated = df_combined.drop_duplicates(subset=['Snapshot_Date', 'Model_Name', 'Provider'], keep='first')
    deduplicated_rows_count = len(df_combined) - len(df_deduplicated)
    if deduplicated_rows_count > 0:
        print(f"--- Debug: Removed {deduplicated_rows_count} duplicate entries (same day, model, provider), keeping highest ELO.")


    # Final sort for visualization
    df_final = df_deduplicated.sort_values(by=['Snapshot_Date', 'ELO_Score'], ascending=[True, False])


    print(f"--- Debug: Final combined DataFrame shape: {df_final.shape}")
    if not df_final.empty:
        print(f"--- Debug: Final combined DataFrame head:\n{df_final.head()}")
        print(f"--- Debug: Final combined DataFrame date range: {df_final['Snapshot_Date'].min()} to {df_final['Snapshot_Date'].max()}")
    else:
        print("--- Debug: Final combined DataFrame is empty.")

    return df_final


# --- Generate Plotly Visualization (Code remains the same as before) ---
# ... (generate_plot function remains exactly the same as the previous version) ...
def generate_plot(df_processed):
    """Generates the Plotly bar chart race visualization."""
    if df_processed is None or df_processed.empty:
        print("No processed data available to generate visualization.")
        # Create an empty placeholder HTML or return None
        try:
            with open(OUTPUT_HTML_FILE, 'w') as f:
                f.write("<html><body>No data available to generate visualization. Check logs for errors loading snapshot data.</body></html>")
            print(f"Created empty placeholder file: {OUTPUT_HTML_FILE}")
        except Exception as e:
            print(f"Error writing placeholder HTML file: {e}")
        return None

    print("\n--- Generating Plotly Visualization ---")

    # Config with REVISED Corporate Identity Colors (Same as before)
    COLOR_MAPPING = {
        'OpenAI': '#10A37F', 'Google': '#4285F4', 'Meta': '#8A2BE2', 'Anthropic': '#D97706',
        'Mistral': '#FF4B4B', 'Alibaba': '#FF8800', 'xAI': '#333333', 'DeepSeek': '#00BFFF',
        'Tencent': '#0052D9', 'Amazon': '#FF9900', 'Cohere': '#8E44AD', '01 AI': '#2ECC71',
        'Zhipu AI': '#2962FF', 'Databricks': '#FF3621', 'Microsoft': '#00A4EF', 'Reka AI': '#6B5B95',
        'Nvidia': '#76B900', 'Together AI': '#FF6B6B', 'Upstage': '#4ECDC4', 'Snowflake': '#29B5E8',
        'Baichuan': '#A0522D', 'Adept': '#FF69B4', 'Teknium': '#4682B4', 'KAIST': '#00427E',
        'EleutherAI': '#000000', 'BigCode': '#FFD700', 'Salesforce': '#00A1E0',
        'Allen Institute for AI': '#00B398', 'Hugging Face': '#FFD700', 'LMSys': '#FBBD01',
        'UC Berkeley': '#003262', 'Stanford': '#8C1515',
        'Unknown': '#B0B0B0', 'Other': '#CCCCCC'
    }

    # Prepare dates
    df_processed['Snapshot_Date'] = pd.to_datetime(df_processed['Snapshot_Date'])
    unique_dates = sorted(df_processed['Snapshot_Date'].unique())
    if not unique_dates:
        print("Error: No unique dates found in processed data. Cannot generate plot.")
        with open(OUTPUT_HTML_FILE, 'w') as f:
            f.write("<html><body>No unique dates found in data. Cannot generate visualization.</body></html>")
        print(f"Created empty placeholder file: {OUTPUT_HTML_FILE}")
        return None

    date_strings = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in unique_dates]
    max_elo_overall = df_processed['ELO_Score'].max()
    min_elo_for_annotation_calc = df_processed['ELO_Score'].min() * 0.9 if pd.notna(df_processed['ELO_Score'].min()) else 0
    min_elo_for_annotation = max(min_elo_for_annotation_calc, 10)

    fig = go.Figure()
    frames = []
    slider_steps = []
    initial_top_df = pd.DataFrame()

    hovertemplate = (
        "<b>Provider:</b> %{customdata[0]}<br>" +
        "<b>Model:</b> %{customdata[1]}<br>" +
        "<b>ELO Score:</b> %{customdata[2]:.1f}<extra></extra>"
    )

    print(f"Creating {len(unique_dates)} frames...")
    for idx, date in enumerate(unique_dates):
        date_str = date_strings[idx]
        snapshot_df = df_processed[df_processed['Snapshot_Date'] == date].copy()
        snapshot_sorted_all = snapshot_df.sort_values(by='ELO_Score', ascending=False)
        top_df = snapshot_sorted_all.head(15).copy() # Top 15 models per frame

        if top_df.empty:
            print(f"Warning: No data for date {date_str}, skipping frame.")
            continue # Skip frame if no data for this date

        # Create unique Y-category including score for sorting/identification
        top_df['Y_Category'] = top_df['Provider'] + ' | ' + top_df['Model_Name'] + ' | ' + top_df['ELO_Score'].round(1).astype(str)

        # Store the data for the last valid frame found
        initial_top_df = top_df # Keep updating until the last loop iteration

        # --- Create Bar Trace for the Frame ---
        frame_plotly_data = go.Bar(
            x=top_df['ELO_Score'], y=top_df['Y_Category'], orientation='h',
            text=top_df['ELO_Score'].round(1).astype(str), textposition='outside',
            hoverinfo='skip', customdata=top_df[['Provider', 'Model_Name', 'ELO_Score']],
            hovertemplate=hovertemplate,
            marker_color=[COLOR_MAPPING.get(p, COLOR_MAPPING['Other']) if p != 'Unknown' else COLOR_MAPPING['Unknown'] for p in top_df['Provider']],
            name=date_str
        )

        # --- Annotations for Frame ---
        frame_annotations = []
        frame_annotations.append(dict(x=0.5, y=1.10, xref='paper', yref='paper', text=f"<b>{date_str}</b>", showarrow=False, font=dict(size=18, color="#333333"), align='center'))
        for i in range(len(top_df)):
            row = top_df.iloc[i]
            frame_annotations.append(dict(
                x=min_elo_for_annotation, y=row['Y_Category'], xref='x', yref='y',
                text=f"<i>{row['Model_Name']}</i>", showarrow=False,
                font=dict(size=8, color='white'), align='left', xanchor='left', xshift=5
            ))

        # --- Y-axis Layout for Frame ---
        frame_yaxis_layout = dict(
            tickmode='array', tickvals=top_df['Y_Category'].tolist(), ticktext=top_df['Provider'].tolist(),
            autorange="reversed", tickfont=dict(size=10), automargin=True, ticklabelstandoff=10
        )

        frame_layout = go.Layout(annotations=frame_annotations, yaxis=frame_yaxis_layout)
        frames.append(go.Frame(data=[frame_plotly_data], name=date_str, layout=frame_layout))

        # --- Slider Step ---
        slider_step = {"args": [[date_str], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}], "label": date_str, "method": "animate"}
        slider_steps.append(slider_step)

    # --- Add Initial Data Trace (based on the data from the *last successful frame*) ---
    initial_annotations = []
    initial_yaxis_layout = {}
    last_valid_date_str = date_strings[unique_dates.index(initial_top_df['Snapshot_Date'].iloc[0])] if not initial_top_df.empty else "Latest"

    if not initial_top_df.empty:
        initial_annotations.append(dict(x=0.5, y=1.10, xref='paper', yref='paper', text=f"<b>{last_valid_date_str}</b>", showarrow=False, font=dict(size=18, color="#333333"), align='center'))
        for i in range(len(initial_top_df)):
            row = initial_top_df.iloc[i]
            initial_annotations.append(dict(x=min_elo_for_annotation, y=row['Y_Category'], xref='x', yref='y', text=f"<i>{row['Model_Name']}</i>", showarrow=False, font=dict(size=8, color='white'), align='left', xanchor='left', xshift=5))

        initial_yaxis_layout = dict(
            tickmode='array', tickvals=initial_top_df['Y_Category'].tolist(), ticktext=initial_top_df['Provider'].tolist(),
            autorange="reversed", tickfont=dict(size=10), automargin=True, ticklabelstandoff=10
        )

        fig.add_trace(go.Bar(
            x=initial_top_df['ELO_Score'], y=initial_top_df['Y_Category'], orientation='h',
            text=initial_top_df['ELO_Score'].round(1).astype(str), textposition='outside',
            hoverinfo='skip', customdata=initial_top_df[['Provider', 'Model_Name', 'ELO_Score']],
            hovertemplate=hovertemplate,
            marker_color=[COLOR_MAPPING.get(p, COLOR_MAPPING['Other']) if p != 'Unknown' else COLOR_MAPPING['Unknown'] for p in initial_top_df['Provider']],
            name=last_valid_date_str
        ))
    else:
        print("Warning: Could not determine initial frame data (initial_top_df is empty). Plot might be empty initially.")
        if not frames:
             fig.add_annotation(text="No data available for visualization.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))


    # --- Configure Overall Layout ---
    fig.frames = frames
    fig.update_layout(
        title=dict(
            text=f"LMSYS Chatbot Arena - Top {len(initial_top_df) if not initial_top_df.empty else 'N'} Models by ELO Score<br><sup>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</sup>",
            font=dict(size=20), x=0.5, xanchor='center'
        ),
        xaxis_title="ELO Score", yaxis_title="Provider",
        xaxis=dict(range=[min_elo_for_annotation * 0.98, max_elo_overall * 1.05 if pd.notna(max_elo_overall) else 1500]),
        yaxis=initial_yaxis_layout, # Apply layout from the last valid frame
        height=800, margin=dict(l=150, r=50, t=150, b=100),
        hovermode='closest', hoverlabel=dict(bgcolor="white", font_size=12),
        annotations=initial_annotations,
        sliders=[{"active": len(slider_steps) - 1 if slider_steps else 0, "currentvalue": {"prefix": "Date: ", "visible": True, "xanchor": "right"}, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.05, "y": 0, "steps": slider_steps}],
        updatemenus=[{"buttons": [{"args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300, "easing": "linear"}}], "label": "► Play", "method": "animate"}, {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}], "label": "❚❚ Pause", "method": "animate"}], "direction": "left", "pad": {"r": 10, "t": 80}, "showactive": False, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0.08, "yanchor": "top"}]
    )

    # --- Save HTML ---
    try:
        fig.write_html(OUTPUT_HTML_FILE)
        print(f"Interactive HTML saved: {OUTPUT_HTML_FILE}")
    except Exception as e:
        print(f"Error saving HTML: {e}")

    return fig

# --- Main Execution ---

def main():
    print("--- Starting Visualization Generation Script ---")
    # Pass both the snapshot pattern and the old file path to the loading function
    df_historical = load_data_sources(SNAPSHOT_FILE_PATTERN, OLD_COMBINED_FILE)
    generate_plot(df_historical)
    print("--- Visualization Generation Script Finished ---")


if __name__ == "__main__":
    main()
