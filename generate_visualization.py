import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob # To find all snapshot files
from datetime import datetime

# --- Configuration ---
DATA_DIR = 'data' 
SNAPSHOT_FILE_PATTERN = os.path.join(DATA_DIR, 'lmsys_snapshot_*.csv')
OLD_COMBINED_FILE = os.path.join(DATA_DIR, 'combined_raw_lmsys_data.csv')
OUTPUT_HTML_FILE = 'index.html'
EXPECTED_SNAPSHOT_COLUMNS = ['Model_Name', 'ELO_Score', 'Provider']
EXPECTED_OLD_DATE_COL = 'snapshot_date_extracted'
EXPECTED_OLD_MODEL_COL = 'model'
EXPECTED_OLD_SCORE_COL = 'arena_score'
EXPECTED_OLD_PROVIDER_COL = 'organization'


# --- Load and Combine Data ---

def load_data_sources(snapshot_pattern, old_file_path):
    """Loads data from daily snapshots and optionally the old combined CSV."""
    all_dataframes = [] # List to hold dataframes before concatenation
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
                df_snap['Provider'] = df_snap['Provider'].astype(str).fillna('Unknown').str.strip()
                all_dataframes.append(df_snap) # Add successfully loaded snapshot df
                processed_files += 1
                print(f"--- Debug: Successfully processed snapshot data for {snapshot_date}")

            except Exception as e:
                print(f"--- Debug: Error processing snapshot file {f}: {e}")
                skipped_files += 1

        print(f"\n--- Debug: Finished processing snapshots ---")
        print(f"--- Debug: Loaded data from {processed_files} snapshot files.")
        if skipped_files > 0: print(f"--- Debug: Skipped {skipped_files} snapshot files.")

    # 2. Load Old Combined File (if exists)
    print(f"\n--- Debug: Checking for old combined file at corrected path: {old_file_path}") # Updated debug message
    old_file_loaded_successfully = False
    if os.path.exists(old_file_path):
        print(f"--- Debug: Found old combined file. Attempting to load...")
        try:
            df_old_raw = pd.read_csv(old_file_path)
            print(f"--- Debug: Loaded Old Combined RAW DataFrame shape: {df_old_raw.shape}")
            print(f"--- Debug: Loaded Old Combined RAW DataFrame columns: {df_old_raw.columns.tolist()}")

            if not df_old_raw.empty:
                df_old = df_old_raw.copy() # Work on a copy

                # --- Define expected old columns and map them ---
                # ** ADJUST THESE based on your actual combined_raw_lmsys_data.csv **
                old_col_map = {
                    EXPECTED_OLD_DATE_COL: 'Snapshot_Date',
                    EXPECTED_OLD_MODEL_COL: 'Model_Name',
                    EXPECTED_OLD_SCORE_COL: 'ELO_Score',
                    EXPECTED_OLD_PROVIDER_COL: 'Provider',
                    # Add 'license': 'License' if applicable and present
                }

                # Check if expected old columns exist before trying to rename
                actual_old_cols = df_old.columns.tolist()
                rename_map_needed = {}
                missing_essential_old = []

                for old_name, new_name in old_col_map.items():
                    if old_name in actual_old_cols:
                        if old_name != new_name: # Only add to map if rename is needed
                            rename_map_needed[old_name] = new_name
                    elif new_name not in actual_old_cols: # If neither old nor new name exists
                         # Check if it's an essential column (Date, Model, Score, Provider)
                         if new_name in ['Snapshot_Date', 'Model_Name', 'ELO_Score', 'Provider']:
                              missing_essential_old.append(f"{new_name} (expected from {old_name})")

                if missing_essential_old:
                    print(f"--- Debug: Skipping old file {old_file_path} - Missing essential source columns: {missing_essential_old}")
                else:
                    # Apply renames if needed
                    if rename_map_needed:
                        print(f"--- Debug: Applying renames to old data: {rename_map_needed}")
                        df_old.rename(columns=rename_map_needed, inplace=True)
                        print(f"--- Debug: Columns after rename: {df_old.columns.tolist()}")

                    # Select only the columns we need (now using standard names)
                    cols_to_keep_old = ['Snapshot_Date', 'Model_Name', 'ELO_Score', 'Provider']
                    if 'License' in df_old.columns: cols_to_keep_old.append('License') # Keep if present

                    # Filter df_old to only keep necessary columns that actually exist
                    final_old_cols = [col for col in cols_to_keep_old if col in df_old.columns]
                    df_old_processed = df_old[final_old_cols].copy()

                    # --- Process Old Data ---
                    print(f"--- Debug: Processing old data (Shape: {df_old_processed.shape})...")
                    # Parse date
                    df_old_processed['Snapshot_Date'] = pd.to_datetime(df_old_processed['Snapshot_Date'], errors='coerce').dt.date
                    # Clean types
                    df_old_processed['ELO_Score'] = pd.to_numeric(df_old_processed['ELO_Score'], errors='coerce')
                    df_old_processed['Model_Name'] = df_old_processed['Model_Name'].astype(str).fillna('Unknown').str.strip()

                    # Drop rows with NaNs in essential columns after processing
                    df_old_processed.dropna(subset=['Snapshot_Date', 'Model_Name', 'ELO_Score', 'Provider'], inplace=True)
                    print(f"--- Debug: Old data shape after processing & dropna: {df_old_processed.shape}")

                    if not df_old_processed.empty:
                         all_dataframes.append(df_old_processed) # Add successfully processed old df
                         old_file_loaded_successfully = True
                         print(f"--- Debug: Successfully processed and added old combined file data.")
                    else:
                         print(f"--- Debug: Old data became empty after processing and dropna.")

            else:
                print(f"--- Debug: Old combined file {old_file_path} is empty.")

        except Exception as e:
            print(f"--- Debug: Error processing old combined file {old_file_path}: {e}")
    else:
        print(f"--- Debug: Old combined file not found at {old_file_path}.")


    # 3. Combine and Final Processing
    if not all_dataframes: # Check if the list of dataframes is empty
        print("--- Debug: No dataframes to concatenate from any source.")
        return pd.DataFrame()

    print(f"\n--- Debug: Concatenating {len(all_dataframes)} dataframes...")
    # Use outer join to keep all columns from all sources initially
    df_combined = pd.concat(all_dataframes, ignore_index=True, join='outer', sort=False)

    print(f"--- Debug: Combined DataFrame shape before final cleaning: {df_combined.shape}")
    if not df_combined.empty:
        print(f"--- Debug: Combined DataFrame columns: {df_combined.columns.tolist()}")
        print(f"--- Debug: Combined DataFrame head before final cleaning:\n{df_combined.head()}")
    else:
        print("--- Debug: Combined DataFrame is unexpectedly empty after concat.")
        return pd.DataFrame() # Return empty if concat resulted in empty


    # Ensure correct types again after concat (can introduce NaNs/object types)
    print("--- Debug: Final type conversion...")
    df_combined['Snapshot_Date'] = pd.to_datetime(df_combined['Snapshot_Date'], errors='coerce')
    df_combined['ELO_Score'] = pd.to_numeric(df_combined['ELO_Score'], errors='coerce')
    df_combined['Model_Name'] = df_combined['Model_Name'].astype(str).fillna('Unknown').str.strip()
    df_combined['Provider'] = df_combined['Provider'].astype(str).fillna('Unknown').str.strip()
    if 'License' in df_combined.columns:
         df_combined['License'] = df_combined['License'].astype(str).fillna('Unknown')


    # Final drop of rows with invalid essential data
    initial_rows_before_final_dropna = len(df_combined)
    df_combined.dropna(subset=['ELO_Score', 'Snapshot_Date', 'Model_Name', 'Provider'], inplace=True)
    # Also drop rows where key identifiers became empty strings or 'Unknown' inappropriately
    df_combined = df_combined[df_combined['Model_Name'].str.lower() != 'unknown']
    df_combined = df_combined[df_combined['Model_Name'] != '']
    df_combined = df_combined[df_combined['Provider'].str.lower() != 'unknown']
    df_combined = df_combined[df_combined['Provider'] != '']
    rows_dropped_final = initial_rows_before_final_dropna - len(df_combined)
    if rows_dropped_final > 0:
        print(f"--- Debug: Dropped {rows_dropped_final} rows during final cleaning due to NaN/Invalid ELO, Date, Model, or Provider.")

    # 4. Deduplicate across all sources
    print(f"--- Debug: Deduplicating combined data ({len(df_combined)} rows)...")
    df_combined.sort_values(by=['Snapshot_Date', 'Model_Name', 'Provider', 'ELO_Score'],
                            ascending=[True, True, True, False],
                            inplace=True)
    df_deduplicated = df_combined.drop_duplicates(subset=['Snapshot_Date', 'Model_Name', 'Provider'], keep='first')
    deduplicated_rows_count = len(df_combined) - len(df_deduplicated)
    if deduplicated_rows_count > 0:
        print(f"--- Debug: Removed {deduplicated_rows_count} duplicate entries (same day, model, provider), keeping highest ELO.")


    # Final sort for visualization
    df_final = df_deduplicated.sort_values(by=['Snapshot_Date', 'ELO_Score'], ascending=[True, False])


    print(f"--- Debug: Final combined DataFrame shape for plotting: {df_final.shape}")
    if not df_final.empty:
        print(f"--- Debug: Final combined DataFrame head:\n{df_final.head()}")
        print(f"--- Debug: Final combined DataFrame date range: {df_final['Snapshot_Date'].min()} to {df_final['Snapshot_Date'].max()}")
    else:
        print("--- Debug: Final combined DataFrame for plotting is empty.")

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

    # ENHANCED COLOR MAPPING: Official brand colors + research-based colors + high-contrast additions
    COLOR_MAPPING = {
        # Major AI Providers - Official Brand Colors
        'OpenAI': '#74AA9C',        # ChatGPT's official green
        'Google': '#4285F4',        # Google's official blue
        'Meta': '#0081FB',          # Meta's official blue (updated from purple)
        'Anthropic': '#DE7356',     # Claude's official peach/coral
        'Microsoft': '#00A4EF',     # Microsoft's official blue
        'Amazon': '#FF9900',        # Amazon's official orange
        
        # Emerging Major Players - Research-based Colors
        'xAI': '#000000',           # Black (X/xAI branding is predominantly black)
        'DeepSeek': '#1E90FF',      # DeepSeek blue (professional blue from research)
        'Mistral': '#FF7A00',       # Mistral orange (updated for better contrast)
        'Alibaba': '#FF6A00',       # Alibaba orange
        'Tencent': '#00599C',       # Tencent blue
        'Nvidia': '#76B900',        # Nvidia green
        'Cohere': '#9333EA',        # Purple for distinctiveness
        
        # Asian AI Companies - Distinctive Colors
        'Zhipu AI': '#2962FF',      # Material Design blue
        'Z.ai': '#7C3AED',          # Purple variant for distinction
        'Moonshot': '#059669',      # Emerald green
        'Baichuan': '#DC6803',      # Amber/orange
        'SenseNova': '#BE185D',     # Pink
        'Stepfun': '#7C2D12',       # Brown
        'MiniMax': '#1565C0',       # Deep blue
        '01 AI': '#16A34A',         # Green
        'Zhipu': '#8B5CF6',        # Light purple (variant of Zhipu AI)
        
        # Research & Academic Institutions
        'UC Berkeley': '#003262',    # Cal blue
        'Stanford': '#8C1515',      # Stanford cardinal
        'Allen AI': '#0891B2',      # Cyan
        'AllenAI/UW': '#4338CA',    # Indigo
        'Princeton': '#FF8C00',     # Princeton orange
        'UW': '#4B0082',           # University purple
        'Tsinghua': '#722F37',     # Maroon
        'KAIST': '#DC2626',        # Red
        
        # Technology Companies
        'Databricks': '#FF3621',    # Databricks red
        'Snowflake': '#29B5E8',     # Snowflake blue
        'Reka AI': '#6366F1',       # Indigo
        'Upstage': '#06B6D4',       # Cyan
        'Together AI': '#EF4444',   # Red
        'Stability AI': '#8B5CF6',  # Purple
        'Nomic AI': '#F59E0B',      # Amber
        'InternLM': '#10B981',      # Emerald
        'TII': '#0EA5E9',          # Sky blue
        
        # Open Source & Community
        'HuggingFace': '#FFD21E',   # Hugging Face yellow
        'LMSYS': '#FBBF24',        # Yellow
        'EleutherAI': '#374151',    # Dark gray
        'BigCode': '#F59E0B',      # Amber
        'OpenChat': '#3B82F6',     # Blue
        'NousResearch': '#8B5CF6', # Purple
        'Cognitive Computations': '#14B8A6', # Teal
        'RWKV': '#991B1B',         # Dark red
        'OpenAssistant': '#059669', # Emerald
        
        # Specialized Companies
        'AI21 Labs': '#7C3AED',    # Purple
        'Ai2': '#0891B2',          # Cyan (Allen Institute variant)
        'Adept': '#EC4899',        # Pink
        'Inflection': '#6366F1',   # Indigo
        'Character.AI': '#A855F7', # Purple
        'Perplexity': '#059669',   # Emerald
        'Anthropic': '#DE7356',    # Repeated for safety
        
        # International/Emerging
        'DeepSeek AI': '#0284C7',  # DeepSeek variant
        'NexusFlow': '#7C3AED',    # Purple
        'Nexusflow': '#8B5CF6',    # Purple variant
        'MosaicML': '#0891B2',     # Cyan
        'Salesforce': '#00A1E0',   # Salesforce blue
        'IBM': '#1F70C1',          # IBM blue
        
        # Additional High-Contrast Colors for New Providers
        'Provider': '#6B7280',      # Gray (header edge case)
        'Unknown': '#9CA3AF',       # Light gray
        'Other': '#D1D5DB',        # Very light gray
        
        # Reserve colors for future providers
        'Reserved1': '#84CC16',     # Lime
        'Reserved2': '#F97316',     # Orange
        'Reserved3': '#E11D48',     # Rose
        'Reserved4': '#7C2D12',     # Brown
        'Reserved5': '#1F2937',     # Dark gray
        'Reserved6': '#065F46',     # Dark green
        'Reserved7': '#92400E',     # Dark amber
        'Reserved8': '#7E22CE',     # Dark purple
    }

    # Add colors for any providers found in the data but not in the mapping
    # Enhanced fallback with diverse color palette
    FALLBACK_COLORS = [
        '#F97316',  # Orange
        '#84CC16',  # Lime
        '#06B6D4',  # Cyan
        '#8B5CF6',  # Purple
        '#F59E0B',  # Amber
        '#EF4444',  # Red
        '#10B981',  # Emerald
        '#3B82F6',  # Blue
        '#EC4899',  # Pink
        '#6366F1',  # Indigo
        '#14B8A6',  # Teal
        '#F97316',  # Orange (repeat for more providers)
        '#A855F7',  # Purple variant
        '#059669',  # Emerald variant
        '#DC2626',  # Red variant
        '#7C3AED',  # Purple variant
        '#0891B2',  # Cyan variant
        '#BE185D',  # Pink variant
        '#16A34A',  # Green variant
        '#1565C0',  # Blue variant
    ]
    
    all_providers = df_processed['Provider'].unique()
    unmapped_providers = []
    for provider in all_providers:
        if provider not in COLOR_MAPPING and provider != 'Unknown':
            unmapped_providers.append(provider)
    
    # Assign fallback colors in a round-robin fashion for better distribution
    for i, provider in enumerate(unmapped_providers):
        fallback_color = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
        COLOR_MAPPING[provider] = fallback_color
        print(f"--- Plot Debug: Adding fallback color {fallback_color} for unmapped provider: {provider}")


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
            marker_color=[COLOR_MAPPING.get(p, COLOR_MAPPING['Other']) for p in top_df['Provider']], # Use updated mapping
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
            marker_color=[COLOR_MAPPING.get(p, COLOR_MAPPING['Other']) for p in initial_top_df['Provider']], # Use updated mapping
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
