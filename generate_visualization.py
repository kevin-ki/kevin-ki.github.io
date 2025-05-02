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
OUTPUT_HTML_FILE = 'index.html'
# Columns expected in the snapshot CSVs (must match output of fetch script)
EXPECTED_COLUMNS = ['Model_Name', 'ELO_Score', 'Provider'] # Add 'License' if it's saved

# --- Helper Function (Standardize Provider - Copied for consistency if needed, though data should be clean) ---
# Note: Providers should already be standardized by the fetch script.
# Including it here defensively or if you need to re-apply logic.
def standardize_provider(provider_name):
    """Standardizes provider names based on common variations."""
    if pd.isna(provider_name) or not isinstance(provider_name, str): return 'Unknown'
    name = provider_name.strip().lower()
    # (Mappings omitted for brevity - use the same list as in the fetch script if needed)
    if name in ['openai', 'openai baseline', 'openai-internal']: return 'OpenAI'
    if name in ['google', 'google research', 'google-research', 'google-internal', 'google deepmind']: return 'Google'
    if name in ['meta', 'metaai', 'facebook', 'meta-llama', 'meta platforms inc.', 'meta platforms inc', 'llama team (meta)', 'meta/llama']: return 'Meta'
    # ... include all other mappings from the fetch script ...
    if name in ['stanford', 'stanford university']: return 'Stanford'
    # Fallback
    return provider_name.strip().title()

# --- Load and Combine Data ---

def load_all_snapshots(pattern):
    """Loads all snapshot CSVs from the data directory."""
    all_files = glob.glob(pattern)
    if not all_files:
        print(f"No snapshot files found matching pattern: {pattern}")
        return pd.DataFrame()

    print(f"Found {len(all_files)} snapshot files.")
    all_data = []
    processed_files = 0
    skipped_files = 0

    for f in sorted(all_files): # Sort files chronologically by name
        try:
            # Extract date from filename
            date_str = os.path.basename(f).replace('lmsys_snapshot_', '').replace('.csv', '')
            snapshot_date = pd.to_datetime(date_str).date() # Convert to date object

            df_snap = pd.read_csv(f)

            # Basic validation
            if df_snap.empty:
                print(f"Warning: Skipping empty snapshot file: {f}")
                skipped_files += 1
                continue
            if not all(col in df_snap.columns for col in EXPECTED_COLUMNS):
                 print(f"Warning: Skipping snapshot file with missing columns: {f}. Expected: {EXPECTED_COLUMNS}, Got: {df_snap.columns.tolist()}")
                 skipped_files += 1
                 continue


            df_snap['Snapshot_Date'] = snapshot_date
            all_data.append(df_snap)
            processed_files += 1
        except Exception as e:
            print(f"Error processing file {f}: {e}")
            skipped_files += 1

    print(f"Successfully loaded data from {processed_files} files.")
    if skipped_files > 0:
        print(f"Skipped {skipped_files} files due to errors or missing columns.")

    if not all_data:
        return pd.DataFrame()

    # Combine all dataframes
    df_combined = pd.concat(all_data, ignore_index=True)

    # Ensure correct types after combining
    df_combined['Snapshot_Date'] = pd.to_datetime(df_combined['Snapshot_Date'])
    df_combined['ELO_Score'] = pd.to_numeric(df_combined['ELO_Score'], errors='coerce')
    df_combined['Model_Name'] = df_combined['Model_Name'].astype(str)
    df_combined['Provider'] = df_combined['Provider'].astype(str)
    # Optional: Re-apply standardization if needed
    # df_combined['Provider'] = df_combined['Provider'].apply(standardize_provider)

    # Drop rows with invalid ELO or date after combining/conversion
    df_combined.dropna(subset=['ELO_Score', 'Snapshot_Date'], inplace=True)

    print(f"Combined historical data has {len(df_combined)} rows.")
    print("Combined DataFrame head:\n", df_combined.head())
    print("Combined DataFrame date range:", df_combined['Snapshot_Date'].min(), "to", df_combined['Snapshot_Date'].max())

    return df_combined

# --- Generate Plotly Visualization ---

def generate_plot(df_processed):
    """Generates the Plotly bar chart race visualization."""
    if df_processed is None or df_processed.empty:
        print("No processed data available to generate visualization.")
        # Create an empty placeholder HTML or return None
        with open(OUTPUT_HTML_FILE, 'w') as f:
            f.write("<html><body>No data available to generate visualization.</body></html>")
        print(f"Created empty placeholder file: {OUTPUT_HTML_FILE}")
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

        if idx == len(unique_dates) - 1:
            initial_top_df = top_df

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

    # --- Add Initial Data Trace (based on the last frame's data) ---
    initial_annotations = []
    initial_yaxis_layout = {}
    if not initial_top_df.empty:
        initial_annotations.append(dict(x=0.5, y=1.10, xref='paper', yref='paper', text=f"<b>{date_strings[-1] if date_strings else 'Latest'}</b>", showarrow=False, font=dict(size=18, color="#333333"), align='center'))
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
            name=date_strings[-1] if date_strings else "Initial Data"
        ))
    else:
        print("Warning: Could not determine initial frame data. Plot might be empty initially.")
        # Add a dummy trace or annotation if the figure would otherwise be empty
        if not frames: # If there are no frames at all
             fig.add_annotation(text="No data available for visualization.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))


    # --- Configure Overall Layout ---
    fig.frames = frames
    fig.update_layout(
        title=dict(
            text=f"LMSYS Chatbot Arena - Top {len(initial_top_df) if not initial_top_df.empty else 'N'} Models by ELO Score<br><sup>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</sup>",
            font=dict(size=20), x=0.5, xanchor='center'
        ),
        xaxis_title="ELO Score", yaxis_title="Provider",
        xaxis=dict(range=[min_elo_for_annotation * 0.98, max_elo_overall * 1.05 if pd.notna(max_elo_overall) else 1500]), # Handle case where max_elo might be NaN
        yaxis=initial_yaxis_layout,
        height=800, margin=dict(l=150, r=50, t=150, b=100),
        hovermode='closest', hoverlabel=dict(bgcolor="white", font_size=12),
        annotations=initial_annotations,
        sliders=[{"active": len(slider_steps) - 1, "currentvalue": {"prefix": "Date: ", "visible": True, "xanchor": "right"}, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.05, "y": 0, "steps": slider_steps}],
        updatemenus=[{"buttons": [{"args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300, "easing": "linear"}}], "label": "► Play", "method": "animate"}, {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}], "label": "❚❚ Pause", "method": "animate"}], "direction": "left", "pad": {"r": 10, "t": 80}, "showactive": False, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0.08, "yanchor": "top"}]
    )

    # --- Save HTML ---
    try:
        fig.write_html(OUTPUT_HTML_FILE)
        print(f"Interactive HTML saved: {OUTPUT_HTML_FILE}")
    except Exception as e:
        print(f"Error saving HTML: {e}")

    return fig # Return figure object if needed elsewhere


# --- Main Execution ---

def main():
    print("--- Starting Visualization Generation Script ---")
    df_historical = load_all_snapshots(SNAPSHOT_FILE_PATTERN)
    generate_plot(df_historical)
    print("--- Visualization Generation Script Finished ---")


if __name__ == "__main__":
    main()
