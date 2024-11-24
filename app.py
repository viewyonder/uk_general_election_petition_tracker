import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import time
import csv
from io import StringIO
from pathlib import Path
import base64
from github import Github

# Page config
st.set_page_config(
    page_title="Petition Tracker",
    page_icon="📊",
    layout="wide"
)

# Initialize GitHub connection
store = Github(st.secrets["GITHUB_TOKEN"]).get_repo(st.secrets["GITHUB_REPO"])

# Initialize session state
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = datetime.min
if 'last_count' not in st.session_state:
    st.session_state.last_count = 0

def get_petition_count(url):
    """Fetch the petition count from the given URL."""
    time_since_last_fetch = datetime.now() - st.session_state.last_fetch_time
    if time_since_last_fetch.total_seconds() < 55:
        return None

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        count_element = soup.select_one('span.count')
        
        st.session_state.last_fetch_time = datetime.now()
        
        if count_element:
            return int(count_element.text.replace(',', ''))
        return None
    except Exception as e:
        st.error(f"Error fetching count: {e}")
        return None

#def load_data():
#    """Load data from CSV file."""
#    if os.path.exists('petition_counts.csv'):
#        df = pd.read_csv('petition_counts.csv')
#        df['timestamp'] = pd.to_datetime(df['timestamp'])
#        return df
#    return pd.DataFrame(columns=['timestamp', 'count'])

def load_data():
    """Load data from GitHub CSV file."""
    try:
        # Get file content from GitHub
        #file = store.repo.get_contents(store.file_path)
        file = store.get_contents('petition_counts.csv')
        content = base64.b64decode(file.content).decode()
        
        # Convert to DataFrame
        df = pd.read_csv(pd.io.common.StringIO(content))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        # If file doesn't exist or there's an error, return empty DataFrame
        return pd.DataFrame(columns=['timestamp', 'count'])
    
#def log_count(timestamp, count):
#    """Log count to CSV file."""
#    file_exists = os.path.exists('petition_counts.csv')
#    
#    with open('petition_counts.csv', 'a', newline='') as f:
#        writer = csv.writer(f)
#        if not file_exists:
#            writer.writerow(['timestamp', 'count'])
#        writer.writerow([timestamp, count])

def log_count(timestamp, count):
    """Log count to GitHub CSV file."""
    try:
        # Try to get existing file
        try:
            file = store.get_contents('petition_counts.csv')
            # Decode existing content
            current_data = base64.b64decode(file.content).decode()
            # Add new row
            new_data = current_data.rstrip() + f"\n{timestamp},{count}"
            # Update file
            store.update_file(
                'petition_counts.csv',
                f"Update petition count: {count}",
                new_data,
                file.sha
            )
        except:
            # File doesn't exist, create new with headers
            content = f"timestamp,count\n{timestamp},{count}"
            store.create_file(
                'petition_counts.csv',
                f"Initial petition count: {count}",
                content
            )
    except Exception as e:
        st.error(f"Error saving to GitHub: {e}")

def calculate_metrics(df):
    """Calculate metrics including rolling averages."""
    if len(df) == 0:
        return 0, 0, 0, pd.Series(), pd.Series()
    
    latest_count = df['count'].iloc[-1]
    
    # Calculate rolling averages
    df = df.set_index('timestamp')
    rolling_5min = df['count'].rolling('5min').mean()
    rolling_15min = df['count'].rolling('15min').mean()
    
    # Calculate rates using masks instead of last()
    current_time = df.index.max()
    hour_ago = current_time - timedelta(hours=1)
    five_min_ago = current_time - timedelta(minutes=5)
    
    # Create masks for time periods
    hour_mask = df.index >= hour_ago
    five_min_mask = df.index >= five_min_ago
    
    # Calculate hourly rate
    if hour_mask.sum() > 1:
        hour_start_count = df.loc[hour_mask, 'count'].iloc[0]
        hourly_rate = latest_count - hour_start_count
    else:
        hourly_rate = 0
    
    # Calculate minute rate
    if five_min_mask.sum() > 1:
        five_min_start_count = df.loc[five_min_mask, 'count'].iloc[0]
        minute_rate = (latest_count - five_min_start_count) / 5
    else:
        minute_rate = 0
    
    return latest_count, hourly_rate, minute_rate, rolling_5min, rolling_15min

def create_live_chart(df, rolling_5min, rolling_15min):
    """Create a live-updating chart with rolling averages."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Raw data
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['count'],
                  name="Raw Count",
                  mode='lines',
                  line=dict(color='blue', width=1)),
        secondary_y=False
    )
    
    # 5-minute rolling average
    #fig.add_trace(
    #    go.Scatter(x=rolling_5min.index, y=rolling_5min,
    #              name="5-min Average",
    #              mode='lines',
    #              line=dict(color='red', width=2)),
    #    secondary_y=False
    #)
    
    # 15-minute rolling average
    #fig.add_trace(
    #    go.Scatter(x=rolling_15min.index, y=rolling_15min,
    #              name="15-min Average",
    #              mode='lines',
    #              line=dict(color='green', width=2)),
    #    secondary_y=False
    #)
    
    # Calculate and plot rate of change using masks
    df_rate = df.copy()
    df_rate['rate'] = df_rate['count'].diff() / ((df_rate['timestamp'] - df_rate['timestamp'].shift(1)).dt.total_seconds() / 60)
    
    fig.add_trace(
        go.Scatter(x=df_rate['timestamp'], y=df_rate['rate'],
                  name="Rate (signatures/min)",
                  mode='lines',
                  line=dict(color='orange', width=1)),
        secondary_y=True
    )
    
    fig.update_layout(
        title='Live Petition Count with Rolling Averages',
        xaxis_title="Time",
        yaxis_title="Signatures",
        yaxis2_title="Signatures per Minute",
        hovermode='x unified',
        height=600
    )
    
    return fig

def create_activity_heatmap(df):
    """Create a heatmap showing signature activity by hour and day."""
    if len(df) < 2:
        return None
    
    # Add day and hour columns
    df_heatmap = df.copy()
    df_heatmap['hour'] = df_heatmap['timestamp'].dt.hour
    df_heatmap['day'] = df_heatmap['timestamp'].dt.date
    
    # Calculate signatures per hour using diff
    df_heatmap['signatures'] = df_heatmap['count'].diff()
    
    # Create pivot table for heatmap
    pivot_table = df_heatmap.pivot_table(
        values='signatures',
        index='day',
        columns='hour',
        aggfunc='sum',
        fill_value=0
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Day: %{y}<br>Hour: %{x}<br>Signatures: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Signature Activity Heatmap',
        xaxis_title='Hour of Day',
        yaxis_title='Date',
        height=400
    )
    
    return fig

def main():
    st.title("📊 Petition Tracker Dashboard")
    
    url = "https://petition.parliament.uk/petitions/700143"
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Load and update data
    df = load_data()
    current_count = get_petition_count(url)
    
    if current_count is not None and (len(df) == 0 or current_count != st.session_state.last_count):
        timestamp = datetime.now()
        log_count(timestamp, current_count)
        st.session_state.last_count = current_count
        df = load_data()
    
    # Calculate metrics
    latest_count, hourly_rate, minute_rate, rolling_5min, rolling_15min = calculate_metrics(df)
    
    # Display metrics
    with col1:
        st.metric("Current Count", f"{latest_count:,}")
    
    with col2:
        st.metric("Signatures per Hour", f"{hourly_rate:,}")
    
    with col3:
        st.metric("Signatures per Minute", f"{minute_rate:.1f}")
    
    with col4:
        if len(df) > 0:
            time_since_start = (datetime.now() - df['timestamp'].iloc[0])
            st.metric("Time Tracking", 
                     f"{time_since_start.days}d {time_since_start.seconds//3600}h {(time_since_start.seconds//60)%60}m")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Live Tracking", "Activity Heatmap"])
    
    with tab1:
        # Update live chart
        if not df.empty:
            fig = create_live_chart(df, rolling_5min, rolling_15min)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show recent data
            st.subheader("Recent Readings")
            n_recent = 10
            
            # Create recent dataframe with rate calculation
            recent_df = df.copy().sort_values('timestamp', ascending=False).head(n_recent)
            # Calculate rate (signatures per minute)
            recent_df['rate'] = recent_df['count'].diff(-1).abs() / \
                            (recent_df['timestamp'].diff(-1).dt.total_seconds() / 60)
            
            # Select and rename columns
            display_df = recent_df[['timestamp', 'count', 'rate']].copy()
            
            # Format the dataframe for display
            st.dataframe(
                display_df.style.format({
                    'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S'),
                    'count': '{:,}',
                    'rate': '{:.1f}'
                }),
                use_container_width=True
            )
    
    with tab2:
        # Display activity heatmap
        if not df.empty:
            heatmap_fig = create_activity_heatmap(df)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # Add summary statistics
                st.subheader("Activity Summary")
                df_summary = df.copy()
                df_summary['hour'] = df_summary['timestamp'].dt.hour
                df_summary['signatures'] = df_summary['count'].diff()
                
                hour_sums = df_summary.groupby('hour')['signatures'].sum()
                peak_hour = hour_sums.idxmax()
                peak_signatures = hour_sums.max()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Peak Activity Hour", f"{peak_hour:02d}:00")
                with col2:
                    st.metric("Peak Hour Signatures", f"{int(peak_signatures):,}")
    
    # Auto-refresh
    time.sleep(1)
    st.rerun()

if __name__ == "__main__":
    main()