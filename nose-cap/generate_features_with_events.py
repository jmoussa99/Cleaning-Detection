"""
Complete script to generate all_features with event column and export to CSV
This recreates the necessary processing from the notebook
"""
import json
import pandas as pd
import numpy as np
import pytz
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta

print("="*70)
print("GENERATING ALL_FEATURES WITH EVENTS")
print("="*70)

# Since we need the all_features dataframe from the notebook session,
# this script provides a simpler approach: load the existing CSV and add events

print("\nOption 1: If all_features CSV already exists")
print("-" * 70)

try:
    # Try to load existing all_features CSV
    print("Attempting to load existing all_features data...")
    all_features = pd.read_csv('all_features_sample_100.csv')
    
    # Convert timestamp columns back to datetime
    timestamp_cols = ['open_start_time', 'open_end_time', 'close_start_time', 'close_end_time']
    for col in timestamp_cols:
        if col in all_features.columns:
            all_features[col] = pd.to_datetime(all_features[col])
    
    print(f"[OK] Loaded all_features with shape: {all_features.shape}")
    
    # Read the Report Summary CSV
    print("\nLoading Report Summary...")
    report_summary = pd.read_csv('Report Summary(14-247 Nose Cap).csv')
    
    # Clean up the report summary
    report_summary = report_summary.dropna(subset=['Date', 'Time'], how='all')
    report_summary['Date'] = pd.to_datetime(report_summary['Date'], format='%m/%d/%Y', errors='coerce')
    report_summary['Event'] = report_summary['Event'].fillna('no event').replace('', 'no event')
    
    print(f"[OK] Loaded Report Summary with {len(report_summary)} entries")
    print(f"  Date range: {report_summary['Date'].min()} to {report_summary['Date'].max()}")
    
    # Create event mapping function
    def get_event_for_timestamp(timestamp):
        """
        Match timestamp to event from report summary based on date and time period.
        Day shift: 7am-7pm (07:00-19:00)
        Night shift: 7pm-7am (19:00-07:00)
        """
        if pd.isna(timestamp):
            return 'no event'
        
        # Extract date and hour from timestamp
        date = timestamp.date()
        hour = timestamp.hour
        
        # Determine shift based on hour
        if 7 <= hour < 19:
            shift = 'Day'
            check_date = date
        else:
            shift = 'Night'
            # For night shift early morning (00:00-06:59), use previous date
            if hour < 7:
                check_date = (pd.Timestamp(date) - pd.Timedelta(days=1)).date()
            else:
                check_date = date
        
        # Find matching event in report summary
        matching_rows = report_summary[
            (report_summary['Date'].dt.date == check_date) & 
            (report_summary['Time'] == shift)
        ]
        
        if len(matching_rows) > 0:
            event = matching_rows.iloc[0]['Event']
            return event if pd.notna(event) and str(event).strip() != '' else 'no event'
        else:
            return 'no event'
    
    # Apply the function to create the event column
    print("\nMapping events to all_features dataframe...")
    all_features['event'] = all_features['open_start_time'].apply(get_event_for_timestamp)
    
    print(f"[OK] Event column added successfully!")
    print(f"  all_features shape: {all_features.shape}")
    
    # Show event distribution
    print(f"\nEvent value counts:")
    event_counts = all_features['event'].value_counts()
    for event, count in event_counts.head(10).items():
        print(f"  {event}: {count}")
    if len(event_counts) > 10:
        print(f"  ... and {len(event_counts) - 10} more unique events")
    
    # Export to CSV
    output_filename = 'all_features_with_events_100.csv'
    all_features.to_csv(output_filename, index=False)
    print(f"\n[OK] Exported all features with events to: {output_filename}")
    print(f"  File shape: {all_features.shape}")
    print(f"  Columns: {list(all_features.columns)}")
    
    # Display sample
    print(f"\nSample rows with events:")
    sample_cols = ['open_start_time', 'date_range', 'operational_phase', 'event']
    available_cols = [col for col in sample_cols if col in all_features.columns]
    print(all_features[available_cols].head(10).to_string())
    
    print("\n" + "="*70)
    print("SUCCESS! File generated successfully")
    print("="*70)
    
except FileNotFoundError as e:
    print(f"\n[ERROR] Could not find required file")
    print(f"  {e}")
    print(f"\nPlease ensure you have run the notebook cells to generate:")
    print(f"  - all_features_sample_100.csv")
    print(f"  - Report Summary(14-247 Nose Cap).csv")
    print(f"\nOr run this code as a new cell in your Jupyter notebook instead.")
    
except Exception as e:
    print(f"\n[ERROR] Error occurred: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)

