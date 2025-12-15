import re
import pandas as pd


# First, read the labeled CSV to find the latest cleaning date
print("Loading labeled data to find latest cleaning...")
all_features = pd.read_csv('nose_cap_14-247-labeled.csv')

# Convert timestamp columns to datetime
all_features['open_start_time'] = pd.to_datetime(all_features['open_start_time'])

# Find the latest cleaning date
labeled_rows = all_features[all_features['label'].notna() & (all_features['label'] != '')]
if len(labeled_rows) > 0:
    latest_cleaning_date = labeled_rows['open_start_time'].max()
    print(f"Latest cleaning found at: {latest_cleaning_date}")
    print(f"Total existing labeled rows: {len(labeled_rows)}")
    cutoff_date = latest_cleaning_date.date()
else:
    cutoff_date = None
    print("No existing cleaning labels found. Will process all events.")

# Read the Report Summary CSV (reload to ensure clean state)
report_summary = pd.read_csv('Report Summary(14-247 Nose Cap).csv')

# Clean up the report summary - remove empty rows
report_summary = report_summary.dropna(subset=['Date', 'Time'], how='all')

# Convert Date column to datetime
report_summary['Date'] = pd.to_datetime(report_summary['Date'], format='%m/%d/%Y', errors='coerce')

# Fill empty Event values with 'no event'  
report_summary['Event'] = report_summary['Event'].fillna('no event').replace('', 'no event')

print(f"Report Summary shape: {report_summary.shape}")
print(f"Date range: {report_summary['Date'].min()} to {report_summary['Date'].max()}")

# Extract cleaning timestamps from events
def extract_cleaning_time(event_text):
    """
    Extract cleaning timestamp from event text if it contains 'clean' and a time.
    Returns time string or None.
    """
    if pd.isna(event_text) or event_text == 'no event':
        return None
    
    event_lower = str(event_text).lower()
    
    # Check if event contains 'clean'
    if 'clean' not in event_lower:
        return None
    
    # Look for time patterns: "6:45PM", "7 am", "8pm", "1am", etc.
    time_pattern = r'(\d{1,2}):?(\d{2})?\s*(am|pm)'
    match = re.search(time_pattern, event_lower)
    
    if match:
        hour = int(match.group(1))
        minute = match.group(2) if match.group(2) else '00'
        period = match.group(3)
        return f"{hour}:{minute} {period}"
    
    return None

# Add cleaning time info to report summary
report_summary['cleaning_time'] = report_summary['Event'].apply(extract_cleaning_time)

# Build a lookup of cleaning times by date and shift
# Only include events AFTER the latest cleaning date
cleaning_lookup = {}
for _, row in report_summary[report_summary['cleaning_time'].notna()].iterrows():
    date = row['Date'].date()
    shift = row['Time']
    cleaning_time = row['cleaning_time']
    
    # Skip if this date is on or before the cutoff date
    if cutoff_date is not None and date <= cutoff_date:
        continue
    
    # Parse cleaning time to get hour and minute
    time_pattern = r'(\d{1,2}):(\d{2})\s*(am|pm)'
    match = re.search(time_pattern, cleaning_time.lower())
    
    if match:
        clean_hour = int(match.group(1))
        clean_minute = int(match.group(2))
        period = match.group(3)
        
        # Convert to 24-hour format
        if period == 'pm' and clean_hour != 12:
            clean_hour += 12
        elif period == 'am' and clean_hour == 12:
            clean_hour = 0
        
        key = (date, shift)
        cleaning_lookup[key] = (clean_hour, clean_minute, cleaning_time)

print(f"\nFound {len(cleaning_lookup)} NEW cleaning times to match (after {cutoff_date})")


# Initialize label column if it doesn't exist, otherwise preserve existing labels
if 'label' not in all_features.columns:
    all_features['label'] = ''
else:
    # Fill NaN values with empty string
    all_features['label'] = all_features['label'].fillna('')

# Process each cleaning event
for (date, shift), (clean_hour, clean_minute, cleaning_time_str) in cleaning_lookup.items():
    print(f"\nProcessing cleaning event: {date} {shift} at {cleaning_time_str}")
    
    # Filter rows for this date and shift
    if shift == 'Day':
        # Day shift: 7am-7pm
        date_rows = all_features[
            (all_features['open_start_time'].dt.date == date) &
            (all_features['open_start_time'].dt.hour >= 7) &
            (all_features['open_start_time'].dt.hour < 19)
        ].copy()
    else:
        # Night shift: 7pm-7am (spans two calendar days)
        next_date = pd.Timestamp(date) + pd.Timedelta(days=1)
        date_rows = all_features[
            (
                ((all_features['open_start_time'].dt.date == date) & (all_features['open_start_time'].dt.hour >= 19)) |
                ((all_features['open_start_time'].dt.date == next_date.date()) & (all_features['open_start_time'].dt.hour < 7))
            )
        ].copy()
    
    if len(date_rows) == 0:
        print(f"  No rows found for this date/shift")
        continue
    
    # Find rows with gap > 0.333 within 10 minutes of the cleaning time
    candidates = []
    for idx, row in date_rows.iterrows():
        gap = row['gap_between_cycles_minutes']
        if pd.notna(gap) and gap > 0.333:
            hour = row['open_start_time'].hour
            minute = row['open_start_time'].minute
            time_diff = abs((hour * 60 + minute) - (clean_hour * 60 + clean_minute))
            
            if time_diff <= 10:  # Within 10 minutes
                candidates.append((idx, time_diff, gap, row['open_start_time']))
    
    if len(candidates) == 0:
        print(f"  No rows with significant gaps within time window")
        continue
    
    # Sort by gap size (descending) to find the row with largest gap
    # This is typically when production resumes (cleaning_end)
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # The row with the largest gap near cleaning time is cleaning_end
    cleaning_end_idx = candidates[0][0]
    all_features.loc[cleaning_end_idx, 'label'] = 'cleaning_end'
    cleaning_end_gap = candidates[0][2]
    print(f"  Labeled cleaning_end at index {cleaning_end_idx} (gap={cleaning_end_gap:.2f})")
    
    # Find the previous row with gap > 0.333 (before cleaning_end in time)
    cleaning_end_time = candidates[0][3]
    
    # Look for rows with gap > 0.333 that are before cleaning_end_time
    potential_starts = []
    for idx, row in date_rows.iterrows():
        gap = row['gap_between_cycles_minutes']
        if pd.notna(gap) and gap > 0.333:
            if row['open_start_time'] < cleaning_end_time:
                potential_starts.append((idx, row['open_start_time'], gap))
    
    if len(potential_starts) > 0:
        # Get the most recent one before cleaning_end
        potential_starts.sort(key=lambda x: x[1], reverse=True)
        cleaning_start_idx = potential_starts[0][0]
        cleaning_start_gap = potential_starts[0][2]
        all_features.loc[cleaning_start_idx, 'label'] = 'cleaning_start'
        print(f"  Labeled cleaning_start at index {cleaning_start_idx} (gap={cleaning_start_gap:.2f})")
    else:
        print(f"  No cleaning_start found before cleaning_end")


# Show rows with cleaning labels
print("\n" + "="*60)
print("Summary of Cleaning Labels:")
print("="*60)

labeled_rows = all_features[all_features['label'] != '']
print(f"Total rows labeled: {len(labeled_rows)}")
print(f"  cleaning_start: {(all_features['label'] == 'cleaning_start').sum()}")
print(f"  cleaning_end: {(all_features['label'] == 'cleaning_end').sum()}")

if len(labeled_rows) > 0:
    print("\nSample of labeled rows:")
    print(labeled_rows[['open_start_time', 'date_range', 'gap_between_cycles_minutes', 'label']].head(20).to_string())
    
    print("\nAll cleaning events by date:")
    for date in sorted(labeled_rows['date_range'].unique()):
        date_labels = labeled_rows[labeled_rows['date_range'] == date].sort_values('open_start_time')
        if len(date_labels) > 0:
            print(f"\n{date}:")
            print(date_labels[['open_start_time', 'gap_between_cycles_minutes', 'label']].to_string())

# Save the updated labeled CSV
print("\n" + "="*60)
print("Saving updated labeled data...")
all_features.to_csv('nose_cap_14-247-labeled.csv', index=False)
print(f"Saved to: nose_cap_14-247-labeled.csv")
print("="*60)