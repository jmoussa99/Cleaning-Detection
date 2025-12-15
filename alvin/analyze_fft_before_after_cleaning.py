"""
FFT Analysis Before and After Cleaning Events

This script analyzes FFT patterns before and after cleaning events to visualize
how the frequency characteristics of valve operations change after maintenance.

Uses a 45-row window before cleaning_start and 45-row window after cleaning_end
to compare the FFT patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD AND PARSE DATA
# ============================================================================

print("=" * 80)
print("FFT ANALYSIS: BEFORE AND AFTER CLEANING EVENTS")
print("=" * 80)

print("\nLoading data...")
df = pd.read_csv('24-119_all_features_with_labels.csv')

# Check if 'label' column exists, otherwise try 'event'
if 'label' not in df.columns:
    if 'event' in df.columns:
        df['label'] = df['event']
    else:
        print("ERROR: No 'label' or 'event' column found!")
        exit(1)

print(f"Data loaded: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Parse FFT columns from string to arrays
print("\nParsing FFT data...")
array_columns = ['open_raw_fft_normalized', 'close_raw_fft_normalized']

for col in array_columns:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        print(f"  Parsed {col}")

# ============================================================================
# DETECT REPAIR PERIODS
# ============================================================================

print("\n" + "=" * 80)
print("DETECTING REPAIR/MAINTENANCE PERIODS")
print("=" * 80)

# Convert timestamp to datetime
df['open_start_time'] = pd.to_datetime(df['open_start_time'])

# Detect repair periods by looking for large gaps in data (>12 hours)
df['time_gap_hours'] = df['open_start_time'].diff().dt.total_seconds() / 3600
large_gaps = df[df['time_gap_hours'] > 12].copy()

print(f"\nFound {len(large_gaps)} large gaps (>12 hours) indicating potential repairs/downtime")

# Mark cycles as "post-repair" if they occur within 48 hours after a large gap
df['is_post_repair'] = False
df['hours_since_repair'] = np.nan

for idx in large_gaps.index:
    gap_time = df.loc[idx, 'open_start_time']
    # Mark next 48 hours as post-repair period
    post_repair_mask = (
        (df['open_start_time'] >= gap_time) & 
        (df['open_start_time'] <= gap_time + pd.Timedelta(hours=48))
    )
    df.loc[post_repair_mask, 'is_post_repair'] = True
    df.loc[post_repair_mask, 'hours_since_repair'] = (
        (df.loc[post_repair_mask, 'open_start_time'] - gap_time).dt.total_seconds() / 3600
    )

post_repair_count = df['is_post_repair'].sum()
print(f"Marked {post_repair_count} cycles as 'post-repair' (within 48h after large gap)")

# ============================================================================
# IDENTIFY CLEANING EVENTS
# ============================================================================

print("\n" + "=" * 80)
print("IDENTIFYING CLEANING EVENTS")
print("=" * 80)

# Find cleaning start and end indices
cleaning_starts = df[df['label'] == 'cleaning_start'].index.tolist()
cleaning_ends = df[df['label'] == 'cleaning_end'].index.tolist()

print(f"\nFound {len(cleaning_starts)} cleaning_start events")
print(f"Found {len(cleaning_ends)} cleaning_end events")

if len(cleaning_starts) == 0 or len(cleaning_ends) == 0:
    print("ERROR: No cleaning events found!")
    exit(1)

# Categorize cleanings as normal or post-repair
normal_cleanings = []
post_repair_cleanings = []

for start_idx in cleaning_starts:
    if df.loc[start_idx, 'is_post_repair']:
        post_repair_cleanings.append(start_idx)
    else:
        normal_cleanings.append(start_idx)

print(f"\nNormal cleanings: {len(normal_cleanings)}")
print(f"Post-repair cleanings: {len(post_repair_cleanings)}")

# Ensure we have matching pairs
num_pairs = min(len(cleaning_starts), len(cleaning_ends))
print(f"\nAnalyzing {num_pairs} cleaning event pairs total")

# ============================================================================
# EXTRACT BEFORE/AFTER WINDOWS FOR EACH CLEANING EVENT
# ============================================================================

window_size = 45
cleaning_analyses = []

print("\n" + "=" * 80)
print(f"EXTRACTING {window_size}-ROW WINDOWS BEFORE/AFTER EACH CLEANING")
print("=" * 80)

for i in range(num_pairs):
    start_idx = cleaning_starts[i]
    end_idx = cleaning_ends[i]
    
    print(f"\n--- Cleaning Event {i+1} ---")
    print(f"Start index: {start_idx}")
    print(f"End index: {end_idx}")
    
    # Extract before window (45 rows before cleaning_start)
    before_start = max(0, start_idx - window_size)
    before_indices = list(range(before_start, start_idx))
    
    # Extract after window (45 rows after cleaning_end)
    after_end = min(len(df), end_idx + window_size + 1)
    after_indices = list(range(end_idx + 1, after_end))
    
    print(f"Before window: {len(before_indices)} rows (indices {before_start} to {start_idx-1})")
    print(f"After window: {len(after_indices)} rows (indices {end_idx+1} to {after_end-1})")
    
    if len(before_indices) == 0 or len(after_indices) == 0:
        print("WARNING: Insufficient data for this cleaning event, skipping...")
        continue
    
    # Extract FFT data
    before_data = df.iloc[before_indices]
    after_data = df.iloc[after_indices]
    
    # Get timestamps if available
    if 'open_start_time' in df.columns:
        start_time = df.iloc[start_idx]['open_start_time']
        end_time = df.iloc[end_idx]['close_end_time'] if 'close_end_time' in df.columns else 'N/A'
    else:
        start_time = f"Index {start_idx}"
        end_time = f"Index {end_idx}"
    
    # Determine if this is a post-repair cleaning
    is_post_repair = df.loc[start_idx, 'is_post_repair']
    hours_since_repair = df.loc[start_idx, 'hours_since_repair'] if is_post_repair else None
    
    # Store analysis info
    cleaning_analyses.append({
        'event_num': i + 1,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'start_time': start_time,
        'end_time': end_time,
        'before_indices': before_indices,
        'after_indices': after_indices,
        'before_data': before_data,
        'after_data': after_data,
        'is_post_repair': is_post_repair,
        'hours_since_repair': hours_since_repair
    })

print(f"\n✓ Successfully extracted data for {len(cleaning_analyses)} cleaning events")

# ============================================================================
# COMPUTE FFT STATISTICS FOR EACH CLEANING EVENT
# ============================================================================

print("\n" + "=" * 80)
print("COMPUTING FFT STATISTICS")
print("=" * 80)

for analysis in cleaning_analyses:
    event_num = analysis['event_num']
    before_data = analysis['before_data']
    after_data = analysis['after_data']
    is_post_repair = analysis['is_post_repair']
    
    repair_status = "POST-REPAIR" if is_post_repair else "NORMAL"
    print(f"\nEvent {event_num} ({repair_status}):")
    
    # Process OPEN FFTs
    if 'open_raw_fft_normalized' in before_data.columns:
        before_open_ffts = np.array(before_data['open_raw_fft_normalized'].tolist())
        after_open_ffts = np.array(after_data['open_raw_fft_normalized'].tolist())
        
        analysis['before_open_mean'] = np.mean(before_open_ffts, axis=0)
        analysis['before_open_std'] = np.std(before_open_ffts, axis=0)
        analysis['after_open_mean'] = np.mean(after_open_ffts, axis=0)
        analysis['after_open_std'] = np.std(after_open_ffts, axis=0)
        
        # Calculate change
        analysis['open_mean_change'] = analysis['after_open_mean'] - analysis['before_open_mean']
        analysis['open_mean_change_pct'] = (analysis['open_mean_change'] / (analysis['before_open_mean'] + 1e-10)) * 100
        
        print(f"  OPEN FFT - Before mean range: [{analysis['before_open_mean'].min():.4f}, {analysis['before_open_mean'].max():.4f}]")
        print(f"  OPEN FFT - After mean range: [{analysis['after_open_mean'].min():.4f}, {analysis['after_open_mean'].max():.4f}]")
        print(f"  OPEN FFT - Mean absolute change: {np.mean(np.abs(analysis['open_mean_change'])):.4f}")
    
    # Process CLOSE FFTs
    if 'close_raw_fft_normalized' in before_data.columns:
        before_close_ffts = np.array(before_data['close_raw_fft_normalized'].tolist())
        after_close_ffts = np.array(after_data['close_raw_fft_normalized'].tolist())
        
        analysis['before_close_mean'] = np.mean(before_close_ffts, axis=0)
        analysis['before_close_std'] = np.std(before_close_ffts, axis=0)
        analysis['after_close_mean'] = np.mean(after_close_ffts, axis=0)
        analysis['after_close_std'] = np.std(after_close_ffts, axis=0)
        
        # Calculate change
        analysis['close_mean_change'] = analysis['after_close_mean'] - analysis['before_close_mean']
        analysis['close_mean_change_pct'] = (analysis['close_mean_change'] / (analysis['before_close_mean'] + 1e-10)) * 100
        
        print(f"  CLOSE FFT - Before mean range: [{analysis['before_close_mean'].min():.4f}, {analysis['before_close_mean'].max():.4f}]")
        print(f"  CLOSE FFT - After mean range: [{analysis['after_close_mean'].min():.4f}, {analysis['after_close_mean'].max():.4f}]")
        print(f"  CLOSE FFT - Mean absolute change: {np.mean(np.abs(analysis['close_mean_change'])):.4f}")
    
    # Calculate FFT STD statistics (key metric for vibration strength)
    if 'open_fft_std' in before_data.columns:
        analysis['before_open_fft_std_mean'] = before_data['open_fft_std'].mean()
        analysis['after_open_fft_std_mean'] = after_data['open_fft_std'].mean()
        analysis['open_fft_std_change'] = analysis['after_open_fft_std_mean'] - analysis['before_open_fft_std_mean']
        analysis['open_fft_std_change_pct'] = (analysis['open_fft_std_change'] / analysis['before_open_fft_std_mean']) * 100
        
        print(f"  OPEN FFT STD - Before: {analysis['before_open_fft_std_mean']:.2f}, "
              f"After: {analysis['after_open_fft_std_mean']:.2f}, "
              f"Change: {analysis['open_fft_std_change']:.2f} ({analysis['open_fft_std_change_pct']:.1f}%)")
    
    if 'close_fft_std' in before_data.columns:
        analysis['before_close_fft_std_mean'] = before_data['close_fft_std'].mean()
        analysis['after_close_fft_std_mean'] = after_data['close_fft_std'].mean()
        analysis['close_fft_std_change'] = analysis['after_close_fft_std_mean'] - analysis['before_close_fft_std_mean']
        analysis['close_fft_std_change_pct'] = (analysis['close_fft_std_change'] / analysis['before_close_fft_std_mean']) * 100
        
        print(f"  CLOSE FFT STD - Before: {analysis['before_close_fft_std_mean']:.2f}, "
              f"After: {analysis['after_close_fft_std_mean']:.2f}, "
              f"Change: {analysis['close_fft_std_change']:.2f} ({analysis['close_fft_std_change_pct']:.1f}%)")

# Skip individual event plots - will create time series instead

# ============================================================================
# AGGREGATE ANALYSIS: ALL CLEANING EVENTS COMBINED
# ============================================================================

print("\n" + "=" * 80)
print("AGGREGATE ANALYSIS: ALL CLEANING EVENTS")
print("=" * 80)

# Collect all before/after data
all_before_open = []
all_after_open = []
all_before_close = []
all_after_close = []

for analysis in cleaning_analyses:
    if 'before_open_mean' in analysis:
        all_before_open.append(analysis['before_open_mean'])
        all_after_open.append(analysis['after_open_mean'])
    
    if 'before_close_mean' in analysis:
        all_before_close.append(analysis['before_close_mean'])
        all_after_close.append(analysis['after_close_mean'])

# Compute aggregate statistics
if len(all_before_open) > 0:
    agg_before_open_mean = np.mean(all_before_open, axis=0)
    agg_before_open_std = np.std(all_before_open, axis=0)
    agg_after_open_mean = np.mean(all_after_open, axis=0)
    agg_after_open_std = np.std(all_after_open, axis=0)
    agg_open_change = agg_after_open_mean - agg_before_open_mean
    
    print(f"\nOPEN FFT Aggregate Statistics (across {len(all_before_open)} cleaning events):")
    print(f"  Before - Mean range: [{agg_before_open_mean.min():.4f}, {agg_before_open_mean.max():.4f}]")
    print(f"  After - Mean range: [{agg_after_open_mean.min():.4f}, {agg_after_open_mean.max():.4f}]")
    print(f"  Average absolute change: {np.mean(np.abs(agg_open_change)):.4f}")
    print(f"  Max absolute change: {np.max(np.abs(agg_open_change)):.4f}")

if len(all_before_close) > 0:
    agg_before_close_mean = np.mean(all_before_close, axis=0)
    agg_before_close_std = np.std(all_before_close, axis=0)
    agg_after_close_mean = np.mean(all_after_close, axis=0)
    agg_after_close_std = np.std(all_after_close, axis=0)
    agg_close_change = agg_after_close_mean - agg_before_close_mean
    
    print(f"\nCLOSE FFT Aggregate Statistics (across {len(all_before_close)} cleaning events):")
    print(f"  Before - Mean range: [{agg_before_close_mean.min():.4f}, {agg_before_close_mean.max():.4f}]")
    print(f"  After - Mean range: [{agg_after_close_mean.min():.4f}, {agg_after_close_mean.max():.4f}]")
    print(f"  Average absolute change: {np.mean(np.abs(agg_close_change)):.4f}")
    print(f"  Max absolute change: {np.max(np.abs(agg_close_change)):.4f}")

# ============================================================================
# VISUALIZE AGGREGATE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING AGGREGATE VISUALIZATION")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'Aggregate FFT Analysis: Before vs After Cleaning\n'
             f'(Average across {len(cleaning_analyses)} cleaning events, {window_size}-row windows)', 
             fontsize=16, fontweight='bold')

if len(all_before_open) > 0:
    freq_bins = np.arange(len(agg_before_open_mean))
    
    # Plot 1: OPEN FFT - Before vs After
    axes[0, 0].plot(freq_bins, agg_before_open_mean, 'b-', 
                   label='Before Cleaning', linewidth=2.5, alpha=0.8)
    axes[0, 0].fill_between(freq_bins, 
                            agg_before_open_mean - agg_before_open_std,
                            agg_before_open_mean + agg_before_open_std,
                            alpha=0.2, color='blue')
    
    axes[0, 0].plot(freq_bins, agg_after_open_mean, 'r-', 
                   label='After Cleaning', linewidth=2.5, alpha=0.8)
    axes[0, 0].fill_between(freq_bins, 
                            agg_after_open_mean - agg_after_open_std,
                            agg_after_open_mean + agg_after_open_std,
                            alpha=0.2, color='red')
    
    axes[0, 0].set_title('OPEN FFT: Before vs After', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('FFT Frequency Bin', fontsize=11)
    axes[0, 0].set_ylabel('Normalized Magnitude', fontsize=11)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: OPEN FFT - Change
    axes[0, 1].plot(freq_bins, agg_open_change, 'purple', linewidth=2.5)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].fill_between(freq_bins, 0, agg_open_change, 
                           where=agg_open_change>=0, 
                           alpha=0.3, color='green', label='Increase')
    axes[0, 1].fill_between(freq_bins, 0, agg_open_change, 
                           where=agg_open_change<0, 
                           alpha=0.3, color='red', label='Decrease')
    
    axes[0, 1].set_title('OPEN FFT: Average Change', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('FFT Frequency Bin', fontsize=11)
    axes[0, 1].set_ylabel('Change in Magnitude', fontsize=11)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: OPEN FFT - Percentage Change
    open_pct_change = (agg_open_change / (agg_before_open_mean + 1e-10)) * 100
    axes[0, 2].plot(freq_bins, open_pct_change, 'orange', linewidth=2.5)
    axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 2].set_title('OPEN FFT: Percentage Change', fontsize=13, fontweight='bold')
    axes[0, 2].set_xlabel('FFT Frequency Bin', fontsize=11)
    axes[0, 2].set_ylabel('Change (%)', fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)

if len(all_before_close) > 0:
    freq_bins = np.arange(len(agg_before_close_mean))
    
    # Plot 4: CLOSE FFT - Before vs After
    axes[1, 0].plot(freq_bins, agg_before_close_mean, 'b-', 
                   label='Before Cleaning', linewidth=2.5, alpha=0.8)
    axes[1, 0].fill_between(freq_bins, 
                            agg_before_close_mean - agg_before_close_std,
                            agg_before_close_mean + agg_before_close_std,
                            alpha=0.2, color='blue')
    
    axes[1, 0].plot(freq_bins, agg_after_close_mean, 'r-', 
                   label='After Cleaning', linewidth=2.5, alpha=0.8)
    axes[1, 0].fill_between(freq_bins, 
                            agg_after_close_mean - agg_after_close_std,
                            agg_after_close_mean + agg_after_close_std,
                            alpha=0.2, color='red')
    
    axes[1, 0].set_title('CLOSE FFT: Before vs After', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('FFT Frequency Bin', fontsize=11)
    axes[1, 0].set_ylabel('Normalized Magnitude', fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: CLOSE FFT - Change
    axes[1, 1].plot(freq_bins, agg_close_change, 'purple', linewidth=2.5)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].fill_between(freq_bins, 0, agg_close_change, 
                           where=agg_close_change>=0, 
                           alpha=0.3, color='green', label='Increase')
    axes[1, 1].fill_between(freq_bins, 0, agg_close_change, 
                           where=agg_close_change<0, 
                           alpha=0.3, color='red', label='Decrease')
    
    axes[1, 1].set_title('CLOSE FFT: Average Change', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('FFT Frequency Bin', fontsize=11)
    axes[1, 1].set_ylabel('Change in Magnitude', fontsize=11)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: CLOSE FFT - Percentage Change
    close_pct_change = (agg_close_change / (agg_before_close_mean + 1e-10)) * 100
    axes[1, 2].plot(freq_bins, close_pct_change, 'orange', linewidth=2.5)
    axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 2].set_title('CLOSE FFT: Percentage Change', fontsize=13, fontweight='bold')
    axes[1, 2].set_xlabel('FFT Frequency Bin', fontsize=11)
    axes[1, 2].set_ylabel('Change (%)', fontsize=11)
    axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fft_before_after_cleaning_aggregate.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: fft_before_after_cleaning_aggregate.png")
plt.close()

# ============================================================================
# CREATE TIME SERIES PLOT: ALL CLEANING WINDOWS
# ============================================================================

print("\n" + "=" * 80)
print("CREATING TIME SERIES PLOT: ALL CLEANING WINDOWS")
print("=" * 80)

# Find the repair time (when the large gap starts)
# Look for the October repair period - find when it STARTED (10-8), not ended (10-20)
repair_start_time = None
repair_end_time = None
for idx in large_gaps.index:
    gap_time = df.loc[idx, 'open_start_time']
    # Convert to timezone-naive if timezone-aware for comparison
    if gap_time.tz is not None:
        gap_time = gap_time.tz_localize(None)
    if gap_time >= pd.to_datetime('2025-10-01') and gap_time <= pd.to_datetime('2025-10-31'):
        if df.loc[idx, 'time_gap_hours'] > 200:  # The big 313-hour gap
            repair_end_time = gap_time
            # The repair started at the previous cycle time (before the gap)
            repair_start_time = df.loc[idx-1, 'open_start_time']
            print(f"\nFound major repair period:")
            print(f"  Started: {repair_start_time}")
            print(f"  Ended: {repair_end_time}")
            print(f"  Duration: {df.loc[idx, 'time_gap_hours']:.1f} hours ({df.loc[idx, 'time_gap_hours']/24:.1f} days)")
            break

# Create figure with 2 rows (OPEN and CLOSE)
fig, axes = plt.subplots(2, 1, figsize=(24, 12))
fig.suptitle(f'FFT STD Time Series: All Cleaning Windows\n'
             f'Blue = Before Cleaning, Red = After Cleaning | Orange = Repair Start, Green = Repair End', 
             fontsize=16, fontweight='bold')

# Plot 1: OPEN FFT STD Time Series
ax = axes[0]

# Plot all cleaning windows
for analysis in cleaning_analyses:
    start_time = pd.to_datetime(analysis['start_time'])
    
    if 'before_open_fft_std_mean' in analysis:
        # Before cleaning (blue)
        ax.scatter(start_time, analysis['before_open_fft_std_mean'], 
                  color='blue', s=150, alpha=0.6, marker='o')
        
        # After cleaning (red)
        ax.scatter(start_time, analysis['after_open_fft_std_mean'], 
                  color='red', s=150, alpha=0.6, marker='o')
        
        # Draw line connecting before/after
        ax.plot([start_time, start_time], 
               [analysis['before_open_fft_std_mean'], analysis['after_open_fft_std_mean']], 
               'gray', alpha=0.3, linewidth=1.5, linestyle='--')

# Draw vertical lines at repair START and END times
if repair_start_time is not None:
    ax.axvline(x=repair_start_time, color='orange', linewidth=3, linestyle='-', alpha=0.8)
if repair_end_time is not None:
    ax.axvline(x=repair_end_time, color='green', linewidth=3, linestyle='-', alpha=0.8)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
           markersize=10, alpha=0.6, label='Before Cleaning'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
           markersize=10, alpha=0.6, label='After Cleaning'),
    Line2D([0], [0], color='orange', linewidth=3, alpha=0.8,
           label=f'Repair Start ({repair_start_time.strftime("%Y-%m-%d") if repair_start_time else "N/A"})'),
    Line2D([0], [0], color='green', linewidth=3, alpha=0.8,
           label=f'Repair End ({repair_end_time.strftime("%Y-%m-%d") if repair_end_time else "N/A"})')
]
ax.legend(handles=legend_elements, fontsize=11, loc='upper left')
ax.set_xlabel('Date/Time', fontsize=13)
ax.set_ylabel('OPEN FFT STD', fontsize=13)
ax.set_title('OPEN FFT STD: Before vs After Cleaning (Time Series)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# Plot 2: CLOSE FFT STD Time Series
ax = axes[1]

# Plot all cleaning windows
for analysis in cleaning_analyses:
    start_time = pd.to_datetime(analysis['start_time'])
    
    if 'before_close_fft_std_mean' in analysis:
        # Before cleaning (blue)
        ax.scatter(start_time, analysis['before_close_fft_std_mean'], 
                  color='blue', s=150, alpha=0.6, marker='o')
        
        # After cleaning (red)
        ax.scatter(start_time, analysis['after_close_fft_std_mean'], 
                  color='red', s=150, alpha=0.6, marker='o')
        
        # Draw line connecting before/after
        ax.plot([start_time, start_time], 
               [analysis['before_close_fft_std_mean'], analysis['after_close_fft_std_mean']], 
               'gray', alpha=0.3, linewidth=1.5, linestyle='--')

# Draw vertical lines at repair START and END times
if repair_start_time is not None:
    ax.axvline(x=repair_start_time, color='orange', linewidth=3, linestyle='-', alpha=0.8)
if repair_end_time is not None:
    ax.axvline(x=repair_end_time, color='green', linewidth=3, linestyle='-', alpha=0.8)

ax.legend(handles=legend_elements, fontsize=11, loc='upper left')
ax.set_xlabel('Date/Time', fontsize=13)
ax.set_ylabel('CLOSE FFT STD', fontsize=13)
ax.set_title('CLOSE FFT STD: Before vs After Cleaning (Time Series)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('fft_std_time_series_all_cleanings.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: fft_std_time_series_all_cleanings.png")
plt.close()

# ============================================================================
# CREATE COMPARISON PLOT: NORMAL VS POST-REPAIR
# ============================================================================

print("\n" + "=" * 80)
print("CREATING COMPARISON: NORMAL VS POST-REPAIR CLEANINGS")
print("=" * 80)

# Separate normal and post-repair cleanings for comparison
normal_analyses = [a for a in cleaning_analyses if not a['is_post_repair']]
post_repair_analyses = [a for a in cleaning_analyses if a['is_post_repair']]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('FFT STD Comparison: Normal vs Post-Repair Cleanings', 
             fontsize=16, fontweight='bold')

# Collect statistics
normal_before_open = [a['before_open_fft_std_mean'] for a in normal_analyses if 'before_open_fft_std_mean' in a]
normal_after_open = [a['after_open_fft_std_mean'] for a in normal_analyses if 'after_open_fft_std_mean' in a]
post_repair_before_open = [a['before_open_fft_std_mean'] for a in post_repair_analyses if 'before_open_fft_std_mean' in a]
post_repair_after_open = [a['after_open_fft_std_mean'] for a in post_repair_analyses if 'after_open_fft_std_mean' in a]

normal_before_close = [a['before_close_fft_std_mean'] for a in normal_analyses if 'before_close_fft_std_mean' in a]
normal_after_close = [a['after_close_fft_std_mean'] for a in normal_analyses if 'after_close_fft_std_mean' in a]
post_repair_before_close = [a['before_close_fft_std_mean'] for a in post_repair_analyses if 'before_close_fft_std_mean' in a]
post_repair_after_close = [a['after_close_fft_std_mean'] for a in post_repair_analyses if 'after_close_fft_std_mean' in a]

# Plot 1: OPEN FFT STD - Box plot comparison
ax = axes[0, 0]
data_to_plot = [normal_before_open, normal_after_open, post_repair_before_open, post_repair_after_open]
labels = ['Normal\nBefore', 'Normal\nAfter', 'Post-Repair\nBefore', 'Post-Repair\nAfter']
colors = ['lightblue', 'blue', 'lightcoral', 'red']

bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel('OPEN FFT STD', fontsize=12)
ax.set_title('OPEN FFT STD: Before vs After Cleaning', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: CLOSE FFT STD - Box plot comparison
ax = axes[0, 1]
data_to_plot = [normal_before_close, normal_after_close, post_repair_before_close, post_repair_after_close]

bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel('CLOSE FFT STD', fontsize=12)
ax.set_title('CLOSE FFT STD: Before vs After Cleaning', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: OPEN FFT STD - Change magnitude
ax = axes[1, 0]
normal_open_changes = [a['open_fft_std_change'] for a in normal_analyses if 'open_fft_std_change' in a]
post_repair_open_changes = [a['open_fft_std_change'] for a in post_repair_analyses if 'open_fft_std_change' in a]

data_to_plot = [normal_open_changes, post_repair_open_changes]
labels = ['Normal\nCleanings', 'Post-Repair\nCleanings']
colors = ['blue', 'red']

bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel('OPEN FFT STD Change (After - Before)', fontsize=12)
ax.set_title('OPEN FFT STD: Change Magnitude', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: CLOSE FFT STD - Change magnitude
ax = axes[1, 1]
normal_close_changes = [a['close_fft_std_change'] for a in normal_analyses if 'close_fft_std_change' in a]
post_repair_close_changes = [a['close_fft_std_change'] for a in post_repair_analyses if 'close_fft_std_change' in a]

data_to_plot = [normal_close_changes, post_repair_close_changes]

bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel('CLOSE FFT STD Change (After - Before)', fontsize=12)
ax.set_title('CLOSE FFT STD: Change Magnitude', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fft_std_normal_vs_post_repair_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: fft_std_normal_vs_post_repair_comparison.png")
plt.close()

# Print summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

if len(normal_before_open) > 0:
    print(f"\nNORMAL CLEANINGS - OPEN FFT STD:")
    print(f"  Before: mean={np.mean(normal_before_open):.2f}, std={np.std(normal_before_open):.2f}")
    print(f"  After:  mean={np.mean(normal_after_open):.2f}, std={np.std(normal_after_open):.2f}")
    print(f"  Change: mean={np.mean(normal_open_changes):.2f}, std={np.std(normal_open_changes):.2f}")

if len(post_repair_before_open) > 0:
    print(f"\nPOST-REPAIR CLEANINGS - OPEN FFT STD:")
    print(f"  Before: mean={np.mean(post_repair_before_open):.2f}, std={np.std(post_repair_before_open):.2f}")
    print(f"  After:  mean={np.mean(post_repair_after_open):.2f}, std={np.std(post_repair_after_open):.2f}")
    print(f"  Change: mean={np.mean(post_repair_open_changes):.2f}, std={np.std(post_repair_open_changes):.2f}")

if len(normal_before_close) > 0:
    print(f"\nNORMAL CLEANINGS - CLOSE FFT STD:")
    print(f"  Before: mean={np.mean(normal_before_close):.2f}, std={np.std(normal_before_close):.2f}")
    print(f"  After:  mean={np.mean(normal_after_close):.2f}, std={np.std(normal_after_close):.2f}")
    print(f"  Change: mean={np.mean(normal_close_changes):.2f}, std={np.std(normal_close_changes):.2f}")

if len(post_repair_before_close) > 0:
    print(f"\nPOST-REPAIR CLEANINGS - CLOSE FFT STD:")
    print(f"  Before: mean={np.mean(post_repair_before_close):.2f}, std={np.std(post_repair_before_close):.2f}")
    print(f"  After:  mean={np.mean(post_repair_after_close):.2f}, std={np.std(post_repair_after_close):.2f}")
    print(f"  Change: mean={np.mean(post_repair_close_changes):.2f}, std={np.std(post_repair_close_changes):.2f}")

# ============================================================================
# SAVE SUMMARY STATISTICS TO CSV
# ============================================================================

print("\n" + "=" * 80)
print("SAVING SUMMARY STATISTICS")
print("=" * 80)

summary_data = []
for analysis in cleaning_analyses:
    event_summary = {
        'event_num': analysis['event_num'],
        'start_idx': analysis['start_idx'],
        'end_idx': analysis['end_idx'],
        'start_time': analysis['start_time'],
        'end_time': analysis['end_time'],
        'is_post_repair': analysis['is_post_repair'],
        'hours_since_repair': analysis['hours_since_repair'],
        'before_window_size': len(analysis['before_indices']),
        'after_window_size': len(analysis['after_indices'])
    }
    
    if 'open_mean_change' in analysis:
        event_summary['open_mean_abs_change'] = np.mean(np.abs(analysis['open_mean_change']))
        event_summary['open_max_abs_change'] = np.max(np.abs(analysis['open_mean_change']))
    
    if 'close_mean_change' in analysis:
        event_summary['close_mean_abs_change'] = np.mean(np.abs(analysis['close_mean_change']))
        event_summary['close_max_abs_change'] = np.max(np.abs(analysis['close_mean_change']))
    
    # Add FFT STD statistics
    if 'before_open_fft_std_mean' in analysis:
        event_summary['open_fft_std_before'] = analysis['before_open_fft_std_mean']
        event_summary['open_fft_std_after'] = analysis['after_open_fft_std_mean']
        event_summary['open_fft_std_change'] = analysis['open_fft_std_change']
        event_summary['open_fft_std_change_pct'] = analysis['open_fft_std_change_pct']
    
    if 'before_close_fft_std_mean' in analysis:
        event_summary['close_fft_std_before'] = analysis['before_close_fft_std_mean']
        event_summary['close_fft_std_after'] = analysis['after_close_fft_std_mean']
        event_summary['close_fft_std_change'] = analysis['close_fft_std_change']
        event_summary['close_fft_std_change_pct'] = analysis['close_fft_std_change_pct']
    
    summary_data.append(event_summary)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('fft_cleaning_analysis_summary.csv', index=False)
print("\n✓ Saved: fft_cleaning_analysis_summary.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print(f"  - fft_std_time_series_all_cleanings.png (time series of all cleaning windows)")
print(f"  - fft_std_normal_vs_post_repair_comparison.png (comparison plot)")
print(f"  - fft_before_after_cleaning_aggregate.png (aggregate analysis)")
print("  - fft_cleaning_analysis_summary.csv (summary statistics with FFT STD)")
print("\n" + "=" * 80)

