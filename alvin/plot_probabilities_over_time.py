"""
Probability Visualization Over Time

This script graphs the GMM + Autoencoder adjusted probabilities over time.

Interactive Controls:
- Use the zoom tool (magnifying glass) in the toolbar to zoom into regions
- Use the pan tool (cross arrows) to move around when zoomed
- Press 'h' or click Home button to reset view
- Scroll wheel also zooms in/out
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Enable interactive mode for zooming/panning
plt.ion()

# Load the predictions data
print("Loading prediction data...")
df = pd.read_csv('cleaning_predictions_all_combined.csv')

# Parse timestamps
df['timestamp'] = pd.to_datetime(df['open_start_time'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Loaded {len(df)} records")
print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Mark actual cleaning events
cleaning_mask = df['label'].isin(['cleaning_start', 'cleaning_end'])
cleaning_times = df.loc[cleaning_mask, 'timestamp'] if cleaning_mask.any() else pd.Series()

# ============================================================================
# PLOT: ADJUSTED PROBABILITIES (FULL TIME RANGE)
# ============================================================================
fig, ax = plt.subplots(figsize=(20, 6))

# Plot time prior and adjusted probability
ax.fill_between(df['timestamp'], df['time_prior'], alpha=0.2, color='#f39c12', label='Time Prior (Schedule)')
ax.plot(df['timestamp'], df['combined_probability_raw'], 
         alpha=0.4, linewidth=1, label='Combined Raw', color='#9b59b6', linestyle='--')
ax.plot(df['timestamp'], df['cleaning_probability'], 
         alpha=0.9, linewidth=2, label='Final Adjusted Probability', color='#e74c3c')

# Mark actual cleaning events
if len(cleaning_times) > 0:
    for ct in cleaning_times:
        ax.axvline(x=ct, color='red', alpha=0.3, linewidth=1.5, linestyle='--')
    ax.axvline(x=cleaning_times.iloc[0], color='red', alpha=0.5, 
                linewidth=2, linestyle='--', label='Actual Cleaning Events')

# Threshold line
ax.axhline(y=0.7, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Detection Threshold (0.7)')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Adjusted Probabilities Over Time (With Time Schedule Prior)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Set strict axis boundaries
ax.set_xlim(df['timestamp'].min(), df['timestamp'].max())
ax.set_ylim(0, 1.55)

# Format x-axis - dates only, no times
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('probabilities_over_time.png', dpi=300, bbox_inches='tight')

print("\n✓ Saved: probabilities_over_time.png")
print("  → Use toolbar zoom (magnifying glass) or scroll wheel to zoom")

# ============================================================================
# ZOOMED VIEW: Sept 20-23
# ============================================================================
print("\nCreating zoomed view for Sept 20-23...")

# Get Sept 20-23 data
year = df['timestamp'].dt.year.mode()[0]  # Get the most common year
# Get timezone from dataframe to ensure compatibility
tz = df['timestamp'].dt.tz
start_time = pd.Timestamp(f'{year}-09-24', tz=tz)
end_time = pd.Timestamp(f'{year}-09-27 23:59:59', tz=tz)
df_zoom = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

if len(df_zoom) > 0:
    fig2, ax2 = plt.subplots(figsize=(20, 6))
    
    # Plot adjusted probabilities
    ax2.fill_between(df_zoom['timestamp'], df_zoom['time_prior'], alpha=0.2, color='#f39c12', label='Time Prior (Schedule)')
    ax2.plot(df_zoom['timestamp'], df_zoom['combined_probability_raw'], 
             alpha=0.4, linewidth=1.5, label='Combined Raw', color='#9b59b6', linestyle='--')
    ax2.plot(df_zoom['timestamp'], df_zoom['cleaning_probability'], 
             alpha=0.9, linewidth=2.5, label='Final Adjusted Probability', color='#e74c3c')
    
    # Mark actual cleaning events in zoom window
    cleaning_mask_zoom = df_zoom['label'].isin(['cleaning_start', 'cleaning_end'])
    if cleaning_mask_zoom.any():
        cleaning_times_zoom = df_zoom.loc[cleaning_mask_zoom, 'timestamp']
        for ct in cleaning_times_zoom:
            ax2.axvline(x=ct, color='red', alpha=0.4, linewidth=2, linestyle='--')
        ax2.axvline(x=cleaning_times_zoom.iloc[0], color='red', alpha=0.6, 
                    linewidth=2, linestyle='--', label='Actual Cleaning Events')
    
    ax2.axhline(y=0.7, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Detection Threshold (0.7)')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Adjusted Probabilities (Sept 20-23)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Set strict axis boundaries
    ax2.set_xlim(start_time, end_time)
    ax2.set_ylim(0, 1.55)
    
    # Format x-axis - dates only
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('probabilities_over_time_zoomed.png', dpi=300, bbox_inches='tight')
    
    print("✓ Saved: probabilities_over_time_zoomed.png")
    print("  → Use toolbar zoom (magnifying glass) or scroll wheel to zoom")
else:
    print(f"No data found for Sept 20-23, {year}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*60)
print("PROBABILITY STATISTICS")
print("="*60)

print("\n--- Raw Probabilities ---")
print(f"Autoencoder:     mean={df['ae_probability'].mean():.3f}, std={df['ae_probability'].std():.3f}")
print(f"GMM Cluster:     mean={df['cluster_probability'].mean():.3f}, std={df['cluster_probability'].std():.3f}")
print(f"Combined Raw:    mean={df['combined_probability_raw'].mean():.3f}, std={df['combined_probability_raw'].std():.3f}")

print("\n--- Adjusted Probabilities ---")
print(f"Time Prior:      mean={df['time_prior'].mean():.3f}, std={df['time_prior'].std():.3f}")
print(f"Final Adjusted:  mean={df['cleaning_probability'].mean():.3f}, std={df['cleaning_probability'].std():.3f}")

print("\n--- Detection Counts (threshold=0.7) ---")
high_prob_raw = (df['combined_probability_raw'] > 0.7).sum()
high_prob_adj = (df['cleaning_probability'] > 0.7).sum()
actual_cleanings = cleaning_mask.sum()
print(f"Raw Combined > 0.7:      {high_prob_raw} samples")
print(f"Adjusted > 0.7:          {high_prob_adj} samples")
print(f"Actual cleaning labels:  {actual_cleanings} samples")

print("\n" + "="*60)
print("DONE!")
print("="*60)

# Show all plots and keep them interactive
print("\n📊 Plots are now displayed. Use toolbar to zoom/pan.")
print("   Close all plot windows to exit.")
plt.show(block=True)

