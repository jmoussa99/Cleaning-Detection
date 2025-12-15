"""
Raw Probability Visualization Over Time

Shows the individual GMM and Autoencoder probabilities with peak annotations.

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
from scipy.signal import find_peaks

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
# HELPER: Find and annotate peaks
# ============================================================================
def find_probability_peaks(timestamps, probabilities, threshold=0.6, min_distance=50):
    """Find peaks in probability signal above threshold."""
    peaks, properties = find_peaks(probabilities, height=threshold, distance=min_distance)
    peak_times = timestamps.iloc[peaks]
    peak_values = probabilities.iloc[peaks] if hasattr(probabilities, 'iloc') else probabilities[peaks]
    return peaks, peak_times, peak_values

# ============================================================================
# PLOT: RAW PROBABILITIES (FULL TIME RANGE)
# ============================================================================
fig, ax = plt.subplots(figsize=(20, 8))

# Plot combined raw probability only
ax.plot(df['timestamp'], df['combined_probability_raw'], 
        alpha=0.9, linewidth=2, label='Combined Raw (30% AE + 70% GMM)', color='#9b59b6')

# Find and mark peaks for combined probability
peaks_idx, peak_times, peak_values = find_probability_peaks(
    df['timestamp'], df['combined_probability_raw'], threshold=0.7, min_distance=100
)

# Scatter peaks
ax.scatter(peak_times, peak_values, color='#e74c3c', s=80, zorder=5, 
           marker='v', label=f'Peaks > 0.7 ({len(peaks_idx)} found)')

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
ax.set_title('Raw Probabilities Over Time (Before Time Adjustment)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Set strict axis boundaries
ax.set_xlim(df['timestamp'].min(), df['timestamp'].max())
ax.set_ylim(0, 1.1)

# Format x-axis - dates only
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('raw_probabilities_over_time.png', dpi=300, bbox_inches='tight')

print("\n✓ Saved: raw_probabilities_over_time.png")
print("  → Use toolbar zoom (magnifying glass) or scroll wheel to zoom")

# ============================================================================
# ZOOMED VIEW: Sept 24-27
# ============================================================================
print("\nCreating zoomed view for Sept 24-27...")

# Get Sept 24-27 data
year = df['timestamp'].dt.year.mode()[0]
# Get timezone from dataframe to ensure compatibility
tz = df['timestamp'].dt.tz
start_time = pd.Timestamp(f'{year}-09-24', tz=tz)
end_time = pd.Timestamp(f'{year}-09-27 23:59:59', tz=tz)
df_zoom = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()

if len(df_zoom) > 0:
    fig2, ax2 = plt.subplots(figsize=(20, 8))
    
    # Plot combined raw probability only
    ax2.plot(df_zoom['timestamp'], df_zoom['combined_probability_raw'], 
             alpha=0.9, linewidth=2.5, label='Combined Raw (30% AE + 70% GMM)', color='#9b59b6')
    
    # Find peaks in zoomed window
    peaks_idx_zoom, peak_times_zoom, peak_values_zoom = find_probability_peaks(
        df_zoom['timestamp'].reset_index(drop=True), 
        df_zoom['combined_probability_raw'].reset_index(drop=True), 
        threshold=0.6, min_distance=30
    )
    
    # Scatter and annotate peaks
    if len(peak_times_zoom) > 0:
        ax2.scatter(peak_times_zoom, peak_values_zoom, color='#e74c3c', s=120, zorder=5, 
                   marker='v', label=f'Peaks > 0.6 ({len(peak_times_zoom)} found)')
        
        # Annotate each peak with its time
        for pt, pv in zip(peak_times_zoom, peak_values_zoom):
            time_str = pt.strftime('%m/%d %H:%M')
            ax2.annotate(time_str, (pt, pv), 
                        textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
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
    ax2.set_title('Raw Probabilities (Sept 24-27) with Peak Times', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Set strict axis boundaries
    ax2.set_xlim(start_time, end_time)
    ax2.set_ylim(0, 1.15)
    
    # Format x-axis - dates only
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('raw_probabilities_zoomed.png', dpi=300, bbox_inches='tight')
    
    print("✓ Saved: raw_probabilities_zoomed.png")
    print("  → Use toolbar zoom (magnifying glass) or scroll wheel to zoom")
    
    # Print peak times
    print(f"\n{'='*60}")
    print(f"DETECTED PEAKS IN SEPT 24-27 WINDOW")
    print(f"{'='*60}")
    if len(peak_times_zoom) > 0:
        for i, (pt, pv) in enumerate(zip(peak_times_zoom, peak_values_zoom)):
            print(f"  Peak {i+1}: {pt.strftime('%Y-%m-%d %H:%M:%S')} - Probability: {pv:.3f}")
    else:
        print("  No peaks found above threshold")
        
else:
    print(f"No data found for Sept 24-27, {year}")

# ============================================================================
# SUMMARY: ALL PEAKS
# ============================================================================
print(f"\n{'='*60}")
print(f"ALL HIGH PROBABILITY PEAKS (> 0.7)")
print(f"{'='*60}")

# Get all peaks above 0.7
all_peaks_idx, all_peak_times, all_peak_values = find_probability_peaks(
    df['timestamp'], df['combined_probability_raw'], threshold=0.7, min_distance=100
)

print(f"\nFound {len(all_peaks_idx)} peaks above 0.7 threshold:")
print(f"{'Date/Time':<25} {'Probability':<12} {'AE Prob':<12} {'GMM Prob':<12}")
print("-" * 60)

for idx, pt, pv in zip(all_peaks_idx, all_peak_times, all_peak_values):
    ae_prob = df.iloc[idx]['ae_probability']
    gmm_prob = df.iloc[idx]['cluster_probability']
    print(f"{pt.strftime('%Y-%m-%d %H:%M'):<25} {pv:<12.3f} {ae_prob:<12.3f} {gmm_prob:<12.3f}")

print(f"\n{'='*60}")
print("DONE!")
print(f"{'='*60}")

# Show all plots and keep them interactive
print("\n📊 Plots are now displayed. Use toolbar to zoom/pan.")
print("   Close all plot windows to exit.")
plt.show(block=True)

