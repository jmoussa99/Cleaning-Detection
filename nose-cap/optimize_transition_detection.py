"""
Cleaning Detection Analysis using TRANSITION/CHANGE Detection
KEY INSIGHT: Cleanings show up as CHANGES in vibration patterns, not specific absolute values

This script uses a change-detection approach:
1. Compute DELTA features (changes in FFT std, mean, etc.)
2. Train autoencoder on cleaning transition patterns (before/after changes)
3. Detect cleanings by finding similar transition patterns

HYPERPARAMETERS:
- Window size: 20 cycles before + 20 after (to capture transition)
- Focus on: Rate of change, volatility, deltas
- Encoding Dim: 16
- Hidden Layers: [128, 64]
- Training Epochs: 50 with early stopping (patience=10)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


def compute_transition_features(df):
    """
    Compute features that capture TRANSITIONS and CHANGES, not absolute values.
    This is key: cleanings show up as changes in vibration strength.
    """
    print("\n" + "="*80)
    print("COMPUTING TRANSITION FEATURES")
    print("="*80)
    
    # Base features to track changes in
    base_features = [
        'open_fft_mean', 'open_fft_std', 'open_fft_max',
        'close_fft_mean', 'close_fft_std', 'close_fft_max',
        'open_psd_mean', 'close_psd_mean',
        'open_spectral_centroid', 'close_spectral_centroid',
        'open_spectral_entropy', 'close_spectral_entropy',
        'cycle_time'
    ]
    
    transition_features = []
    feature_names = []
    
    for col in base_features:
        if col not in df.columns:
            continue
            
        # 1. Delta from previous cycle (immediate change)
        delta = df[col].diff().fillna(0)
        transition_features.append(delta.values)
        feature_names.append(f'{col}_delta')
        
        # 2. Rolling volatility (how much it's changing)
        volatility = df[col].rolling(window=5, min_periods=1).std().fillna(0)
        transition_features.append(volatility.values)
        feature_names.append(f'{col}_volatility')
        
        # 3. Percent change (relative change)
        pct_change = df[col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
        transition_features.append(pct_change.values)
        feature_names.append(f'{col}_pct_change')
        
        # 4. Rolling mean difference (trend change)
        rolling_mean = df[col].rolling(window=10, min_periods=1).mean()
        mean_diff = (df[col] - rolling_mean).fillna(0)
        transition_features.append(mean_diff.values)
        feature_names.append(f'{col}_from_rolling_mean')
    
    # Combine all transition features
    X = np.column_stack(transition_features)
    
    print(f"Created {X.shape[1]} transition features from {len(base_features)} base features")
    print(f"Feature types: delta, volatility, pct_change, deviation_from_rolling_mean")
    
    return X, feature_names


# Load data
print("Loading data...")
df = pd.read_csv('nose_cap_14-247-labeled.csv')

# Check if 'label' column exists
if 'label' not in df.columns:
    df['label'] = ''

# Create binary label
df['is_cleaning'] = df['label'].isin(['cleaning_start', 'cleaning_end']).astype(int)

print(f"\nData shape: {df.shape}")
print(f"Cleaning events: {df['is_cleaning'].sum()}")
print(f"Normal operations: {(df['is_cleaning'] == 0).sum()}")

# Compute transition features
X_all, feature_names = compute_transition_features(df)
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\nFull dataset shape: {X_all.shape}")
print(f"Number of transition features: {len(feature_names)}")

# ============================================================================
# EXTRACT WINDOWS AROUND CLEANING EVENTS
# ============================================================================

def extract_cleaning_windows(df, X_all, window_size=20):
    """
    Extract windows around cleaning events to capture the transition pattern.
    Smaller window (20 cycles) to focus on immediate before/after transition.
    """
    print("\n" + "="*80)
    print("EXTRACTING TRANSITION WINDOWS AROUND CLEANING EVENTS")
    print("="*80)
    
    cleaning_starts = df[df['label'] == 'cleaning_start'].index.tolist()
    cleaning_ends = df[df['label'] == 'cleaning_end'].index.tolist()
    
    print(f"\nFound {len(cleaning_starts)} cleaning_start events")
    print(f"Found {len(cleaning_ends)} cleaning_end events")
    
    if len(cleaning_starts) == 0 or len(cleaning_ends) == 0:
        print("WARNING: No cleaning events found!")
        return None
    
    window_indices = []
    
    for i, (start_idx, end_idx) in enumerate(zip(cleaning_starts, cleaning_ends)):
        # Window before cleaning (captures "high vibration strength" state)
        before_start = max(0, start_idx - window_size)
        before_indices = list(range(before_start, start_idx))
        
        # Window after cleaning (captures "low vibration strength" state)
        after_end = min(len(df), end_idx + window_size + 1)
        after_indices = list(range(end_idx + 1, after_end))
        
        print(f"Event {i+1}: Before [{before_start}:{start_idx}], After [{end_idx+1}:{after_end}]")
        
        window_indices.extend(before_indices)
        window_indices.extend(after_indices)
    
    # Remove duplicates and sort
    window_indices = sorted(list(set(window_indices)))
    
    print(f"\nTotal unique window indices: {len(window_indices)}")
    
    return X_all[window_indices]


X_training = extract_cleaning_windows(df, X_all, window_size=20)

if X_training is None:
    print("ERROR: Could not extract cleaning windows!")
    exit(1)

# ============================================================================
# NORMALIZE DATA
# ============================================================================

scaler = StandardScaler()
X_training_scaled = scaler.fit_transform(X_training)
X_all_scaled = scaler.transform(X_all)

print(f"\nNormalized training data (cleaning transition windows): {X_training_scaled.shape}")
print(f"Normalized full dataset: {X_all_scaled.shape}")

# ============================================================================
# BUILD AND TRAIN AUTOENCODER ON TRANSITION PATTERNS
# ============================================================================

X_train, X_val = train_test_split(X_training_scaled, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# Build autoencoder
input_dim = X_train.shape[1]
encoding_dim = 16  # Smaller encoding for transition patterns

print("\n" + "="*80)
print("BUILDING AUTOENCODER FOR TRANSITION DETECTION")
print("="*80)
print(f"Input dim: {input_dim}")
print(f"Encoding dim: {encoding_dim}")
print(f"Architecture: {input_dim} -> 128 -> 64 -> {encoding_dim} -> 64 -> 128 -> {input_dim}")

# Encoder
input_layer = keras.Input(shape=(input_dim,))
encoded = layers.Dense(128, activation='relu')(input_layer)
encoded = layers.BatchNormalization()(encoded)
encoded = layers.Dropout(0.2)(encoded)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.BatchNormalization()(encoded)
encoded = layers.Dropout(0.2)(encoded)
encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(encoded)

# Decoder
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.BatchNormalization()(decoded)
decoded = layers.Dropout(0.2)(decoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.BatchNormalization()(decoded)
decoded = layers.Dropout(0.2)(decoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

# Autoencoder model
autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

print(autoencoder.summary())

# Train
print("\n" + "="*80)
print("TRAINING AUTOENCODER ON CLEANING TRANSITION PATTERNS")
print("="*80)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping],
    verbose=1
)

# ============================================================================
# CALCULATE RECONSTRUCTION ERRORS AND PROBABILITIES
# ============================================================================

print("\n" + "="*80)
print("CALCULATING RECONSTRUCTION ERRORS")
print("="*80)

X_train_pred = autoencoder.predict(X_train)
X_all_pred = autoencoder.predict(X_all_scaled)

# Calculate MSE for each sample
train_mse = np.mean(np.square(X_train - X_train_pred), axis=1)
all_mse = np.mean(np.square(X_all_scaled - X_all_pred), axis=1)

# Use low percentile threshold (low error = similar to training = cleaning transition)
threshold_percentile = 10
threshold = np.percentile(train_mse, threshold_percentile)

print(f"\nReconstruction Error Statistics (on training transitions):") 
print(f"Training data - Mean: {train_mse.mean():.6f}, Std: {train_mse.std():.6f}")
print(f"Training data - Median: {np.median(train_mse):.6f}")
print(f"Threshold ({threshold_percentile}th percentile): {threshold:.6f}")

# Calculate probabilities: LOW error = HIGH probability (similar to cleaning transitions)
ae_probability = np.exp(-all_mse / threshold)
ae_probability = np.clip(ae_probability, 0, 1)

print(f"\nAE Probability Statistics:")
print(f"Mean: {ae_probability.mean():.6f}")
print(f"Median: {np.median(ae_probability):.6f}")
print(f"Max: {ae_probability.max():.6f}")
print(f"Min: {ae_probability.min():.6f}")

# ============================================================================
# GMM CLUSTERING ON TRANSITION PATTERNS
# ============================================================================

print("\n" + "="*80)
print("GMM CLUSTERING ON TRANSITION PATTERNS")
print("="*80)

n_clusters = 5
gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
clusters = gmm.fit_predict(X_all_scaled)

# Find which clusters contain cleaning events
cleaning_indices = df[df['is_cleaning'] == 1].index.tolist()
cleaning_clusters = clusters[cleaning_indices]
cluster_counts = pd.Series(cleaning_clusters).value_counts()

print(f"\nCleaning events by cluster:")
print(cluster_counts)

# Assign higher probability to clusters with more cleaning events
cluster_probabilities = {}
total_cleanings = len(cleaning_indices)
for cluster_id in range(n_clusters):
    count = cluster_counts.get(cluster_id, 0)
    cluster_probabilities[cluster_id] = count / total_cleanings if total_cleanings > 0 else 0.2

print(f"\nCluster probabilities:")
for cluster_id, prob in cluster_probabilities.items():
    print(f"Cluster {cluster_id}: {prob:.3f}")

cluster_probability = np.array([cluster_probabilities[c] for c in clusters])

# ============================================================================
# COMBINE PROBABILITIES
# ============================================================================

print("\n" + "="*80)
print("COMBINING PROBABILITIES")
print("="*80)

# Weighted combination: 70% autoencoder, 30% GMM
ae_weight = 0.7
gmm_weight = 0.3

combined_probability = (ae_weight * ae_probability) + (gmm_weight * cluster_probability)

print(f"Weights: AE={ae_weight}, GMM={gmm_weight}")
print(f"\nCombined Probability Statistics:")
print(f"Mean: {combined_probability.mean():.6f}")
print(f"Median: {np.median(combined_probability):.6f}")
print(f"Max: {combined_probability.max():.6f}")
print(f"Min: {combined_probability.min():.6f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Create results dataframe
results_df = df[['open_start_time', 'close_end_time']].copy()
results_df['cluster'] = clusters
results_df['reconstruction_error'] = all_mse
results_df['ae_probability'] = ae_probability
results_df['cluster_probability'] = cluster_probability
results_df['cleaning_probability'] = combined_probability
results_df['predicted_cleaning'] = (combined_probability > 0.5).astype(int)
results_df['label'] = df['label']

# Determine operational phase
def get_operational_phase(row):
    if row['label'] in ['cleaning_start', 'cleaning_end']:
        return 'cleaning'
    elif row['label'] == 'startup':
        return 'startup'
    elif row['label'] == 'shutdown':
        return 'shutdown'
    else:
        return 'normal_operation'

results_df['operational_phase'] = results_df.apply(get_operational_phase, axis=1)

# Save all predictions
output_file = 'cleaning_predictions_transition_detection.csv'
results_df.to_csv(output_file, index=False)
print(f"Saved all predictions to: {output_file}")

# Save high probability predictions
high_prob_df = results_df[results_df['cleaning_probability'] > 0.5].copy()
high_prob_file = 'cleaning_predictions_high_probability_transition.csv'
high_prob_df.to_csv(high_prob_file, index=False)
print(f"Saved high probability predictions to: {high_prob_file}")
print(f"High probability predictions: {len(high_prob_df)}")

# ============================================================================
# ANALYZE SPECIFIC DATES
# ============================================================================

print("\n" + "="*80)
print("ANALYZING SPECIFIC DATES (10-28 and 10-29)")
print("="*80)

# Convert to datetime
results_df['open_start_time'] = pd.to_datetime(results_df['open_start_time'])

# Check 10-28 around 6pm (18:00)
oct28_6pm = results_df[
    (results_df['open_start_time'] >= '2025-10-28 17:45:00') &
    (results_df['open_start_time'] <= '2025-10-28 18:15:00')
].sort_values('cleaning_probability', ascending=False)

print("\n10-28 around 6pm (cleaning reported at 6pm):")
print("Top 10 highest probabilities:")
print(oct28_6pm[['open_start_time', 'reconstruction_error', 'ae_probability', 
                  'cluster', 'cluster_probability', 'cleaning_probability']].head(10).to_string(index=False))

# Check 10-29 around 1:30am
oct29_1am = results_df[
    (results_df['open_start_time'] >= '2025-10-29 01:00:00') &
    (results_df['open_start_time'] <= '2025-10-29 02:00:00')
].sort_values('cleaning_probability', ascending=False)

print("\n10-29 around 1:30am (cleaning reported at 1:30am):")
print("Top 10 highest probabilities:")
print(oct29_1am[['open_start_time', 'reconstruction_error', 'ae_probability',
                  'cluster', 'cluster_probability', 'cleaning_probability']].head(10).to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)











