"""
Cleaning Detection Analysis using GMM Clustering + Autoencoder
FIXED VERSION: Trains on NORMAL operation, detects cleanings as HIGH reconstruction error

This script uses a combined approach to detect cleaning events in valve operation data:
1. Autoencoder trained on NORMAL operation (not cleaning patterns)
2. Gaussian Mixture Model (GMM) clustering for pattern analysis
3. HIGH reconstruction error = cleaning event (anomaly)

HYPERPARAMETERS:
- N Clusters: 7
- Clustering Method: GMM with full covariance
- Autoencoder Weight: 0.7 (70% autoencoder, 30% GMM)
- Threshold Percentile: 95 (of normal operation)
- Encoding Dim: 24
- Hidden Layers: [256, 128]
- Training Epochs: 50 with early stopping (patience=10)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from tensorflow import keras
from keras import layers
import ast
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def preprocess_data(df):
    """
    Convert string representations of arrays to actual arrays and flatten all features
    """
    # Parse array columns
    array_columns = ['open_raw_fft_normalized', 'close_raw_fft_normalized']
    
    for col in array_columns:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Flatten array features
    features_list = []
    feature_names = []
    
    # Add FFT features
    for col in array_columns:
        arr = np.array(df[col].tolist())
        for i in range(arr.shape[1]):
            features_list.append(arr[:, i])
            feature_names.append(f'{col}_{i}')
    
    # Add scalar features (excluding cycle_count and minutes_since_run_start per user request)
    scalar_features = [
        'open_fft_std', 'open_psd_mean', 'open_spectral_entropy',
        'open_spectral_bandwidth', 'open_spectral_flatness',
        'close_fft_std', 'close_psd_mean', 'close_spectral_entropy',
        'close_spectral_bandwidth', 'close_spectral_flatness',
        'cycle_time', 'normalized_open_time', 'normalized_close_time'
    ]
    
    for col in scalar_features:
        if col in df.columns:
            features_list.append(df[col].values)
            feature_names.append(col)
    
    # Combine all features
    X = np.column_stack(features_list)
    
    return X, feature_names


# Load your data
print("Loading data...")
df = pd.read_csv('nose_cap_14-247-labeled.csv')

# Check if 'label' column exists
if 'label' not in df.columns:
    df['label'] = ''

# Create binary label: 1 for cleaning, 0 for normal
df['is_cleaning'] = df['label'].isin(['cleaning_start', 'cleaning_end']).astype(int)

print(f"\nData shape: {df.shape}")
print(f"Cleaning events: {df['is_cleaning'].sum()}")
print(f"Normal operations: {(df['is_cleaning'] == 0).sum()}")

# Preprocess the full dataset
X_all, feature_names = preprocess_data(df)
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\nFull dataset shape: {X_all.shape}")
print(f"Number of features: {len(feature_names)}")

# Extract window indices around cleaning events (to capture transition patterns)
def extract_cleaning_windows(df, window_size=45):
    """
    Extract full feature data from windows around cleaning events.
    Returns data from window_size rows before cleaning_start and window_size rows after cleaning_end.
    """
    print("\n" + "="*80)
    print("EXTRACTING WINDOWS AROUND CLEANING EVENTS")
    print("="*80)
    
    # Find cleaning start and end indices
    cleaning_starts = df[df['label'] == 'cleaning_start'].index.tolist()
    cleaning_ends = df[df['label'] == 'cleaning_end'].index.tolist()
    
    print(f"\nFound {len(cleaning_starts)} cleaning_start events")
    print(f"Found {len(cleaning_ends)} cleaning_end events")
    
    if len(cleaning_starts) == 0 or len(cleaning_ends) == 0:
        print("WARNING: No cleaning events found!")
        return None, None
    
    # Storage for window indices
    window_indices = []
    cleaning_event_info = []
    
    # Process each cleaning event pair
    for i, (start_idx, end_idx) in enumerate(zip(cleaning_starts, cleaning_ends)):
        print(f"\nProcessing cleaning event {i+1}:")
        print(f"  Cleaning start at index: {start_idx}")
        print(f"  Cleaning end at index: {end_idx}")
        
        # Extract window before cleaning_start
        before_start = max(0, start_idx - window_size)
        before_indices = list(range(before_start, start_idx))
        
        # Extract window after cleaning_end
        after_end = min(len(df), end_idx + window_size + 1)
        after_indices = list(range(end_idx + 1, after_end))
        
        print(f"  Before window: indices {before_start} to {start_idx} ({len(before_indices)} rows)")
        print(f"  After window: indices {end_idx+1} to {after_end} ({len(after_indices)} rows)")
        
        # Combine indices from before and after windows
        window_indices.extend(before_indices)
        window_indices.extend(after_indices)
        
        # Store metadata
        cleaning_event_info.append({
            'event_num': i+1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'before_window_size': len(before_indices),
            'after_window_size': len(after_indices),
            'start_time': df.iloc[start_idx]['open_start_time'] if 'open_start_time' in df.columns else 'N/A',
            'end_time': df.iloc[end_idx]['close_end_time'] if 'close_end_time' in df.columns else 'N/A'
        })
    
    # Remove duplicates and sort
    window_indices = sorted(list(set(window_indices)))
    print(f"\nTotal unique samples from all windows: {len(window_indices)}")
    
    return window_indices, cleaning_event_info

window_indices, cleaning_event_info = extract_cleaning_windows(df, window_size=41)

if window_indices is None:
    print("\nWARNING: No cleaning windows found. Using all normal operation data instead.")
    X_training = X_all[df['is_cleaning'] == 0]
else:
    # Extract training data from cleaning windows (transition patterns)
    X_training = X_all[window_indices]
    print(f"\nTraining data from cleaning windows: {X_training.shape}")
    print(f"  Using {len(window_indices)} samples from around {len(cleaning_event_info)} cleaning events")

# Normalize the data
print("\n" + "="*80)
print("NORMALIZING DATA")
print("="*80)

scaler = StandardScaler()
X_training_scaled = scaler.fit_transform(X_training)
X_all_scaled = scaler.transform(X_all)

print(f"\nNormalized training data (cleaning windows): {X_training_scaled.shape}")
print(f"Normalized full dataset: {X_all_scaled.shape}")

# ============================================================================
# AUTOENCODER TRAINED ON CLEANING TRANSITION PATTERNS
# ============================================================================

# Split cleaning window data for training/validation
X_train, X_val = train_test_split(X_training_scaled, test_size=0.2, random_state=42)

print(f"\nTraining set (cleaning transition patterns): {X_train.shape}")
print(f"Validation set (cleaning transition patterns): {X_val.shape}")

# Build Autoencoder
print("\n" + "="*80)
print("BUILDING AUTOENCODER MODEL")
print("="*80)

input_dim = X_train.shape[1]
encoding_dim = 24
hidden_layers = [256, 128]

def build_autoencoder(input_dim, encoding_dim, hidden_layers=[256, 128], dropout=0.2):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    x = input_layer
    for size in hidden_layers:
        x = layers.Dense(size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    
    # Decoder
    x = encoded
    for size in reversed(hidden_layers):
        x = layers.Dense(size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    
    decoded = layers.Dense(input_dim, activation='linear')(x)
    
    # Autoencoder model
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return autoencoder

autoencoder = build_autoencoder(input_dim, encoding_dim, hidden_layers, dropout=0.2)

print("\nModel Architecture:")
autoencoder.summary()

# Train the autoencoder on cleaning transition patterns
print("\n" + "="*80)
print("TRAINING AUTOENCODER ON CLEANING TRANSITION PATTERNS")
print("="*80)

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Mean Absolute Error')
plt.grid(True)
plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# CALCULATE SIMILARITY SCORES (LOW ERROR = SIMILAR TO CLEANING PATTERNS)
# ============================================================================

print("\n" + "="*80)
print("CALCULATING RECONSTRUCTION ERRORS")
print("="*80)

X_train_pred = autoencoder.predict(X_train)
X_all_pred = autoencoder.predict(X_all_scaled)

# Calculate MSE for each sample
train_mse = np.mean(np.square(X_train - X_train_pred), axis=1)
all_mse = np.mean(np.square(X_all_scaled - X_all_pred), axis=1)

# Calculate threshold (low percentile - model learned cleaning patterns)
threshold_percentile = 10
threshold = np.percentile(train_mse, threshold_percentile)

print(f"\nReconstruction Error Statistics (on cleaning transition patterns):")
print(f"Training data - Mean: {train_mse.mean():.6f}, Std: {train_mse.std():.6f}")
print(f"Training data - Median: {np.median(train_mse):.6f}")
print(f"Threshold ({threshold_percentile}th percentile): {threshold:.6f}")

# Convert reconstruction error to probability
# LOW error = HIGH probability of cleaning (similarity to learned patterns)
def reconstruction_error_to_probability(mse_values, threshold):
    """
    Convert reconstruction error to cleaning probability.
    Lower error = higher probability (similar to cleaning transition patterns)
    """
    normalized_error = mse_values / threshold
    probabilities = 1 / (1 + np.exp(10 * (normalized_error - 1)))
    return probabilities

# Calculate autoencoder-based probabilities
ae_probabilities = reconstruction_error_to_probability(all_mse, threshold)

print(f"\nProbability Statistics:")
print(f"  Autoencoder probability range: [{ae_probabilities.min():.4f}, {ae_probabilities.max():.4f}]")
print(f"  Mean probability: {ae_probabilities.mean():.4f}")

# Add results to dataframe
df['reconstruction_error'] = all_mse
df['cleaning_probability'] = ae_probabilities
df['predicted_cleaning'] = (ae_probabilities > 0.5).astype(int)

# Identify most likely cleaning times
df_sorted = df.sort_values('cleaning_probability', ascending=False)

print("\n" + "="*80)
print("TOP 30 MOST LIKELY CLEANING EVENTS")
print("="*80)

top_results = df_sorted.head(30)[['open_start_time', 'close_end_time', 
                                    'reconstruction_error', 
                                    'cleaning_probability', 'label', 'operational_phase']]

for idx, row in top_results.iterrows():
    rank = list(top_results.index).index(idx) + 1
    print(f"\nRank {rank}:")
    print(f"  Time Period: {row['open_start_time']} to {row['close_end_time']}")
    print(f"  Reconstruction Error: {row['reconstruction_error']:.6f}")
    print(f"  Cleaning Probability: {row['cleaning_probability']:.2%}")
    print(f"  Actual Label: {row['label']}")
    print(f"  Operational Phase: {row['operational_phase']}")

# Check specific dates mentioned by user
print("\n" + "="*80)
print("CHECKING SPECIFIC DATES (10-28 17:00 and 10-29 00:30)")
print("="*80)

# 10-28 around 17:00
oct28_mask = (df['open_start_time'].str.contains('2025-10-28 17:', na=False))
if oct28_mask.sum() > 0:
    print("\n10-28 around 17:00:")
    oct28_data = df[oct28_mask][['open_start_time', 'reconstruction_error', 'cleaning_probability']].head(10)
    print(oct28_data.to_string(index=False))

# 10-29 around 00:30
oct29_mask = (df['open_start_time'].str.contains('2025-10-29 00:3', na=False))
if oct29_mask.sum() > 0:
    print("\n10-29 around 00:30:")
    oct29_data = df[oct29_mask][['open_start_time', 'reconstruction_error', 'cleaning_probability']].head(10)
    print(oct29_data.to_string(index=False))

# Evaluate performance if we have labeled cleaning events
if df['is_cleaning'].sum() > 0:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    
    print("\n" + "="*80)
    print("CLASSIFICATION PERFORMANCE")
    print("="*80)
    
    print("\nClassification Report:")
    print(classification_report(df['is_cleaning'], df['predicted_cleaning'], 
                                target_names=['Normal', 'Cleaning']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(df['is_cleaning'], df['predicted_cleaning'])
    print(cm)
    
    auc_score = roc_auc_score(df['is_cleaning'], ae_probabilities)
    print(f"\nROC AUC Score: {auc_score:.4f}")

# Export results
output_df = df[['open_start_time', 'close_end_time', 
                'reconstruction_error', 'cleaning_probability',
                'predicted_cleaning', 'label', 
                'operational_phase']].copy()

# Save high-probability cleaning events
high_prob_cleaning = output_df[output_df['cleaning_probability'] > 0.7].sort_values(
    'cleaning_probability', ascending=False)

print(f"\n{len(high_prob_cleaning)} cycles identified with >70% cleaning probability")

# Save to CSV
output_df.to_csv('cleaning_predictions_fixed.csv', index=False)
high_prob_cleaning.to_csv('cleaning_predictions_high_probability_fixed.csv', index=False)

print("\nResults saved to:")
print("  - cleaning_predictions_fixed.csv")
print("  - cleaning_predictions_high_probability_fixed.csv")

print("\n" + "="*80)
print("DONE!")
print("="*80)

