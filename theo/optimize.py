"""
Automated Optimization Script for Cleaning Detection
Tries different hyperparameters to maximize ROC-AUC score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from keras import layers
import ast
import warnings
warnings.filterwarnings('ignore')

# Import functions from main script
import sys

# Load and preprocess data (simplified versions)
def preprocess_data(df):
    array_columns = ['open_raw_fft_normalized', 'close_raw_fft_normalized']
    
    for col in array_columns:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    features_list = []
    feature_names = []
    
    for col in array_columns:
        arr = np.array(df[col].tolist())
        for i in range(arr.shape[1]):
            features_list.append(arr[:, i])
            feature_names.append(f'{col}_{i}')
    
    scalar_features = [
        'open_fft_std', 'open_psd_mean', 'open_spectral_entropy',
        'open_spectral_bandwidth', 'open_spectral_flatness',
        'close_fft_std', 'close_psd_mean', 'close_spectral_entropy',
        'close_spectral_bandwidth', 'close_spectral_flatness',
        'cycle_time', 'cycle_count', 'normalized_open_time',
        'normalized_close_time', 'minutes_since_run_start'
    ]
    
    for col in scalar_features:
        if col in df.columns:
            features_list.append(df[col].values)
            feature_names.append(col)
    
    X = np.column_stack(features_list)
    return X, feature_names


def extract_cleaning_windows(df, window_size=50):
    cleaning_starts = df[df['label'] == 'cleaning_start'].index.tolist()
    cleaning_ends = df[df['label'] == 'cleaning_end'].index.tolist()
    
    if len(cleaning_starts) == 0 or len(cleaning_ends) == 0:
        return None
    
    window_indices = []
    
    for start_idx, end_idx in zip(cleaning_starts, cleaning_ends):
        before_start = max(0, start_idx - window_size)
        before_indices = list(range(before_start, start_idx))
        
        after_end = min(len(df), end_idx + window_size + 1)
        after_indices = list(range(end_idx + 1, after_end))
        
        window_indices.extend(before_indices)
        window_indices.extend(after_indices)
    
    window_indices = sorted(list(set(window_indices)))
    return window_indices


def build_autoencoder(input_dim, encoding_dim, hidden_sizes=[128, 64], dropout=0.2):
    input_layer = layers.Input(shape=(input_dim,))
    
    # Encoder
    x = input_layer
    for size in hidden_sizes:
        x = layers.Dense(size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    
    # Decoder
    x = encoded
    for size in reversed(hidden_sizes):
        x = layers.Dense(size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    
    decoded = layers.Dense(input_dim, activation='linear')(x)
    
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return autoencoder


def reconstruction_error_to_probability(mse_values, threshold):
    normalized_error = mse_values / threshold
    probabilities = 1 / (1 + np.exp(10 * (normalized_error - 1)))
    return probabilities


# Load data
print("="*80)
print("OPTIMIZING CLEANING DETECTION HYPERPARAMETERS")
print("="*80)

print("\nLoading data...")
df = pd.read_csv('24-121_all_features_labels.csv')

if 'label' not in df.columns:
    if 'event' in df.columns:
        df['label'] = df['event']
    else:
        print("ERROR: No 'label' or 'event' column found!")
        exit(1)

df['is_cleaning'] = df['label'].isin(['cleaning_start', 'cleaning_end']).astype(int)

print(f"Data shape: {df.shape}")
print(f"Cleaning events: {df['is_cleaning'].sum()}")

# Preprocess
X_all, feature_names = preprocess_data(df)
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Features shape: {X_all.shape}")

# Grid search parameters
param_grid = {
    'window_size': [40, 41, 42, 43, 44, 45],
    'n_clusters': [6, 7, 8],
    'clustering_method': ['gmm'],
    'ae_weight': [0.5, 0.6, 0.7, 0.8],
    'threshold_percentile': [3, 5, 10, 15],
    'encoding_dim': [24, 32, 48, 64],
    'hidden_sizes': [[512, 256], [256, 128]],
}

print("\n" + "="*80)
print("FULL GRID SEARCH - TESTING ALL PARAMETER COMBINATIONS")
print("="*80)

# Calculate total combinations
from itertools import product
total_combinations = 1
for key, values in param_grid.items():
    total_combinations *= len(values)

print(f"Total parameter combinations to test: {total_combinations}")
print(f"This may take a while...\n")

# Store all results
all_results = []
combination_count = 0

# Iterate through all parameter combinations
for window_size in param_grid['window_size']:
    print(f"\n{'='*80}")
    print(f"TESTING window_size={window_size}")
    print('='*80)
    
    window_indices = extract_cleaning_windows(df, window_size=window_size)
    if window_indices is None:
        print(f"  WARNING: No cleaning windows found for window_size={window_size}, skipping...")
        continue
    
    X_cleaning_patterns = X_all[window_indices]
    scaler = StandardScaler()
    X_cleaning_scaled = scaler.fit_transform(X_cleaning_patterns)
    X_all_scaled = scaler.transform(X_all)
    
    for n_clusters in param_grid['n_clusters']:
        print(f"\n  Clustering with n_clusters={n_clusters}...")
        
        # Dictionary to store cluster models and probabilities
        cluster_models = {}
        cluster_probs = {}
        
        for clustering_method in param_grid['clustering_method']:
            if clustering_method == 'kmeans':
                # K-Means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_all_scaled)
                cleaning_labels = labels[window_indices]
                
                # Calculate enrichment
                cleaning_counts = pd.Series(cleaning_labels).value_counts()
                overall_counts = pd.Series(labels).value_counts()
                
                cleaning_freq = cleaning_counts / len(cleaning_labels)
                overall_freq = overall_counts / len(labels)
                enrichment = cleaning_freq / overall_freq
                enrichment = enrichment.fillna(0)
                
                cleaning_clusters = enrichment[enrichment > 1.2].index.tolist()
                
                # Cluster probabilities
                probs = np.array([
                    0.7 if c in cleaning_clusters else 0.3 
                    for c in labels
                ])
                
                cluster_models['kmeans'] = kmeans
                cluster_probs['kmeans'] = probs
                
            elif clustering_method == 'gmm':
                # Gaussian Mixture Model
                gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
                gmm.fit(X_all_scaled)
                labels = gmm.predict(X_all_scaled)
                gmm_probs = gmm.predict_proba(X_all_scaled)
                
                cleaning_labels = labels[window_indices]
                cleaning_counts = pd.Series(cleaning_labels).value_counts()
                overall_counts = pd.Series(labels).value_counts()
                
                cleaning_freq = cleaning_counts / len(cleaning_labels)
                overall_freq = overall_counts / len(labels)
                enrichment = cleaning_freq / overall_freq
                enrichment = enrichment.fillna(0)
                
                # GMM: Use probability-weighted score with enrichment
                probs = np.array([
                    np.sum(gmm_probs[i] * [enrichment.get(j, 0.3) for j in range(n_clusters)])
                    for i in range(len(gmm_probs))
                ])
                probs = (probs - probs.min()) / (probs.max() - probs.min())
                probs = probs * 0.7 + 0.3  # Scale to [0.3, 1.0]
                
                cluster_models['gmm'] = gmm
                cluster_probs['gmm'] = probs
        
        # Train autoencoders for different architectures
        for encoding_dim in param_grid['encoding_dim']:
            for hidden_sizes in param_grid['hidden_sizes']:
                print(f"    Training autoencoder: encoding_dim={encoding_dim}, hidden={hidden_sizes}...")
                
                X_train, X_val = train_test_split(X_cleaning_scaled, test_size=0.2, random_state=42)
                
                autoencoder = build_autoencoder(
                    X_train.shape[1], 
                    encoding_dim=encoding_dim, 
                    hidden_sizes=hidden_sizes,
                    dropout=0.2
                )
                
                history = autoencoder.fit(
                    X_train, X_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, X_val),
                    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                    verbose=0
                )
                
                # Calculate reconstruction errors
                X_train_pred = autoencoder.predict(X_train, verbose=0)
                X_all_pred = autoencoder.predict(X_all_scaled, verbose=0)
                
                train_mse = np.mean(np.square(X_train - X_train_pred), axis=1)
                all_mse = np.mean(np.square(X_all_scaled - X_all_pred), axis=1)
                
                # Test all threshold and weight combinations
                for threshold_pct in param_grid['threshold_percentile']:
                    threshold = np.percentile(train_mse, threshold_pct)
                    ae_prob = reconstruction_error_to_probability(all_mse, threshold)
                    
                    for ae_weight in param_grid['ae_weight']:
                        cluster_weight = 1 - ae_weight
                        
                        for clustering_method in param_grid['clustering_method']:
                            # Combine autoencoder and clustering probabilities
                            combined_prob = ae_weight * ae_prob + cluster_weight * cluster_probs[clustering_method]
                            auc = roc_auc_score(df['is_cleaning'], combined_prob)
                            
                            combination_count += 1
                            
                            all_results.append({
                                'window_size': window_size,
                                'n_clusters': n_clusters,
                                'clustering': clustering_method,
                                'ae_weight': ae_weight,
                                'threshold_pct': threshold_pct,
                                'encoding_dim': encoding_dim,
                                'hidden': str(hidden_sizes),
                                'auc': auc
                            })
                            
                            if combination_count % 10 == 0:
                                print(f"      [{combination_count}/{total_combinations}] "
                                      f"cluster={clustering_method}, AE_w={ae_weight}, "
                                      f"thresh={threshold_pct}, AUC={auc:.4f}")

# Convert to DataFrame and analyze
print(f"\n\nCompleted {combination_count} parameter combinations!")
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('auc', ascending=False)

print("\n" + "="*80)
print("TOP 10 CONFIGURATIONS")
print("="*80)
print(results_df.head(10).to_string(index=False))

# Get best configuration
best = results_df.iloc[0]
print("\n" + "="*80)
print("BEST CONFIGURATION FOUND")
print("="*80)
print(f"Window Size: {best['window_size']}")
print(f"N Clusters: {best['n_clusters']}")
print(f"Clustering Method: {best['clustering']}")
print(f"Autoencoder Weight: {best['ae_weight']}")
print(f"Threshold Percentile: {best['threshold_pct']}")
print(f"Encoding Dim: {best['encoding_dim']}")
print(f"Hidden Layers: {best['hidden']}")
print(f"ROC-AUC: {best['auc']:.4f}")

# Save results
results_df.to_csv('hyperparameter_optimization_results.csv', index=False)
print("\nAll results saved to 'hyperparameter_optimization_results.csv'")

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: AUC by window size
axes[0, 0].boxplot([results_df[results_df['window_size'] == ws]['auc'].values 
                     for ws in sorted(results_df['window_size'].unique())])
axes[0, 0].set_xticklabels(sorted(results_df['window_size'].unique()))
axes[0, 0].set_xlabel('Window Size')
axes[0, 0].set_ylabel('ROC-AUC')
axes[0, 0].set_title('AUC by Window Size')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: AUC by n_clusters
axes[0, 1].boxplot([results_df[results_df['n_clusters'] == nc]['auc'].values 
                     for nc in sorted(results_df['n_clusters'].unique())])
axes[0, 1].set_xticklabels(sorted(results_df['n_clusters'].unique()))
axes[0, 1].set_xlabel('Number of Clusters')
axes[0, 1].set_ylabel('ROC-AUC')
axes[0, 1].set_title('AUC by Number of Clusters')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: AUC by clustering method
clustering_methods = results_df['clustering'].unique()
auc_by_method = [results_df[results_df['clustering'] == m]['auc'].values for m in clustering_methods]
axes[0, 2].boxplot(auc_by_method)
axes[0, 2].set_xticklabels(clustering_methods)
axes[0, 2].set_xlabel('Clustering Method')
axes[0, 2].set_ylabel('ROC-AUC')
axes[0, 2].set_title('AUC by Clustering Method')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: AUC by ae_weight
axes[1, 0].boxplot([results_df[results_df['ae_weight'] == aw]['auc'].values 
                     for aw in sorted(results_df['ae_weight'].unique())])
axes[1, 0].set_xticklabels(sorted(results_df['ae_weight'].unique()))
axes[1, 0].set_xlabel('Autoencoder Weight')
axes[1, 0].set_ylabel('ROC-AUC')
axes[1, 0].set_title('AUC by Autoencoder Weight')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: AUC by threshold percentile
axes[1, 1].boxplot([results_df[results_df['threshold_pct'] == tp]['auc'].values 
                     for tp in sorted(results_df['threshold_pct'].unique())])
axes[1, 1].set_xticklabels(sorted(results_df['threshold_pct'].unique()))
axes[1, 1].set_xlabel('Threshold Percentile')
axes[1, 1].set_ylabel('ROC-AUC')
axes[1, 1].set_title('AUC by Threshold Percentile')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Distribution of all AUC scores
axes[1, 2].hist(results_df['auc'], bins=20, edgecolor='black', alpha=0.7)
axes[1, 2].axvline(best['auc'], color='red', linestyle='--', linewidth=2, label=f'Best: {best["auc"]:.4f}')
axes[1, 2].set_xlabel('ROC-AUC')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Distribution of AUC Scores')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hyperparameter_optimization_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved to 'hyperparameter_optimization_analysis.png'")

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE!")
print("="*80)