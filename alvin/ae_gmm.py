"""
Cleaning Detection Analysis using GMM Clustering + Autoencoder

This script uses a combined approach to detect cleaning events in valve operation data:
1. Gaussian Mixture Model (GMM) clustering with probability-weighted scoring
2. Autoencoder trained on cleaning patterns to detect anomalies

FEATURES USED:
- FFT DIFFERENCES (not absolute magnitudes):
  * Open-Close differences (within cycle)
  * Temporal differences (cycle-to-cycle changes)
  * Baseline deviations (difference from rolling median)
- Scalar features: std, PSD, spectral entropy, bandwidth, flatness, timing

HYPERPARAMETERS (matching optimization script):
- Window Size: 41 (rows before/after cleaning events)
- N Clusters: 7
- Clustering Method: GMM with full covariance
- Autoencoder Weight: 0.3 (30% autoencoder, 70% GMM)
- Threshold Percentile: 10
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
from sklearn.manifold import TSNE
from tensorflow import keras
from keras import layers
import ast
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def preprocess_data(df, use_differences=True):
    """
    Convert string representations of arrays to actual arrays and flatten all features.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with FFT data
    use_differences : bool
        If True, use FFT differences instead of absolute magnitudes.
        Computes: open-close differences, temporal differences, and baseline deviations.
    
    Returns:
    --------
    X : array
        Feature matrix
    feature_names : list
        List of feature names
    """
    # Parse array columns
    array_columns = ['open_raw_fft_normalized', 'close_raw_fft_normalized']
    
    for col in array_columns:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Convert to numpy arrays
    open_fft_arr = np.array(df['open_raw_fft_normalized'].tolist())
    close_fft_arr = np.array(df['close_raw_fft_normalized'].tolist())
    
    # Flatten array features
    features_list = []
    feature_names = []
    
    if use_differences:
        print("\n" + "="*80)
        print("COMPUTING FFT DIFFERENCES (not absolute magnitudes)")
        print("="*80)
        
        # 1. Open-Close Differences (within same cycle)
        print("\n1. Computing Open-Close differences (within cycle)...")
        open_close_diff = open_fft_arr - close_fft_arr
        for i in range(open_close_diff.shape[1]):
            features_list.append(open_close_diff[:, i])
            feature_names.append(f'open_close_diff_{i}')
        print(f"   Added {open_close_diff.shape[1]} open-close difference features")
        
        # 2. Temporal Differences (cycle-to-cycle changes)
        print("\n2. Computing temporal differences (cycle-to-cycle)...")
        # Open FFT temporal difference
        open_temporal_diff = np.zeros_like(open_fft_arr)
        open_temporal_diff[1:] = open_fft_arr[1:] - open_fft_arr[:-1]
        for i in range(open_temporal_diff.shape[1]):
            features_list.append(open_temporal_diff[:, i])
            feature_names.append(f'open_temporal_diff_{i}')
        
        # Close FFT temporal difference
        close_temporal_diff = np.zeros_like(close_fft_arr)
        close_temporal_diff[1:] = close_fft_arr[1:] - close_fft_arr[:-1]
        for i in range(close_temporal_diff.shape[1]):
            features_list.append(close_temporal_diff[:, i])
            feature_names.append(f'close_temporal_diff_{i}')
        print(f"   Added {open_temporal_diff.shape[1] + close_temporal_diff.shape[1]} temporal difference features")
        
        # 3. Difference from Baseline (deviation from normal operation)
        print("\n3. Computing baseline deviations...")
        # Calculate baseline as rolling median (window=100 cycles)
        # Use median to be robust to outliers/cleaning events
        window_size = 100
        
        # Open baseline deviation
        open_baseline = np.zeros_like(open_fft_arr)
        for i in range(open_fft_arr.shape[1]):
            open_baseline[:, i] = pd.Series(open_fft_arr[:, i]).rolling(
                window=window_size, center=True, min_periods=10
            ).median().bfill().ffill().values
        open_baseline_diff = open_fft_arr - open_baseline
        
        for i in range(open_baseline_diff.shape[1]):
            features_list.append(open_baseline_diff[:, i])
            feature_names.append(f'open_baseline_diff_{i}')
        
        # Close baseline deviation
        close_baseline = np.zeros_like(close_fft_arr)
        for i in range(close_fft_arr.shape[1]):
            close_baseline[:, i] = pd.Series(close_fft_arr[:, i]).rolling(
                window=window_size, center=True, min_periods=10
            ).median().bfill().ffill().values
        close_baseline_diff = close_fft_arr - close_baseline
        
        for i in range(close_baseline_diff.shape[1]):
            features_list.append(close_baseline_diff[:, i])
            feature_names.append(f'close_baseline_diff_{i}')
        print(f"   Added {open_baseline_diff.shape[1] + close_baseline_diff.shape[1]} baseline deviation features")
        
        print(f"\nTotal FFT difference features: {len(feature_names)}")
        
    else:
        # Original: use absolute magnitudes
        print("\nUsing absolute FFT magnitudes (not differences)")
        for col in array_columns:
            arr = np.array(df[col].tolist())
            for i in range(arr.shape[1]):
                features_list.append(arr[:, i])
                feature_names.append(f'{col}_{i}')
    
    # Add scalar features (excluding time/categorical columns)
    scalar_features = [
        'open_fft_std', 'open_psd_mean', 'open_spectral_entropy',
        'open_spectral_bandwidth', 'open_spectral_flatness',
        'close_fft_std', 'close_psd_mean', 'close_spectral_entropy',
        'close_spectral_bandwidth', 'close_spectral_flatness',
        'cycle_time', 'cycle_count', 'normalized_open_time',
        'normalized_close_time', 'minutes_since_run_start'
    ]
    
    print(f"\nAdding scalar features...")
    for col in scalar_features:
        if col in df.columns:
            features_list.append(df[col].values)
            feature_names.append(col)
    print(f"   Added {len([c for c in scalar_features if c in df.columns])} scalar features")
    
    # Combine all features
    X = np.column_stack(features_list)
    
    print(f"\nTotal features: {X.shape[1]}")
    
    return X, feature_names


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


def perform_gmm_clustering(X_data, n_clusters_range=range(3, 11), random_state=42):
    """
    Perform GMM clustering and find optimal number of clusters.
    """
    print("\n" + "="*80)
    print("PERFORMING GMM CLUSTERING ANALYSIS")
    print("="*80)
    
    bic_scores = []
    aic_scores = []
    silhouette_scores = []
    davies_bouldin_scores = []
    
    print(f"\nTesting cluster counts from {min(n_clusters_range)} to {max(n_clusters_range)}...")
    
    for n_clusters in n_clusters_range:
        gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = gmm.fit_predict(X_data)
        
        bic_scores.append(gmm.bic(X_data))
        aic_scores.append(gmm.aic(X_data))
        sil_score = silhouette_score(X_data, cluster_labels)
        db_score = davies_bouldin_score(X_data, cluster_labels)
        
        silhouette_scores.append(sil_score)
        davies_bouldin_scores.append(db_score)
        
        print(f"  n_clusters={n_clusters}: BIC={gmm.bic(X_data):.2f}, AIC={gmm.aic(X_data):.2f}, "
              f"Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}")
    
    # Plot clustering metrics
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    axes[0, 0].plot(n_clusters_range, bic_scores, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Clusters', fontsize=12)
    axes[0, 0].set_ylabel('BIC', fontsize=12)
    axes[0, 0].set_title('BIC Score (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(n_clusters_range, aic_scores, 'mo-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Clusters', fontsize=12)
    axes[0, 1].set_ylabel('AIC', fontsize=12)
    axes[0, 1].set_title('AIC Score (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(n_clusters_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of Clusters', fontsize=12)
    axes[1, 0].set_ylabel('Silhouette Score', fontsize=12)
    axes[1, 0].set_title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(n_clusters_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of Clusters', fontsize=12)
    axes[1, 1].set_ylabel('Davies-Bouldin Score', fontsize=12)
    axes[1, 1].set_title('Davies-Bouldin Score (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gmm_cluster_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal number of clusters based on silhouette score
    optimal_idx = np.argmax(silhouette_scores)
    optimal_clusters = list(n_clusters_range)[optimal_idx]
    
    print(f"\n✓ Optimal number of clusters (by Silhouette): {optimal_clusters}")
    print(f"  Silhouette Score: {silhouette_scores[optimal_idx]:.4f}")
    print(f"  Davies-Bouldin Score: {davies_bouldin_scores[optimal_idx]:.4f}")
    print(f"  BIC: {bic_scores[optimal_idx]:.2f}")
    print(f"  AIC: {aic_scores[optimal_idx]:.2f}")
    
    return optimal_clusters


def fit_gmm_and_analyze(X_all_scaled, X_cleaning_scaled, df, window_indices, n_clusters=7, random_state=42):
    """
    Fit GMM on all data and analyze cluster distributions.
    Uses probability-weighted approach for better performance.
    """
    print("\n" + "="*80)
    print(f"FITTING GMM WITH {n_clusters} CLUSTERS")
    print("="*80)
    
    # Fit GMM on all data with full covariance (matches optimization script)
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, covariance_type='full')
    gmm.fit(X_all_scaled)
    all_cluster_labels = gmm.predict(X_all_scaled)
    gmm_probs = gmm.predict_proba(X_all_scaled)
    
    # Get cluster labels for cleaning window samples
    cleaning_cluster_labels = all_cluster_labels[window_indices]
    
    # Add cluster labels to dataframe
    df['cluster'] = all_cluster_labels
    
    # Analyze cluster distribution
    print(f"\nOverall Cluster Distribution:")
    cluster_counts = pd.Series(all_cluster_labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(all_cluster_labels)) * 100
        print(f"  Cluster {cluster_id}: {count} samples ({percentage:.2f}%)")
    
    print(f"\nCleaning Window Cluster Distribution ({len(window_indices)} samples):")
    cleaning_cluster_counts = pd.Series(cleaning_cluster_labels).value_counts().sort_index()
    for cluster_id, count in cleaning_cluster_counts.items():
        percentage = (count / len(cleaning_cluster_labels)) * 100
        print(f"  Cluster {cluster_id}: {count} samples ({percentage:.2f}%)")
    
    # Calculate enrichment scores for each cluster
    cleaning_cluster_freq = cleaning_cluster_counts / len(cleaning_cluster_labels)
    overall_cluster_freq = cluster_counts / len(all_cluster_labels)
    
    cleaning_enrichment = cleaning_cluster_freq / overall_cluster_freq
    cleaning_enrichment = cleaning_enrichment.fillna(0)
    
    print(f"\nCluster Enrichment in Cleaning Windows:")
    print("(Ratio > 1 means cluster is over-represented in cleaning patterns)")
    for cluster_id in sorted(cleaning_enrichment.index):
        enrichment = cleaning_enrichment[cluster_id]
        print(f"  Cluster {cluster_id}: {enrichment:.2f}x enrichment")
    
    # Define cleaning-associated clusters (enrichment > 1.2)
    cleaning_clusters = cleaning_enrichment[cleaning_enrichment > 1.2].index.tolist()
    print(f"\n✓ Cleaning-associated clusters (enrichment > 1.2x): {cleaning_clusters}")
    
    # Calculate probability-weighted cluster scores (matches optimization script)
    print(f"\nCalculating probability-weighted cluster scores...")
    cluster_probabilities = np.array([
        np.sum(gmm_probs[i] * [cleaning_enrichment.get(j, 0.3) for j in range(n_clusters)])
        for i in range(len(gmm_probs))
    ])
    
    # Normalize to [0, 1] then scale to [0.3, 1.0]
    cluster_probabilities = (cluster_probabilities - cluster_probabilities.min()) / (cluster_probabilities.max() - cluster_probabilities.min())
    cluster_probabilities = cluster_probabilities * 0.7 + 0.3
    
    print(f"  Cluster probability range: [{cluster_probabilities.min():.4f}, {cluster_probabilities.max():.4f}]")
    print(f"  Mean cluster probability: {cluster_probabilities.mean():.4f}")
    
    # Visualize cluster distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Overall cluster distribution
    axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('Cluster ID', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Overall Cluster Distribution (All Data)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Cleaning window cluster distribution
    axes[0, 1].bar(cleaning_cluster_counts.index, cleaning_cluster_counts.values, 
                   color='coral', alpha=0.7)
    axes[0, 1].set_xlabel('Cluster ID', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title('Cluster Distribution (Cleaning Windows)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cluster enrichment
    colors = ['green' if e > 1.2 else 'gray' for e in cleaning_enrichment.values]
    axes[1, 0].bar(cleaning_enrichment.index, cleaning_enrichment.values, color=colors, alpha=0.7)
    axes[1, 0].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    axes[1, 0].axhline(y=1.2, color='orange', linestyle='--', linewidth=2, label='Threshold (1.2x)')
    axes[1, 0].set_xlabel('Cluster ID', fontsize=12)
    axes[1, 0].set_ylabel('Enrichment Ratio', fontsize=12)
    axes[1, 0].set_title('Cluster Enrichment in Cleaning Windows', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cluster timeline
    axes[1, 1].scatter(df.index, all_cluster_labels, c=all_cluster_labels, 
                      cmap='tab10', alpha=0.5, s=20)
    cleaning_window_mask = np.zeros(len(df), dtype=bool)
    cleaning_window_mask[window_indices] = True
    axes[1, 1].scatter(df.index[cleaning_window_mask], 
                      all_cluster_labels[cleaning_window_mask],
                      color='red', s=80, marker='o', 
                      edgecolors='black', linewidths=1.5,
                      label='Cleaning Windows', zorder=5)
    axes[1, 1].set_xlabel('Sample Index', fontsize=12)
    axes[1, 1].set_ylabel('Cluster ID', fontsize=12)
    axes[1, 1].set_title('Cluster Assignment Over Time', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gmm_cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return gmm, all_cluster_labels, cleaning_clusters, cluster_probabilities


def visualize_clusters_2d(X_scaled, cluster_labels, cleaning_indices, cleaning_clusters, method='both'):
    """
    Visualize clusters in 2D using PCA and/or t-SNE dimensionality reduction.
    
    Parameters:
    -----------
    X_scaled : array-like
        Scaled feature data
    cluster_labels : array-like
        Cluster assignments for each sample
    cleaning_indices : array-like
        Indices of samples from cleaning windows
    cleaning_clusters : list
        List of cluster IDs associated with cleaning
    method : str
        'pca', 'tsne', or 'both'
    """
    print("\n" + "="*80)
    print("VISUALIZING CLUSTERS IN 2D")
    print("="*80)
    
    # Create mask for cleaning window samples
    cleaning_mask = np.zeros(len(X_scaled), dtype=bool)
    cleaning_mask[cleaning_indices] = True
    
    n_clusters = len(np.unique(cluster_labels))
    
    if method in ['pca', 'both']:
        print("\nPerforming PCA dimensionality reduction...")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        explained_var = pca.explained_variance_ratio_
        print(f"PCA explained variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}, Total={sum(explained_var):.2%}")
    
    if method in ['tsne', 'both']:
        print("\nPerforming t-SNE dimensionality reduction (this may take a moment)...")
        # Use PCA for initialization if dataset is large
        if X_scaled.shape[0] > 10000:
            pca_init = PCA(n_components=50, random_state=42)
            X_init = pca_init.fit_transform(X_scaled)
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, init='pca')
            X_tsne = tsne.fit_transform(X_init)
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            X_tsne = tsne.fit_transform(X_scaled)
        print("t-SNE complete!")
    
    # Set up figure
    if method == 'both':
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        axes = axes.flatten()
    elif method in ['pca', 'tsne']:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Color map for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot index
    plot_idx = 0
    
    # PCA plots
    if method in ['pca', 'both']:
        # Plot 1: PCA - All clusters
        ax = axes[plot_idx] if method == 'both' else axes[0]
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            is_cleaning_cluster = cluster_id in cleaning_clusters
            marker = 'o' if is_cleaning_cluster else 's'
            alpha = 0.7 if is_cleaning_cluster else 0.4
            label = f'Cluster {cluster_id}' + (' (Cleaning)' if is_cleaning_cluster else '')
            
            ax.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
                      c=[colors[cluster_id]], marker=marker, s=30, alpha=alpha, 
                      label=label, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
        ax.set_title('PCA: Cluster Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
        
        # Plot 2: PCA - Cleaning windows highlighted
        ax = axes[plot_idx] if method == 'both' else axes[1]
        # Plot all points in gray
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c='lightgray', s=20, alpha=0.3, label='Normal Operation')
        # Highlight cleaning window samples
        for cluster_id in range(n_clusters):
            cluster_cleaning_mask = cleaning_mask & (cluster_labels == cluster_id)
            if np.any(cluster_cleaning_mask):
                ax.scatter(X_pca[cluster_cleaning_mask, 0], X_pca[cluster_cleaning_mask, 1],
                          c=[colors[cluster_id]], s=80, alpha=0.8, 
                          label=f'Cleaning Window - Cluster {cluster_id}',
                          edgecolors='black', linewidths=1.5, marker='o')
        
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
        ax.set_title('PCA: Cleaning Windows Highlighted', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # t-SNE plots
    if method in ['tsne', 'both']:
        # Plot 3: t-SNE - All clusters
        ax = axes[plot_idx] if method == 'both' else axes[0]
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            is_cleaning_cluster = cluster_id in cleaning_clusters
            marker = 'o' if is_cleaning_cluster else 's'
            alpha = 0.7 if is_cleaning_cluster else 0.4
            label = f'Cluster {cluster_id}' + (' (Cleaning)' if is_cleaning_cluster else '')
            
            ax.scatter(X_tsne[cluster_mask, 0], X_tsne[cluster_mask, 1], 
                      c=[colors[cluster_id]], marker=marker, s=30, alpha=alpha, 
                      label=label, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('t-SNE: Cluster Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
        
        # Plot 4: t-SNE - Cleaning windows highlighted
        ax = axes[plot_idx] if method == 'both' else axes[1]
        # Plot all points in gray
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c='lightgray', s=20, alpha=0.3, label='Normal Operation')
        # Highlight cleaning window samples
        for cluster_id in range(n_clusters):
            cluster_cleaning_mask = cleaning_mask & (cluster_labels == cluster_id)
            if np.any(cluster_cleaning_mask):
                ax.scatter(X_tsne[cluster_cleaning_mask, 0], X_tsne[cluster_cleaning_mask, 1],
                          c=[colors[cluster_id]], s=80, alpha=0.8, 
                          label=f'Cleaning Window - Cluster {cluster_id}',
                          edgecolors='black', linewidths=1.5, marker='o')
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('t-SNE: Cleaning Windows Highlighted', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('cluster_visualization_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Cluster visualization saved to 'cluster_visualization_2d.png'")
    
    # Return transformed data for potential further use
    result = {}
    if method in ['pca', 'both']:
        result['pca'] = {'transformed': X_pca, 'model': pca}
    if method in ['tsne', 'both']:
        result['tsne'] = {'transformed': X_tsne}
    
    return result


def calculate_feature_importance(autoencoder, X_scaled, y_true, feature_names):
    """
    Calculate feature importance based on autoencoder reconstruction errors
    and correlation with predictions.
    """
    print("\n" + "="*80)
    print("CALCULATING FEATURE IMPORTANCE")
    print("="*80)
    
    # Get predictions
    X_reconstructed = autoencoder.predict(X_scaled, verbose=0)
    
    # Calculate per-feature reconstruction errors
    feature_errors = np.mean(np.abs(X_scaled - X_reconstructed), axis=0)
    
    # Calculate per-feature variance (how much each feature varies)
    feature_variance = np.var(X_scaled, axis=0)
    
    # Calculate correlation between each feature and true labels
    feature_correlations = np.array([
        np.abs(np.corrcoef(X_scaled[:, i], y_true)[0, 1])
        for i in range(X_scaled.shape[1])
    ])
    
    # Handle NaN correlations (constant features)
    feature_correlations = np.nan_to_num(feature_correlations, nan=0.0)
    
    # Get encoder weights (first layer)
    encoder_weights = autoencoder.layers[1].get_weights()[0]  # First Dense layer
    feature_weight_importance = np.mean(np.abs(encoder_weights), axis=1)
    
    # Combine metrics (normalized)
    def normalize(arr):
        if arr.max() == 0:
            return arr
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    normalized_errors = normalize(feature_errors)
    normalized_variance = normalize(feature_variance)
    normalized_correlations = normalize(feature_correlations)
    normalized_weights = normalize(feature_weight_importance)
    
    # Weighted combination: 
    # - 40% reconstruction error (high error = important for reconstruction)
    # - 30% correlation with labels (high correlation = predictive)
    # - 20% encoder weights (high weights = important for encoding)
    # - 10% variance (high variance = more information)
    combined_importance = (
        0.4 * normalized_errors + 
        0.3 * normalized_correlations + 
        0.2 * normalized_weights +
        0.1 * normalized_variance
    )
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'combined_importance': combined_importance,
        'reconstruction_error': feature_errors,
        'correlation_with_label': feature_correlations,
        'encoder_weight_magnitude': feature_weight_importance,
        'variance': feature_variance
    })
    
    # Sort by combined importance
    importance_df = importance_df.sort_values('combined_importance', ascending=False)
    
    print(f"\nTop 20 Most Influential Features:")
    print("="*80)
    for idx, row in importance_df.head(20).iterrows():
        print(f"{row['feature']:50s} | Importance: {row['combined_importance']:.4f} | "
              f"Error: {row['reconstruction_error']:.4f} | "
              f"Corr: {row['correlation_with_label']:.4f}")
    
    # Visualize top features
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    top_n = 30
    top_features = importance_df.head(top_n)
    
    # Plot 1: Combined importance
    axes[0, 0].barh(range(top_n), top_features['combined_importance'].values, color='steelblue', alpha=0.7)
    axes[0, 0].set_yticks(range(top_n))
    axes[0, 0].set_yticklabels(top_features['feature'].values, fontsize=8)
    axes[0, 0].set_xlabel('Combined Importance Score', fontsize=12)
    axes[0, 0].set_title(f'Top {top_n} Most Influential Features', fontsize=14, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Reconstruction error
    axes[0, 1].barh(range(top_n), top_features['reconstruction_error'].values, color='coral', alpha=0.7)
    axes[0, 1].set_yticks(range(top_n))
    axes[0, 1].set_yticklabels(top_features['feature'].values, fontsize=8)
    axes[0, 1].set_xlabel('Mean Absolute Reconstruction Error', fontsize=12)
    axes[0, 1].set_title('Reconstruction Error by Feature', fontsize=14, fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Correlation with labels
    axes[1, 0].barh(range(top_n), top_features['correlation_with_label'].values, color='green', alpha=0.7)
    axes[1, 0].set_yticks(range(top_n))
    axes[1, 0].set_yticklabels(top_features['feature'].values, fontsize=8)
    axes[1, 0].set_xlabel('Absolute Correlation with Cleaning Label', fontsize=12)
    axes[1, 0].set_title('Feature Correlation with Labels', fontsize=14, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Feature importance breakdown (stacked bar for top 10)
    top_10 = importance_df.head(10)
    metrics = ['reconstruction_error', 'correlation_with_label', 'encoder_weight_magnitude', 'variance']
    metric_labels = ['Reconstruction Error', 'Correlation', 'Encoder Weights', 'Variance']
    
    # Normalize each metric for stacking
    normalized_data = np.column_stack([
        normalize(top_10['reconstruction_error'].values),
        normalize(top_10['correlation_with_label'].values),
        normalize(top_10['encoder_weight_magnitude'].values),
        normalize(top_10['variance'].values)
    ])
    
    x = np.arange(len(top_10))
    width = 0.6
    bottom = np.zeros(len(top_10))
    
    colors = ['coral', 'green', 'steelblue', 'orange']
    for i, (metric_data, label, color) in enumerate(zip(normalized_data.T, metric_labels, colors)):
        axes[1, 1].bar(x, metric_data, width, bottom=bottom, label=label, color=color, alpha=0.8)
        bottom += metric_data
    
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(top_10['feature'].values, rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel('Normalized Contribution', fontsize=12)
    axes[1, 1].set_title('Top 10 Features: Importance Breakdown', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Feature importance visualization saved to 'feature_importance_analysis.png'")
    
    return importance_df


def visualize_fft_patterns(df, window_indices, cleaning_event_info):
    """
    Visualize FFT patterns from the cleaning windows using mean and variance.
    """
    print("\n" + "="*80)
    print("COMPUTING FFT STATISTICS FROM CLEANING WINDOWS")
    print("="*80)
    
    # Parse FFT columns for visualization
    df_viz = df.iloc[window_indices].copy()
    for col in ['open_raw_fft_normalized', 'close_raw_fft_normalized']:
        if col in df_viz.columns:
            df_viz[col] = df_viz[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Extract FFT arrays for both open and close
    has_open = 'open_raw_fft_normalized' in df_viz.columns
    has_close = 'close_raw_fft_normalized' in df_viz.columns
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    if has_open:
        open_ffts = np.array(df_viz['open_raw_fft_normalized'].tolist())
        
        # Compute statistics: mean and standard deviation
        mean_open_fft = np.mean(open_ffts, axis=0)
        std_open_fft = np.std(open_ffts, axis=0)
        median_open_fft = np.median(open_ffts, axis=0)
        
        # Compute variance and coefficient of variation
        var_open_fft = np.var(open_ffts, axis=0)
        cv_open_fft = std_open_fft / (mean_open_fft + 1e-10)  # Coefficient of variation
        
        print(f"\nOPEN FFT Statistics from {len(open_ffts)} samples:")
        print(f"  Mean range: [{mean_open_fft.min():.4f}, {mean_open_fft.max():.4f}]")
        print(f"  Std range: [{std_open_fft.min():.4f}, {std_open_fft.max():.4f}]")
        print(f"  Variance range: [{var_open_fft.min():.4f}, {var_open_fft.max():.4f}]")
        print(f"  Coefficient of Variation range: [{cv_open_fft.min():.4f}, {cv_open_fft.max():.4f}]")
        
        # Plot 1: Mean ± 1 Standard Deviation (Open)
        freq_bins = np.arange(len(mean_open_fft))
        axes[0, 0].plot(freq_bins, mean_open_fft, 'b-', label='Mean', linewidth=2)
        axes[0, 0].fill_between(freq_bins, 
                                mean_open_fft - std_open_fft, 
                                mean_open_fft + std_open_fft, 
                                alpha=0.3, label='±1 Std Dev')
        axes[0, 0].plot(freq_bins, median_open_fft, 'g--', label='Median', alpha=0.7)
        axes[0, 0].set_title('Open FFT: Mean ± Standard Deviation', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('FFT Frequency Bin')
        axes[0, 0].set_ylabel('Normalized Magnitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Variance and Coefficient of Variation (Open)
        ax2_twin = axes[0, 1].twinx()
        axes[0, 1].plot(freq_bins, var_open_fft, 'r-', label='Variance', linewidth=2)
        ax2_twin.plot(freq_bins, cv_open_fft, 'orange', label='Coeff. of Variation', linewidth=2)
        axes[0, 1].set_title('Open FFT: Variance & Coefficient of Variation', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('FFT Frequency Bin')
        axes[0, 1].set_ylabel('Variance', color='r')
        ax2_twin.set_ylabel('Coefficient of Variation', color='orange')
        axes[0, 1].tick_params(axis='y', labelcolor='r')
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        axes[0, 1].legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)
    
    if has_close:
        close_ffts = np.array(df_viz['close_raw_fft_normalized'].tolist())
        
        # Compute statistics: mean and standard deviation
        mean_close_fft = np.mean(close_ffts, axis=0)
        std_close_fft = np.std(close_ffts, axis=0)
        median_close_fft = np.median(close_ffts, axis=0)
        
        # Compute variance and coefficient of variation
        var_close_fft = np.var(close_ffts, axis=0)
        cv_close_fft = std_close_fft / (mean_close_fft + 1e-10)
        
        print(f"\nCLOSE FFT Statistics from {len(close_ffts)} samples:")
        print(f"  Mean range: [{mean_close_fft.min():.4f}, {mean_close_fft.max():.4f}]")
        print(f"  Std range: [{std_close_fft.min():.4f}, {std_close_fft.max():.4f}]")
        print(f"  Variance range: [{var_close_fft.min():.4f}, {var_close_fft.max():.4f}]")
        print(f"  Coefficient of Variation range: [{cv_close_fft.min():.4f}, {cv_close_fft.max():.4f}]")
        
        # Plot 3: Mean ± 1 Standard Deviation (Close)
        freq_bins = np.arange(len(mean_close_fft))
        axes[1, 0].plot(freq_bins, mean_close_fft, 'b-', label='Mean', linewidth=2)
        axes[1, 0].fill_between(freq_bins, 
                                mean_close_fft - std_close_fft, 
                                mean_close_fft + std_close_fft, 
                                alpha=0.3, label='±1 Std Dev')
        axes[1, 0].plot(freq_bins, median_close_fft, 'g--', label='Median', alpha=0.7)
        axes[1, 0].set_title('Close FFT: Mean ± Standard Deviation', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('FFT Frequency Bin')
        axes[1, 0].set_ylabel('Normalized Magnitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Variance and Coefficient of Variation (Close)
        ax4_twin = axes[1, 1].twinx()
        axes[1, 1].plot(freq_bins, var_close_fft, 'r-', label='Variance', linewidth=2)
        ax4_twin.plot(freq_bins, cv_close_fft, 'orange', label='Coeff. of Variation', linewidth=2)
        axes[1, 1].set_title('Close FFT: Variance & Coefficient of Variation', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('FFT Frequency Bin')
        axes[1, 1].set_ylabel('Variance', color='r')
        ax4_twin.set_ylabel('Coefficient of Variation', color='orange')
        axes[1, 1].tick_params(axis='y', labelcolor='r')
        ax4_twin.tick_params(axis='y', labelcolor='orange')
        axes[1, 1].legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fft_mean_variance_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nFFT statistical analysis saved to 'fft_mean_variance_patterns.png'")
    
    # Return statistics for potential further use
    stats = {}
    if has_open:
        stats['open'] = {
            'mean': mean_open_fft,
            'std': std_open_fft,
            'variance': var_open_fft,
            'median': median_open_fft,
            'cv': cv_open_fft
        }
    if has_close:
        stats['close'] = {
            'mean': mean_close_fft,
            'std': std_close_fft,
            'variance': var_close_fft,
            'median': median_close_fft,
            'cv': cv_close_fft
        }
    
    return stats


# Load your data
print("Loading data...")
df = pd.read_csv('24-119_all_features_with_labels.csv')

# Check if 'label' column exists, otherwise try 'event'
if 'label' not in df.columns:
    if 'event' in df.columns:
        df['label'] = df['event']
    else:
        print("ERROR: No 'label' or 'event' column found!")
        exit(1)

# Create binary label: 1 for cleaning, 0 for normal
df['is_cleaning'] = df['label'].isin(['cleaning_start', 'cleaning_end']).astype(int)

print(f"\nData shape: {df.shape}")
print(f"Cleaning events: {df['is_cleaning'].sum()}")
print(f"Normal operations: {(df['is_cleaning'] == 0).sum()}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# Extract window indices around cleaning events (window_size = 41)
window_indices, cleaning_event_info = extract_cleaning_windows(df, window_size=41)

if window_indices is None:
    print("\nERROR: Could not extract cleaning windows. Exiting.")
    exit(1)

# Preprocess the full dataset
# Set use_differences=True to use FFT differences instead of absolute magnitudes
X_all, feature_names = preprocess_data(df, use_differences=True)
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\nFull dataset shape: {X_all.shape}")
print(f"Number of features: {len(feature_names)}")

# Extract training data from cleaning windows
X_cleaning_patterns = X_all[window_indices]

print(f"\nTraining data from cleaning windows: {X_cleaning_patterns.shape}")
print(f"  Using {len(window_indices)} samples from around {len(cleaning_event_info)} cleaning events")

# Visualize FFT patterns from cleaning windows
fft_stats = visualize_fft_patterns(df, window_indices, cleaning_event_info)

# Normalize the data
print("\n" + "="*80)
print("NORMALIZING DATA")
print("="*80)

scaler = StandardScaler()
X_cleaning_scaled = scaler.fit_transform(X_cleaning_patterns)
X_all_scaled = scaler.transform(X_all)

print(f"\nNormalized cleaning patterns: {X_cleaning_scaled.shape}")
print(f"Normalized full dataset: {X_all_scaled.shape}")

# ============================================================================
# GMM CLUSTERING
# ============================================================================

# Find optimal number of clusters (for analysis only)
optimal_k = perform_gmm_clustering(X_all_scaled, n_clusters_range=range(6, 8))

# Fit GMM with 7 clusters (as specified)
gmm, all_cluster_labels, cleaning_clusters, cluster_probabilities = fit_gmm_and_analyze(
    X_all_scaled, X_cleaning_scaled, df, window_indices, 
    n_clusters=7, random_state=42
)

# Visualize clusters in 2D space
cluster_viz_data = visualize_clusters_2d(
    X_all_scaled, all_cluster_labels, window_indices, 
    cleaning_clusters, method='both'
)

# ============================================================================
# AUTOENCODER WITH CLUSTER-AWARE TRAINING
# ============================================================================

# Split cleaning patterns for training/validation
X_train, X_val = train_test_split(X_cleaning_scaled, test_size=0.2, random_state=42)

print(f"\nTraining set (cleaning patterns): {X_train.shape}")
print(f"Validation set (cleaning patterns): {X_val.shape}")

# Build Autoencoder with specified architecture (encoding_dim=24, hidden_layers=[256, 128])
print("\n" + "="*80)
print("BUILDING AUTOENCODER MODEL")
print("="*80)

input_dim = X_train.shape[1]
encoding_dim = 32
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

# Train the autoencoder on cleaning patterns
print("\n" + "="*80)
print("TRAINING AUTOENCODER ON CLEANING PATTERNS")
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
# COMBINED GMM + AUTOENCODER PREDICTIONS
# ============================================================================

# Calculate reconstruction errors
print("\n" + "="*80)
print("CALCULATING RECONSTRUCTION ERRORS")
print("="*80)

X_train_pred = autoencoder.predict(X_train)
X_all_pred = autoencoder.predict(X_all_scaled)

# Calculate MSE for each sample
train_mse = np.mean(np.square(X_train - X_train_pred), axis=1)
all_mse = np.mean(np.square(X_all_scaled - X_all_pred), axis=1)

# Calculate threshold (10th percentile as specified)
threshold_percentile = 5
threshold = np.percentile(train_mse, threshold_percentile)

print(f"\nReconstruction Error Statistics (on cleaning patterns):")
print(f"Training data - Mean: {train_mse.mean():.6f}, Std: {train_mse.std():.6f}")
print(f"Training data - Median: {np.median(train_mse):.6f}")
print(f"Threshold ({threshold_percentile}th percentile): {threshold:.6f}")

# Convert reconstruction error to probability
def reconstruction_error_to_probability(mse_values, threshold):
    """
    Convert reconstruction error to cleaning probability.
    Lower error = higher probability (similar to cleaning patterns)
    """
    normalized_error = mse_values / threshold
    probabilities = 1 / (1 + np.exp(10 * (normalized_error - 1)))
    return probabilities

# Calculate autoencoder-based probabilities
ae_probabilities = reconstruction_error_to_probability(all_mse, threshold)

# Note: cluster_probabilities already calculated in fit_gmm_and_analyze using probability-weighted approach

# Combine both approaches: weighted average (30% autoencoder, 70% GMM)
autoencoder_weight = 0.3
clustering_weight = 0.7
combined_probabilities = autoencoder_weight * ae_probabilities + clustering_weight * cluster_probabilities

print("\n" + "="*80)
print("COMBINING GMM AND AUTOENCODER PREDICTIONS")
print("="*80)
print(f"\nWeighting: {int(autoencoder_weight*100)}% Autoencoder + {int(clustering_weight*100)}% GMM Clustering")
print(f"Cleaning-associated clusters: {cleaning_clusters}")
print(f"\nProbability Statistics:")
print(f"  Autoencoder probability range: [{ae_probabilities.min():.4f}, {ae_probabilities.max():.4f}]")
print(f"  GMM cluster probability range: [{cluster_probabilities.min():.4f}, {cluster_probabilities.max():.4f}]")
print(f"  Combined probability range: [{combined_probabilities.min():.4f}, {combined_probabilities.max():.4f}]")

# ============================================================================
# APPLY LEARNED TIME-BASED PRIOR FROM LABELED CLEANING EVENTS
# ============================================================================
# Each labeled cleaning event is treated as 100% cleaning probability.
# Points around labeled cleanings follow a normal distribution.
# This creates a data-driven prior for predicting unlabeled data.

print("\n" + "="*80)
print("LEARNING TIME-BASED PRIOR FROM LABELED CLEANING EVENTS")
print("="*80)

# Parse timestamps
df['timestamp'] = pd.to_datetime(df['open_start_time'])

# Find all labeled cleaning events
cleaning_labels = df[df['label'].isin(['cleaning_start', 'cleaning_end'])].copy()
cleaning_timestamps = cleaning_labels['timestamp'].values

print(f"\nFound {len(cleaning_timestamps)} labeled cleaning events to learn from")

# Parameters for the normal distribution around each cleaning event
# sigma_hours controls the spread of the distribution
sigma_hours = 1.5  # Standard deviation in hours (68% of probability within ±1.5 hours)
sigma_seconds = sigma_hours * 3600  # Convert to seconds for timestamp math

def calculate_learned_prior(timestamp, cleaning_timestamps, sigma_seconds):
    """
    Calculate time-based prior using normal distribution around labeled cleanings.
    
    For each sample, the prior is the maximum probability from all labeled
    cleaning events, where probability follows a Gaussian centered on each
    cleaning event.
    
    Parameters:
    - timestamp: The timestamp to calculate prior for
    - cleaning_timestamps: Array of labeled cleaning event timestamps
    - sigma_seconds: Standard deviation of the normal distribution in seconds
    
    Returns:
    - Prior probability (0 to 1.0, where 1.0 = at a labeled cleaning)
    """
    if len(cleaning_timestamps) == 0:
        return 0.5  # Default prior if no labeled events
    
    # Convert timestamp to numpy datetime64 for vectorized operations
    ts = np.datetime64(timestamp)
    
    # Calculate time differences to all cleaning events (in seconds)
    time_diffs = np.abs((cleaning_timestamps - ts).astype('timedelta64[s]').astype(float))
    
    # Calculate Gaussian probability for each cleaning event
    # exp(-0.5 * (x/sigma)^2)
    probabilities = np.exp(-0.5 * (time_diffs / sigma_seconds) ** 2)
    
    # Take the maximum probability (closest cleaning event dominates)
    max_prob = np.max(probabilities)
    
    # Scale to [0.3, 1.0] range so even far points have some prior
    # At cleaning event: 1.0, Far away: approaches 0.3
    scaled_prob = 0.3 + 0.7 * max_prob
    
    return scaled_prob

# Calculate learned prior for each sample
print(f"\nCalculating learned prior with sigma = {sigma_hours} hours...")
time_priors = df['timestamp'].apply(
    lambda ts: calculate_learned_prior(ts, cleaning_timestamps, sigma_seconds)
)

# Set labeled cleaning events to exactly 1.0 prior
labeled_mask = df['label'].isin(['cleaning_start', 'cleaning_end'])
time_priors.loc[labeled_mask] = 1.0

print(f"\nLearned Prior Statistics:")
print(f"  Prior range: [{time_priors.min():.3f}, {time_priors.max():.3f}]")
print(f"  Prior mean: {time_priors.mean():.3f}")
print(f"  Prior std: {time_priors.std():.3f}")
print(f"  Samples with prior > 0.8: {(time_priors > 0.8).sum()}")
print(f"  Samples with prior > 0.9: {(time_priors > 0.9).sum()}")

# Show distribution of cleaning events by hour of day (learned pattern)
cleaning_hours = pd.to_datetime(cleaning_labels['timestamp']).dt.hour
print(f"\nLearned cleaning time pattern (hour of day):")
hour_counts = cleaning_hours.value_counts().sort_index()
for hour, count in hour_counts.items():
    print(f"  {hour:02d}:00 - {count} events")

# Apply learned prior to combined probability
# Multiply raw probabilities by time prior, then normalize
adjusted_probabilities = combined_probabilities * time_priors.values
# Normalize to keep in reasonable range
if adjusted_probabilities.max() > 0:
    adjusted_probabilities = adjusted_probabilities / adjusted_probabilities.max()
# Scale back to 0-1 range
adjusted_probabilities = np.clip(adjusted_probabilities, 0, 1)

print(f"\nAdjusted probability range (after time prior): [{adjusted_probabilities.min():.4f}, {adjusted_probabilities.max():.4f}]")

# Add results to dataframe
# Save RAW probabilities (before time prior adjustment)
df['reconstruction_error'] = all_mse
df['ae_probability'] = ae_probabilities
df['cluster_probability'] = cluster_probabilities
df['combined_probability_raw'] = combined_probabilities  # Raw combined probability
df['time_prior'] = time_priors.values  # Time-based prior
df['cleaning_probability'] = adjusted_probabilities  # Final probability after time prior
df['predicted_cleaning'] = (adjusted_probabilities > 0.7).astype(int)

# Identify most likely cleaning times
df_sorted = df.sort_values('cleaning_probability', ascending=False)

print("\n" + "="*80)
print("TOP 30 MOST LIKELY CLEANING EVENTS (Combined Score)")
print("="*80)

top_results = df_sorted.head(30)[['open_start_time', 'close_end_time', 
                                    'cluster', 'reconstruction_error', 
                                    'ae_probability', 'cluster_probability',
                                    'combined_probability_raw', 'time_prior',
                                    'cleaning_probability', 'label', 'operational_phase']]

for idx, row in top_results.iterrows():
    rank = list(top_results.index).index(idx) + 1
    print(f"\nRank {rank}:")
    print(f"  Time Period: {row['open_start_time']} to {row['close_end_time']}")
    print(f"  Cluster: {row['cluster']}")
    print(f"  Reconstruction Error: {row['reconstruction_error']:.6f}")
    print(f"  Autoencoder Probability: {row['ae_probability']:.2%}")
    print(f"  Cluster Probability: {row['cluster_probability']:.2%}")
    print(f"  Combined Probability (Raw): {row['combined_probability_raw']:.2%}")
    print(f"  Time Prior: {row['time_prior']:.2%}")
    print(f"  Final Probability: {row['cleaning_probability']:.2%}")
    print(f"  Actual Label: {row['label']}")
    print(f"  Operational Phase: {row['operational_phase']}")

# Summary statistics by actual label
print("\n" + "="*80)
print("STATISTICS BY ACTUAL LABEL")
print("="*80)

label_stats = df.groupby('label').agg({
    'reconstruction_error': ['mean', 'std'],
    'ae_probability': ['mean', 'std'],
    'cluster_probability': ['mean', 'std'],
    'combined_probability_raw': ['mean', 'std'],
    'time_prior': ['mean', 'std'],
    'cleaning_probability': ['mean', 'std'],
    'label': 'count'
})
label_stats.columns = ['_'.join(col).strip() for col in label_stats.columns]
print(label_stats)

# Evaluate performance - AUTOENCODER ONLY
print("\n" + "="*80)
print("CLASSIFICATION PERFORMANCE - AUTOENCODER ONLY")
print("="*80)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

ae_predicted = (ae_probabilities > 0.7).astype(int)
print("\nClassification Report (Autoencoder):")
print(classification_report(df['is_cleaning'], ae_predicted, 
                            target_names=['Normal', 'Cleaning']))

print("\nConfusion Matrix (Autoencoder):")
cm_ae = confusion_matrix(df['is_cleaning'], ae_predicted)
print(cm_ae)

auc_ae = roc_auc_score(df['is_cleaning'], ae_probabilities)
print(f"\nROC AUC Score (Autoencoder): {auc_ae:.4f}")

# Evaluate performance - COMBINED
print("\n" + "="*80)
print("CLASSIFICATION PERFORMANCE - COMBINED (GMM + AUTOENCODER)")
print("="*80)

print("\nClassification Report (Combined):")
print(classification_report(df['is_cleaning'], df['predicted_cleaning'], 
                            target_names=['Normal', 'Cleaning']))

print("\nConfusion Matrix (Combined):")
cm_combined = confusion_matrix(df['is_cleaning'], df['predicted_cleaning'])
print(cm_combined)

auc_combined_raw = roc_auc_score(df['is_cleaning'], combined_probabilities)
auc_combined = roc_auc_score(df['is_cleaning'], adjusted_probabilities)
print(f"\nROC AUC Score (Combined Raw): {auc_combined_raw:.4f}")
print(f"\nROC AUC Score (Combined): {auc_combined:.4f}")

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

# Calculate feature importance
feature_importance_df = calculate_feature_importance(
    autoencoder, X_all_scaled, df['is_cleaning'].values, feature_names
)

# Save feature importance to CSV
feature_importance_df.to_csv('feature_importance_rankings.csv', index=False)
print("\n✓ Feature importance saved to 'feature_importance_rankings.csv'")

# Plot ROC curves comparison
fpr_ae, tpr_ae, _ = roc_curve(df['is_cleaning'], ae_probabilities)
fpr_combined_raw, tpr_combined_raw, _ = roc_curve(df['is_cleaning'], combined_probabilities)
fpr_combined, tpr_combined, _ = roc_curve(df['is_cleaning'], adjusted_probabilities)

plt.figure(figsize=(10, 6))
plt.plot(fpr_ae, tpr_ae, label=f'Autoencoder Only (AUC = {auc_ae:.4f})', linewidth=2)
plt.plot(fpr_combined_raw, tpr_combined_raw, label=f'Combined Raw (AUC = {auc_combined_raw:.4f})', linewidth=2, linestyle='--')
plt.plot(fpr_combined, tpr_combined, label=f'Combined + Time Prior (AUC = {auc_combined:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison - Cleaning Detection', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization of results
fig, axes = plt.subplots(4, 1, figsize=(15, 16))

# Plot 1: Reconstruction error over time
ax1 = axes[0]
ax1.plot(df.index, all_mse, alpha=0.7, label='Reconstruction Error')
ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold_percentile}th percentile)')
cleaning_indices = df[df['is_cleaning'] == 1].index
ax1.scatter(cleaning_indices, all_mse[cleaning_indices], color='red', 
           s=100, marker='x', label='Actual Cleaning Events', zorder=5)
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Reconstruction Error')
ax1.set_title('Reconstruction Error Over Time (Lower = More Similar to Cleaning)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Autoencoder probability
ax2 = axes[1]
ax2.plot(df.index, ae_probabilities, alpha=0.7, color='blue', label='Autoencoder Probability')
ax2.axhline(y=0.7, color='r', linestyle='--', label='Threshold (0.7)')
ax2.scatter(cleaning_indices, ae_probabilities[cleaning_indices], 
           color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Probability')
ax2.set_title('Autoencoder Cleaning Probability Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Cluster assignments with probability overlay
ax3 = axes[2]
scatter = ax3.scatter(df.index, all_cluster_labels, c=cluster_probabilities, 
                     cmap='RdYlGn', alpha=0.6, s=30, vmin=0, vmax=1)
ax3.scatter(cleaning_indices, all_cluster_labels[cleaning_indices], 
           color='red', s=100, marker='o', edgecolors='black', 
           linewidths=2, label='Actual Cleaning Events', zorder=5)
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Cluster ID')
ax3.set_title('Cluster Assignments with Cleaning Probability (Color)')
plt.colorbar(scatter, ax=ax3, label='Cluster Probability')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Combined probability (raw and adjusted)
ax4 = axes[3]
ax4.plot(df.index, combined_probabilities, alpha=0.5, color='purple', label='Combined Raw', linestyle='--')
ax4.plot(df.index, adjusted_probabilities, alpha=0.7, color='purple', label='Combined + Time Prior')
ax4.axhline(y=0.7, color='r', linestyle='--', label='Threshold (0.7)')
ax4.scatter(cleaning_indices, adjusted_probabilities[cleaning_indices], 
           color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
ax4.set_xlabel('Sample Index')
ax4.set_ylabel('Probability')
ax4.set_title('Combined GMM + Autoencoder Cleaning Probability (Raw and Time-Adjusted)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cleaning_detection_results_combined.png', dpi=300, bbox_inches='tight')
plt.show()

# Export results (include raw probabilities)
output_df = df[['open_start_time', 'close_end_time', 'cluster', 
                'reconstruction_error', 'ae_probability', 'cluster_probability',
                'combined_probability_raw', 'time_prior', 'cleaning_probability', 
                'predicted_cleaning', 'label', 'operational_phase', 'cycle_count']].copy()

# Remove duplicates based on timestamps (keep first occurrence)
print("\n" + "="*80)
print("REMOVING DUPLICATE TIMESTAMPS")
print("="*80)
print(f"Total rows before deduplication: {len(output_df)}")

output_df_dedup = output_df.drop_duplicates(subset=['open_start_time', 'close_end_time'], keep='first')
num_duplicates = len(output_df) - len(output_df_dedup)

print(f"Duplicates removed: {num_duplicates}")
print(f"Total rows after deduplication: {len(output_df_dedup)}")

# Save high-probability cleaning events
high_prob_cleaning = output_df_dedup[output_df_dedup['cleaning_probability'] > 0.7].sort_values(
    'cleaning_probability', ascending=False)

print(f"\nProbability columns saved:")
print(f"  - combined_probability_raw: Raw combined probability (GMM + Autoencoder)")
print(f"  - time_prior: Time-based prior from labeled cleaning events")
print(f"  - cleaning_probability: Final probability (raw * time_prior)")

print(f"\n{len(high_prob_cleaning)} unique cycles identified with >70% cleaning probability")

# Save to CSV
output_df_dedup.to_csv('cleaning_predictions_all_combined.csv', index=False)
high_prob_cleaning.to_csv('cleaning_predictions_high_probability_combined.csv', index=False)

print("\nResults saved to:")
print("  CSV Files:")
print("    - cleaning_predictions_all_combined.csv (all predictions, deduplicated)")
print("    - cleaning_predictions_high_probability_combined.csv (>70% probability)")
print("    - feature_importance_rankings.csv (feature influence on predictions)")
print("  Visualizations:")
print("    - gmm_cluster_metrics.png")
print("    - gmm_cluster_analysis.png")
print("    - cluster_visualization_2d.png")
print("    - feature_importance_analysis.png")
print("    - fft_mean_variance_patterns.png")
print("    - training_history.png")
print("    - roc_curve_comparison.png")
print("    - cleaning_detection_results_combined.png")

# Save cleaning event information
if cleaning_event_info:
    cleaning_events_df = pd.DataFrame(cleaning_event_info)
    cleaning_events_df.to_csv('cleaning_events_analyzed.csv', index=False)
    print("  - cleaning_events_analyzed.csv")
    
    print("\n" + "="*80)
    print("CLEANING EVENTS ANALYZED")
    print("="*80)
    print(cleaning_events_df.to_string(index=False))

print("\n" + "="*80)
print("DONE!")
print("="*80)