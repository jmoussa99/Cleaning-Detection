import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_vae_outliers(data_path, recon_error_col='reconstruction_error', 
                      kl_div_col='kl_divergence', percentile=97):
    """
    Identify outlier samples in VAE training data
    
    Args:
        data_path: Path to CSV file with VAE metrics
        recon_error_col: Column name for reconstruction error
        kl_div_col: Column name for KL divergence
        percentile: Percentile threshold for outlier detection (default 97th)
    
    Returns:
        DataFrame with outlier samples and their indices
    """
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Calculate thresholds
    recon_threshold = np.percentile(df[recon_error_col], percentile)
    kl_threshold = np.percentile(df[kl_div_col], percentile)
    
    print(f"\n{'='*70}")
    print(f"OUTLIER DETECTION THRESHOLDS ({percentile}th percentile)")
    print(f"{'='*70}")
    print(f"Reconstruction Error Threshold: {recon_threshold:.6f}")
    print(f"KL Divergence Threshold: {kl_threshold:.6e}")
    
    # Find outliers
    recon_outliers = df[df[recon_error_col] > recon_threshold].copy()
    kl_outliers = df[df[kl_div_col] > kl_threshold].copy()
    
    # Combined outliers (either metric exceeds threshold)
    combined_outliers = df[
        (df[recon_error_col] > recon_threshold) | 
        (df[kl_div_col] > kl_threshold)
    ].copy()
    
    print(f"\n{'='*70}")
    print(f"OUTLIER COUNTS")
    print(f"{'='*70}")
    print(f"Reconstruction Error outliers: {len(recon_outliers)} ({len(recon_outliers)/len(df)*100:.2f}%)")
    print(f"KL Divergence outliers: {len(kl_outliers)} ({len(kl_outliers)/len(df)*100:.2f}%)")
    print(f"Combined outliers (union): {len(combined_outliers)} ({len(combined_outliers)/len(df)*100:.2f}%)")
    
    # Show most extreme outliers
    print(f"\n{'='*70}")
    print(f"TOP 10 MOST EXTREME OUTLIERS (by Reconstruction Error)")
    print(f"{'='*70}")
    top_recon = df.nlargest(10, recon_error_col)
    for idx, row in top_recon.iterrows():
        print(f"Sample Index {idx:6d}: Recon Error = {row[recon_error_col]:12.6f}, "
              f"KL Div = {row[kl_div_col]:12.6e}")
    
    print(f"\n{'='*70}")
    print(f"TOP 10 MOST EXTREME OUTLIERS (by KL Divergence)")
    print(f"{'='*70}")
    top_kl = df.nlargest(10, kl_div_col)
    for idx, row in top_kl.iterrows():
        print(f"Sample Index {idx:6d}: Recon Error = {row[recon_error_col]:12.6f}, "
              f"KL Div = {row[kl_div_col]:12.6e}")
    
    # Export outliers to CSV
    output_file = data_path.replace('.csv', '_outliers.csv')
    combined_outliers['original_index'] = combined_outliers.index
    combined_outliers.to_csv(output_file, index=False)
    print(f"\n✅ Exported outliers to: {output_file}")
    
    return combined_outliers


def visualize_outliers(data_path, outlier_indices, recon_error_col='reconstruction_error',
                       kl_div_col='kl_divergence', percentile=97):
    """
    Create visualization showing where outliers occur in the training data
    """
    df = pd.read_csv(data_path)
    
    # Calculate thresholds
    recon_threshold = np.percentile(df[recon_error_col], percentile)
    kl_threshold = np.percentile(df[kl_div_col], percentile)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Reconstruction Error
    ax1 = axes[0]
    ax1.plot(df.index, df[recon_error_col], 'b-', alpha=0.5, linewidth=0.5, label='Reconstruction Error')
    ax1.axhline(y=recon_threshold, color='r', linestyle='--', linewidth=2, 
                label=f'Threshold ({percentile}th percentile)')
    ax1.scatter(outlier_indices, df.loc[outlier_indices, recon_error_col], 
                color='red', s=50, marker='x', label='Outliers', zorder=5)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Reconstruction Error')
    ax1.set_title('VAE Reconstruction Error with Outliers Highlighted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: KL Divergence
    ax2 = axes[1]
    ax2.plot(df.index, df[kl_div_col], 'g-', alpha=0.5, linewidth=0.5, label='KL Divergence')
    ax2.axhline(y=kl_threshold, color='r', linestyle='--', linewidth=2,
                label=f'Threshold ({percentile}th percentile)')
    ax2.scatter(outlier_indices, df.loc[outlier_indices, kl_div_col],
                color='red', s=50, marker='x', label='Outliers', zorder=5)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('VAE KL Divergence with Outliers Highlighted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_plot = data_path.replace('.csv', '_outliers_visualization.png')
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_plot}")
    plt.show()


def get_outlier_indices(data_path, recon_error_col='reconstruction_error',
                        kl_div_col='kl_divergence', percentile=97):
    """
    Get list of outlier sample indices
    """
    df = pd.read_csv(data_path)
    
    recon_threshold = np.percentile(df[recon_error_col], percentile)
    kl_threshold = np.percentile(df[kl_div_col], percentile)
    
    # Get indices of outlier samples
    outlier_mask = (df[recon_error_col] > recon_threshold) | (df[kl_div_col] > kl_threshold)
    outlier_indices = df[outlier_mask].index.tolist()
    
    print(f"\n{'='*70}")
    print(f"OUTLIER INDICES")
    print(f"{'='*70}")
    print(f"Total samples: {len(df)}")
    print(f"Outlier samples: {len(outlier_indices)} ({len(outlier_indices)/len(df)*100:.2f}%)")
    print(f"Clean samples: {len(df) - len(outlier_indices)} ({(len(df)-len(outlier_indices))/len(df)*100:.2f}%)")
    
    # Save outlier indices to file
    output_file = data_path.replace('.csv', '_outlier_indices.txt')
    with open(output_file, 'w') as f:
        for idx in outlier_indices:
            f.write(f"{idx}\n")
    print(f"\n✅ Saved outlier sample indices to: {output_file}")
    
    return outlier_indices


if __name__ == "__main__":
    # Example usage - update with your actual data path
    DATA_PATH = 'vae_cleaning_predictions_all.csv'  # Replace with your CSV file
    
    print("="*70)
    print("VAE TRAINING OUTLIER DETECTION")
    print("="*70)
    
    # Configuration
    PERCENTILE = 97  # Adjust this to be more/less aggressive
    
    # Find outliers
    outliers = find_vae_outliers(
        data_path=DATA_PATH,
        recon_error_col='reconstruction_error',  # Update column names as needed
        kl_div_col='kl_divergence',
        percentile=PERCENTILE
    )
    
    # Get outlier indices
    outlier_indices = get_outlier_indices(
        data_path=DATA_PATH,
        percentile=PERCENTILE
    )
    
    # Visualize outliers
    visualize_outliers(
        data_path=DATA_PATH,
        outlier_indices=outliers.index,
        percentile=PERCENTILE
    )
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("1. Review the outliers CSV to understand what's different about them")
    print("2. Use the outlier_indices.txt file to exclude outliers when retraining VAE")
    print("3. Or investigate the outliers to see if they contain data quality issues")
    print("")
    print("Example: Retrain VAE excluding outlier data:")
    print("  outlier_indices = pd.read_csv('..._outlier_indices.txt', header=None)[0].tolist()")
    print("  clean_data = training_data.drop(outlier_indices)")
    print("  # Then retrain VAE with clean_data")
