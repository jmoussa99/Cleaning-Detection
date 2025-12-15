import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import ast
import warnings
warnings.filterwarnings('ignore')

"""
Alternative Cleaning Detection using Isolation Forest

Isolation Forest is often better than autoencoders for:
- Severe class imbalance
- Sparse anomalies
- When anomalies are truly different from normal

Key advantages:
- Faster training
- No neural network tuning needed
- Better with high-dimensional data
- More robust to unlabeled anomalies in training
"""

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
    
    # Add scalar features
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
    
    # Combine all features
    X = np.column_stack(features_list)
    
    return X, feature_names

print("="*80)
print("CLEANING DETECTION USING ISOLATION FOREST")
print("="*80)

# Load your data
df = pd.read_csv('24-121_all_features_labels.csv')

# Create binary label
df['is_cleaning'] = df['label'].isin(['cleaning_start', 'cleaning_end']).astype(int)

print(f"\nData Summary:")
print(f"  Total samples: {len(df)}")
print(f"  Cleaning events: {df['is_cleaning'].sum()} ({df['is_cleaning'].sum()/len(df)*100:.2f}%)")
print(f"  Normal operations: {(df['is_cleaning'] == 0).sum()}")

# Preprocess features
X, feature_names = preprocess_data(df)

# Handle NaN and inf values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\nFeatures: {len(feature_names)}")
print(f"Data shape: {X.shape}")

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separate training data (normal only)
X_normal = X_scaled[df['is_cleaning'] == 0]
print(f"\nTraining on {len(X_normal)} normal samples")

print("\n" + "="*80)
print("TRAINING ISOLATION FOREST")
print("="*80)

# Try different contamination rates
contamination_rates = [0.01, 0.02, 0.05]
results = {}

for contamination in contamination_rates:
    print(f"\nTraining with contamination={contamination:.2%}...")
    
    # Train Isolation Forest
    clf = IsolationForest(
        contamination=contamination,  # Expected proportion of outliers
        random_state=42,
        n_estimators=100,
        max_samples='auto',
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1
    )
    
    clf.fit(X_normal)
    
    # Get anomaly scores (more negative = more anomalous)
    anomaly_scores = clf.score_samples(X_scaled)
    
    # Convert to probabilities (higher = more likely to be anomaly/cleaning)
    # Normalize scores to [0, 1] range
    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    cleaning_probabilities = 1 - (anomaly_scores - min_score) / (max_score - min_score)
    
    # Get predictions
    predictions = clf.predict(X_scaled)
    predicted_cleaning = (predictions == -1).astype(int)  # -1 = anomaly
    
    results[contamination] = {
        'model': clf,
        'scores': anomaly_scores,
        'probabilities': cleaning_probabilities,
        'predictions': predicted_cleaning
    }
    
    n_flagged = predicted_cleaning.sum()
    print(f"  Flagged as cleaning: {n_flagged} ({n_flagged/len(df)*100:.2f}%)")
    
    # Evaluate if we have labels
    if df['is_cleaning'].sum() > 0:
        print(f"\n  Performance metrics:")
        cm = confusion_matrix(df['is_cleaning'], predicted_cleaning)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall: {recall:.3f}")
        print(f"    F1-Score: {f1:.3f}")
        print(f"    False Positive Rate: {fpr:.3f}")
        print(f"    True Positives: {tp}")
        print(f"    False Positives: {fp}")

# Choose best contamination rate (default: 0.01)
best_contamination = 0.01
clf = results[best_contamination]['model']
anomaly_scores = results[best_contamination]['scores']
cleaning_probabilities = results[best_contamination]['probabilities']
predicted_cleaning = results[best_contamination]['predictions']

print("\n" + "="*80)
print(f"USING CONTAMINATION RATE: {best_contamination:.2%}")
print("="*80)

# Add results to dataframe
df['anomaly_score'] = anomaly_scores
df['cleaning_probability'] = cleaning_probabilities
df['predicted_cleaning'] = predicted_cleaning

# Get threshold-based predictions at different levels
percentiles = [90, 95, 99]
for p in percentiles:
    threshold = np.percentile(cleaning_probabilities, p)
    df[f'high_confidence_{p}'] = (cleaning_probabilities >= threshold).astype(int)

# Top predictions
df_sorted = df.sort_values('cleaning_probability', ascending=False)

print("\n" + "="*80)
print("TOP 20 MOST LIKELY CLEANING EVENTS")
print("="*80)

top_results = df_sorted.head(20)[['open_start_time', 'close_end_time', 
                                    'anomaly_score', 'cleaning_probability', 
                                    'label', 'operational_phase']]

for idx, row in top_results.iterrows():
    print(f"\nRank {top_results.index.get_loc(idx) + 1}:")
    print(f"  Time Period: {row['open_start_time']} to {row['close_end_time']}")
    print(f"  Anomaly Score: {row['anomaly_score']:.6f}")
    print(f"  Cleaning Probability: {row['cleaning_probability']:.2%}")
    print(f"  Actual Label: {row['label']}")
    print(f"  Operational Phase: {row['operational_phase']}")

# Statistics by label
print("\n" + "="*80)
print("STATISTICS BY ACTUAL LABEL")
print("="*80)

label_stats = df.groupby('label').agg({
    'anomaly_score': ['mean', 'std', 'min', 'max'],
    'cleaning_probability': ['mean', 'std', 'min', 'max'],
    'label': 'count'
})
label_stats.columns = ['_'.join(col).strip() for col in label_stats.columns]
print(label_stats)

# Detailed evaluation
if df['is_cleaning'].sum() > 0:
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION PERFORMANCE")
    print("="*80)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(df['is_cleaning'], predicted_cleaning, 
                                target_names=['Normal', 'Cleaning'],
                                zero_division=0))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(df['is_cleaning'], predicted_cleaning)
    print(cm)
    
    # ROC AUC
    try:
        auc_score = roc_auc_score(df['is_cleaning'], cleaning_probabilities)
        print(f"\nROC AUC Score: {auc_score:.4f}")
        
        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(df['is_cleaning'], cleaning_probabilities)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Isolation Forest')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curve_isolation_forest.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"\n⚠️  Could not calculate ROC AUC: {e}")

# Visualizations
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Anomaly scores over time
ax1 = axes[0]
ax1.plot(df.index, anomaly_scores, alpha=0.7, label='Anomaly Score')
threshold_score = np.percentile(anomaly_scores, 1)  # Bottom 1%
ax1.axhline(y=threshold_score, color='r', linestyle='--', 
           label=f'Anomaly Threshold (1st percentile)', linewidth=2)
if df['is_cleaning'].sum() > 0:
    cleaning_indices = df[df['is_cleaning'] == 1].index
    ax1.scatter(cleaning_indices, anomaly_scores[cleaning_indices], 
               color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Anomaly Score (lower = more anomalous)')
ax1.set_title('Isolation Forest Anomaly Scores Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cleaning probability over time
ax2 = axes[1]
ax2.plot(df.index, cleaning_probabilities, alpha=0.7, color='orange', 
        label='Cleaning Probability')
for p in [90, 95, 99]:
    threshold = np.percentile(cleaning_probabilities, p)
    ax2.axhline(y=threshold, linestyle='--', alpha=0.5, label=f'{p}th percentile')
if df['is_cleaning'].sum() > 0:
    ax2.scatter(cleaning_indices, cleaning_probabilities[cleaning_indices], 
               color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Probability')
ax2.set_title('Predicted Cleaning Probability Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution comparison
ax3 = axes[2]
normal_probs = cleaning_probabilities[df['is_cleaning'] == 0]
ax3.hist(normal_probs, bins=50, alpha=0.5, label='Normal Operations', density=True)
if df['is_cleaning'].sum() > 0:
    cleaning_probs = cleaning_probabilities[df['is_cleaning'] == 1]
    ax3.hist(cleaning_probs, bins=30, alpha=0.5, label='Actual Cleaning Events', 
            density=True, color='red')
for p in [90, 95, 99]:
    threshold = np.percentile(cleaning_probabilities, p)
    ax3.axvline(x=threshold, linestyle='--', alpha=0.5, label=f'{p}th percentile')
ax3.set_xlabel('Cleaning Probability')
ax3.set_ylabel('Density')
ax3.set_title('Distribution of Cleaning Probabilities')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cleaning_detection_isolation_forest.png', dpi=300, bbox_inches='tight')
plt.show()

# Export results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_df = df[['open_start_time', 'close_end_time', 'anomaly_score', 
                'cleaning_probability', 'predicted_cleaning', 'label', 
                'operational_phase', 'cycle_count']].copy()

# Remove duplicates
output_df = output_df.drop_duplicates(subset=['open_start_time'], keep='first')

# Save at different confidence levels
for p in [90, 95, 99]:
    threshold = np.percentile(cleaning_probabilities, p)
    high_conf = output_df[output_df['cleaning_probability'] >= threshold].sort_values(
        'cleaning_probability', ascending=False)
    
    filename = f'cleaning_predictions_isolation_forest_{p}pct.csv'
    high_conf.to_csv(filename, index=False)
    print(f"  {len(high_conf)} predictions saved to: {filename}")

# Save all predictions
output_df.to_csv('cleaning_predictions_isolation_forest_all.csv', index=False)
print(f"  All predictions saved to: cleaning_predictions_isolation_forest_all.csv")

# Feature importance (based on anomaly score contribution)
# Calculate by permutation: how much does score change when feature is shuffled
print("\n" + "="*80)
print("ESTIMATING FEATURE IMPORTANCE")
print("="*80)

# Sample subset for faster computation
sample_size = min(1000, len(X_normal))
X_sample = X_normal[:sample_size]
baseline_scores = clf.score_samples(X_sample)

importances = []
for i, feat_name in enumerate(feature_names):
    X_permuted = X_sample.copy()
    X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
    permuted_scores = clf.score_samples(X_permuted)
    importance = np.abs(permuted_scores - baseline_scores).mean()
    importances.append(importance)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

feature_importance.to_csv('feature_importance_isolation_forest.csv', index=False)

print("\n" + "="*80)
print("COMPARISON WITH CONTAMINATION RATES")
print("="*80)

comparison = pd.DataFrame({
    'contamination_rate': list(results.keys()),
    'n_flagged': [results[c]['predictions'].sum() for c in results.keys()],
    'pct_flagged': [results[c]['predictions'].sum() / len(df) * 100 for c in results.keys()]
})

if df['is_cleaning'].sum() > 0:
    comparison['precision'] = [
        (df['is_cleaning'] & results[c]['predictions']).sum() / results[c]['predictions'].sum()
        if results[c]['predictions'].sum() > 0 else 0
        for c in results.keys()
    ]
    comparison['recall'] = [
        (df['is_cleaning'] & results[c]['predictions']).sum() / df['is_cleaning'].sum()
        for c in results.keys()
    ]

print(comparison.to_string(index=False))

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)