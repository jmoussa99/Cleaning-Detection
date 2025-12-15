import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    
    # Add scalar features (excluding time/categorical columns)
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

# Load your data
df = pd.read_csv('24-121_all_features_labels.csv')

# Create binary label: 1 for cleaning, 0 for normal
df['is_cleaning'] = df['label'].isin(['cleaning_start', 'cleaning_end']).astype(int)

# Preprocess features
X, feature_names = preprocess_data(df)

# Handle NaN and inf values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Data shape: {X.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Cleaning events: {df['is_cleaning'].sum()}")
print(f"Normal operations: {(df['is_cleaning'] == 0).sum()}")

# Separate normal and cleaning data
X_normal = X[df['is_cleaning'] == 0]
X_cleaning = X[df['is_cleaning'] == 1]

# Normalize the data
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)
X_all_scaled = scaler.transform(X)

# Split normal data for training
X_train, X_val = train_test_split(X_normal_scaled, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# Build Autoencoder
input_dim = X_train.shape[1]
encoding_dim = 32  # Adjust based on your needs

def build_autoencoder(input_dim, encoding_dim):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    
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
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return autoencoder

# Build and train the model
autoencoder = build_autoencoder(input_dim, encoding_dim)

print("\nModel Architecture:")
autoencoder.summary()

# Train the autoencoder
history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
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

# Calculate reconstruction errors
X_train_pred = autoencoder.predict(X_train)
X_all_pred = autoencoder.predict(X_all_scaled)

# Calculate MSE for each sample
train_mse = np.mean(np.square(X_train - X_train_pred), axis=1)
all_mse = np.mean(np.square(X_all_scaled - X_all_pred), axis=1)

# Calculate threshold (using percentile of normal data)
threshold_percentile = 98
threshold = np.percentile(train_mse, threshold_percentile)

print(f"\nReconstruction Error Statistics:")
print(f"Normal data - Mean: {train_mse.mean():.6f}, Std: {train_mse.std():.6f}")
print(f"Threshold ({threshold_percentile}th percentile): {threshold:.6f}")

# Convert reconstruction error to probability
def reconstruction_error_to_probability(mse_values, threshold):
    """
    Convert reconstruction error to cleaning probability
    Using sigmoid function centered at threshold
    """
    # Normalize by threshold
    normalized_error = mse_values / threshold
    
    # Apply sigmoid transformation
    # Higher error = higher probability of cleaning
    probabilities = 1 / (1 + np.exp(-5 * (normalized_error - 1)))
    
    return probabilities

# Calculate cleaning probabilities
cleaning_probabilities = reconstruction_error_to_probability(all_mse, threshold)

# Add results to dataframe
df['reconstruction_error'] = all_mse
df['cleaning_probability'] = cleaning_probabilities
df['predicted_cleaning'] = (cleaning_probabilities > 0.5).astype(int)

# Identify most likely cleaning times
df_sorted = df.sort_values('cleaning_probability', ascending=False)

print("\n" + "="*80)
print("TOP 20 MOST LIKELY CLEANING EVENTS")
print("="*80)

top_results = df_sorted.head(20)[['open_start_time', 'close_end_time', 
                                    'reconstruction_error', 'cleaning_probability', 
                                    'label', 'operational_phase']]

for idx, row in top_results.iterrows():
    print(f"\nRank {top_results.index.get_loc(idx) + 1}:")
    print(f"  Time Period: {row['open_start_time']} to {row['close_end_time']}")
    print(f"  Reconstruction Error: {row['reconstruction_error']:.6f}")
    print(f"  Cleaning Probability: {row['cleaning_probability']:.2%}")
    print(f"  Actual Label: {row['label']}")
    print(f"  Operational Phase: {row['operational_phase']}")

# Summary statistics by actual label
print("\n" + "="*80)
print("STATISTICS BY ACTUAL LABEL")
print("="*80)

label_stats = df.groupby('label').agg({
    'reconstruction_error': ['mean', 'std', 'min', 'max'],
    'cleaning_probability': ['mean', 'std', 'min', 'max'],
    'label': 'count'
})
label_stats.columns = ['_'.join(col).strip() for col in label_stats.columns]
print(label_stats)

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
    
    # ROC AUC
    auc_score = roc_auc_score(df['is_cleaning'], cleaning_probabilities)
    print(f"\nROC AUC Score: {auc_score:.4f}")
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(df['is_cleaning'], cleaning_probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Cleaning Detection')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualization of results
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Reconstruction error over time
ax1 = axes[0]
ax1.plot(df.index, all_mse, alpha=0.7, label='Reconstruction Error')
ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold_percentile}th percentile)')
if df['is_cleaning'].sum() > 0:
    cleaning_indices = df[df['is_cleaning'] == 1].index
    ax1.scatter(cleaning_indices, all_mse[cleaning_indices], color='red', 
               s=100, marker='x', label='Actual Cleaning Events', zorder=5)
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Reconstruction Error')
ax1.set_title('Reconstruction Error Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cleaning probability over time
ax2 = axes[1]
ax2.plot(df.index, cleaning_probabilities, alpha=0.7, color='orange', label='Cleaning Probability')
ax2.axhline(y=0.95, color='r', linestyle='--', label='Decision Threshold (0.95)')
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
ax3.hist(train_mse, bins=50, alpha=0.5, label='Normal Operations (Training)', density=True)
if df['is_cleaning'].sum() > 0:
    cleaning_errors = all_mse[df['is_cleaning'] == 1]
    ax3.hist(cleaning_errors, bins=30, alpha=0.5, label='Actual Cleaning Events', density=True)
ax3.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
ax3.set_xlabel('Reconstruction Error')
ax3.set_ylabel('Density')
ax3.set_title('Distribution of Reconstruction Errors')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cleaning_detection_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Export results
output_df = df[['open_start_time', 'close_end_time', 'reconstruction_error', 
                'cleaning_probability', 'predicted_cleaning', 'label', 
                'operational_phase', 'cycle_count']].copy()
output_df = output_df[output_df['reconstruction_error'] < 0.31].drop_duplicates(subset=['open_start_time'], keep='first')

# Save high-probability cleaning events
high_prob_cleaning = output_df[output_df['cleaning_probability'] > 0.5].sort_values(
    'cleaning_probability', ascending=False)

print(f"\n{len(high_prob_cleaning)} cycles identified with >95% cleaning probability")

# Save to CSV
output_df.to_csv('cleaning_predictions_all.csv', index=False)
high_prob_cleaning.to_csv('cleaning_predictions_high_probability.csv', index=False)

print("\nResults saved to:")
print("  - cleaning_predictions_all.csv")
print("  - cleaning_predictions_high_probability.csv")

# Feature importance based on reconstruction error contribution
feature_errors = np.mean(np.square(X_all_scaled - X_all_pred), axis=0)
feature_importance = pd.DataFrame({
    'feature': feature_names[:len(feature_errors)],
    'avg_squared_error': feature_errors
}).sort_values('avg_squared_error', ascending=False)

print("\n" + "="*80)
print("TOP 15 FEATURES CONTRIBUTING TO RECONSTRUCTION ERROR")
print("="*80)
print(feature_importance.head(15).to_string(index=False))

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)