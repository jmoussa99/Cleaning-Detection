import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, backend as K
import tensorflow as tf
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

# Sampling layer for VAE
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a sample."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Build VAE
def build_vae(input_dim, latent_dim=32, intermediate_dims=[128, 64]):
    """
    Build Variational Autoencoder
    
    Args:
        input_dim: Dimension of input features
        latent_dim: Dimension of latent space
        intermediate_dims: List of dimensions for hidden layers
    
    Returns:
        encoder, decoder, vae models
    """
    
    # ===== ENCODER =====
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = encoder_inputs
    
    # Encoder hidden layers
    for dim in intermediate_dims:
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    
    # Latent space parameters
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    # Sampling
    z = Sampling()([z_mean, z_log_var])
    
    # Encoder model
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    
    # ===== DECODER =====
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = latent_inputs
    
    # Decoder hidden layers (reverse of encoder)
    for dim in reversed(intermediate_dims):
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    
    # Output layer
    decoder_outputs = layers.Dense(input_dim, activation='linear')(x)
    
    # Decoder model
    decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
    
    # ===== VAE =====
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = keras.Model(encoder_inputs, outputs, name='vae')
    
    return encoder, decoder, vae

# VAE Loss function
class VAELoss(keras.losses.Loss):
    """Custom VAE loss combining reconstruction loss and KL divergence"""
    
    def __init__(self, encoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.beta = beta  # Weight for KL divergence term
    
    def call(self, y_true, y_pred):
        # Reconstruction loss (MSE)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(y_true - y_pred), axis=1
            )
        )
        
        # KL divergence
        # We need to get z_mean and z_log_var from encoder
        # This is a bit tricky, so we'll store them as layer attributes
        z_mean = self.encoder.get_layer('z_mean').output
        z_log_var = self.encoder.get_layer('z_log_var').output
        
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )
        
        # Total loss
        total_loss = reconstruction_loss + self.beta * kl_loss
        
        return total_loss

# Custom VAE model with built-in loss
class VAEModel(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(data - reconstruction), axis=1
                )
            )
            
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )
            
            # Total loss
            total_loss = reconstruction_loss + self.beta * kl_loss

        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        data = data[0]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(data - reconstruction), axis=1
            )
        )
        
        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )
        
        # Total loss
        total_loss = reconstruction_loss + self.beta * kl_loss
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

# Calculate VAE-specific anomaly scores
def calculate_vae_anomaly_scores(vae_model, encoder, decoder, X, n_samples=10):
    """
    Calculate anomaly scores using VAE
    
    Returns reconstruction error, KL divergence, and combined anomaly score
    """
    # Get latent representations
    z_mean, z_log_var, _ = encoder.predict(X, verbose=0)
    
    # Reconstruction error (using mean of latent space)
    z_mean_reconstruction = decoder.predict(z_mean, verbose=0)
    reconstruction_error = np.mean(np.square(X - z_mean_reconstruction), axis=1)
    
    # Monte Carlo sampling for better reconstruction estimate
    mc_reconstructions = []
    for _ in range(n_samples):
        # Sample from latent distribution
        epsilon = np.random.normal(0, 1, size=z_mean.shape)
        z_sample = z_mean + np.exp(0.5 * z_log_var) * epsilon
        reconstruction = decoder.predict(z_sample, verbose=0)
        mc_reconstructions.append(reconstruction)
    
    mc_reconstructions = np.array(mc_reconstructions)
    mc_reconstruction_error = np.mean(
        [np.mean(np.square(X - mc_reconstructions[i]), axis=1) 
         for i in range(n_samples)], axis=0
    )
    
    # KL divergence for each sample
    kl_divergence = -0.5 * np.sum(
        1 + z_log_var - np.square(z_mean) - np.exp(z_log_var),
        axis=1
    )
    
    # Combined anomaly score
    # Normalize both components
    reconstruction_normalized = (mc_reconstruction_error - mc_reconstruction_error.mean()) / (mc_reconstruction_error.std() + 1e-8)
    kl_normalized = (kl_divergence - kl_divergence.mean()) / (kl_divergence.std() + 1e-8)
    
    combined_score = reconstruction_normalized + kl_normalized
    
    return {
        'reconstruction_error': reconstruction_error,
        'mc_reconstruction_error': mc_reconstruction_error,
        'kl_divergence': kl_divergence,
        'combined_score': combined_score
    }

# Main execution
print("="*80)
print("VARIATIONAL AUTOENCODER FOR CLEANING EVENT DETECTION")
print("="*80)

# Load your data
df = pd.read_csv('24-121_all_features_labels.csv')

# Create binary label: 1 for cleaning, 0 for normal
df['is_cleaning'] = df['label'].isin(['cleaning_start', 'cleaning_end']).astype(int)

# Preprocess features
X, feature_names = preprocess_data(df)

# Handle NaN and inf values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\nData shape: {X.shape}")
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

# Build VAE
input_dim = X_train.shape[1]
latent_dim = 32
intermediate_dims = [128, 64]
beta = 1.0  # Beta-VAE parameter (higher = more regularization)

encoder, decoder, _ = build_vae(input_dim, latent_dim, intermediate_dims)

print("\nEncoder Architecture:")
encoder.summary()

print("\nDecoder Architecture:")
decoder.summary()

# Create VAE model
vae = VAEModel(encoder, decoder, beta=beta)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

# Train the VAE
print("\n" + "="*80)
print("TRAINING VAE")
print("="*80)

history = vae.fit(
    X_train, X_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_total_loss',
            patience=15,
            restore_best_weights=True,
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_total_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            mode='min'
        )
    ],
    verbose=1
)

# Plot training history
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Total loss
ax1 = axes[0]
ax1.plot(history.history['total_loss'], label='Training')
ax1.plot(history.history['val_total_loss'], label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Loss')
ax1.set_title('Total Loss (Reconstruction + KL)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Reconstruction loss
ax2 = axes[1]
ax2.plot(history.history['reconstruction_loss'], label='Training')
ax2.plot(history.history['val_reconstruction_loss'], label='Validation')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Reconstruction Loss')
ax2.set_title('Reconstruction Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# KL divergence
ax3 = axes[2]
ax3.plot(history.history['kl_loss'], label='Training')
ax3.plot(history.history['val_kl_loss'], label='Validation')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('KL Divergence')
ax3.set_title('KL Divergence')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vae_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate anomaly scores
print("\n" + "="*80)
print("CALCULATING ANOMALY SCORES")
print("="*80)

train_scores = calculate_vae_anomaly_scores(vae, encoder, decoder, X_train, n_samples=10)
all_scores = calculate_vae_anomaly_scores(vae, encoder, decoder, X_all_scaled, n_samples=10)

print("Done!")

# Determine thresholds
threshold_percentile = 97
thresholds = {
    'reconstruction_error': np.percentile(train_scores['reconstruction_error'], threshold_percentile),
    'mc_reconstruction_error': np.percentile(train_scores['mc_reconstruction_error'], threshold_percentile),
    'kl_divergence': np.percentile(train_scores['kl_divergence'], threshold_percentile),
    'combined_score': np.percentile(train_scores['combined_score'], threshold_percentile)
}

print(f"\nAnomaly Score Thresholds ({threshold_percentile}th percentile):")
for score_type, threshold in thresholds.items():
    print(f"  {score_type}: {threshold:.6f}")

# Convert to probabilities
def scores_to_probability(scores, threshold, method='sigmoid'):
    """Convert anomaly scores to probabilities"""
    if method == 'sigmoid':
        normalized = scores / (threshold + 1e-8)
        probabilities = 1 / (1 + np.exp(-5 * (normalized - 1)))
    elif method == 'percentile':
        # Empirical percentile-based probability
        probabilities = np.array([np.mean(train_scores['combined_score'] <= s) 
                                 for s in scores])
    return probabilities

# Calculate probabilities using combined score
cleaning_probabilities = scores_to_probability(
    all_scores['combined_score'], 
    thresholds['combined_score'],
    method='sigmoid'
)

# Alternative: use MC reconstruction error
cleaning_probabilities_recon = scores_to_probability(
    all_scores['mc_reconstruction_error'],
    thresholds['mc_reconstruction_error'],
    method='sigmoid'
)

# Add results to dataframe
df['reconstruction_error'] = all_scores['reconstruction_error']
df['mc_reconstruction_error'] = all_scores['mc_reconstruction_error']
df['kl_divergence'] = all_scores['kl_divergence']
df['combined_score'] = all_scores['combined_score']
df['cleaning_probability'] = cleaning_probabilities
df['cleaning_probability_recon'] = cleaning_probabilities_recon
df['predicted_cleaning'] = (cleaning_probabilities > 0.5).astype(int)

# Identify most likely cleaning times
df_sorted = df.sort_values('cleaning_probability', ascending=False)

print("\n" + "="*80)
print("TOP 20 MOST LIKELY CLEANING EVENTS")
print("="*80)

top_results = df_sorted.head(20)[['open_start_time', 'close_end_time', 
                                    'reconstruction_error', 'mc_reconstruction_error',
                                    'kl_divergence', 'combined_score',
                                    'cleaning_probability', 'label', 'operational_phase']]

for idx, row in top_results.iterrows():
    rank = list(top_results.index).index(idx) + 1
    print(f"\nRank {rank}:")
    print(f"  Time Period: {row['open_start_time']} to {row['close_end_time']}")
    print(f"  Reconstruction Error: {row['reconstruction_error']:.6f}")
    print(f"  MC Reconstruction Error: {row['mc_reconstruction_error']:.6f}")
    print(f"  KL Divergence: {row['kl_divergence']:.6f}")
    print(f"  Combined Anomaly Score: {row['combined_score']:.6f}")
    print(f"  Cleaning Probability: {row['cleaning_probability']:.2%}")
    print(f"  Actual Label: {row['label']}")
    print(f"  Operational Phase: {row['operational_phase']}")

# Summary statistics by actual label
print("\n" + "="*80)
print("STATISTICS BY ACTUAL LABEL")
print("="*80)

label_stats = df.groupby('label').agg({
    'reconstruction_error': ['mean', 'std'],
    'mc_reconstruction_error': ['mean', 'std'],
    'kl_divergence': ['mean', 'std'],
    'combined_score': ['mean', 'std'],
    'cleaning_probability': ['mean', 'std'],
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
    
    print("\nClassification Report (Combined Score):")
    print(classification_report(df['is_cleaning'], df['predicted_cleaning'], 
                                target_names=['Normal', 'Cleaning']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(df['is_cleaning'], df['predicted_cleaning'])
    print(cm)
    
    # ROC AUC for different scores
    auc_scores = {}
    for score_name in ['reconstruction_error', 'mc_reconstruction_error', 
                       'kl_divergence', 'combined_score']:
        auc = roc_auc_score(df['is_cleaning'], 
                           scores_to_probability(all_scores[score_name], 
                                               thresholds[score_name]))
        auc_scores[score_name] = auc
        print(f"\nROC AUC ({score_name}): {auc:.4f}")
    
    # Plot ROC curves for different scores
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, (score_name, auc) in enumerate(auc_scores.items()):
        probs = scores_to_probability(all_scores[score_name], thresholds[score_name])
        fpr, tpr, _ = roc_curve(df['is_cleaning'], probs)
        
        axes[idx].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        axes[idx].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        axes[idx].set_xlabel('False Positive Rate')
        axes[idx].set_ylabel('True Positive Rate')
        axes[idx].set_title(f'ROC Curve - {score_name.replace("_", " ").title()}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vae_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualization of results
fig, axes = plt.subplots(4, 1, figsize=(15, 16))

# Plot 1: Reconstruction error over time
ax1 = axes[0]
ax1.plot(df.index, all_scores['mc_reconstruction_error'], alpha=0.7, 
         label='MC Reconstruction Error', color='blue')
ax1.axhline(y=thresholds['mc_reconstruction_error'], color='r', linestyle='--', 
            label=f'Threshold ({threshold_percentile}th percentile)')
if df['is_cleaning'].sum() > 0:
    cleaning_indices = df[df['is_cleaning'] == 1].index
    ax1.scatter(cleaning_indices, all_scores['mc_reconstruction_error'][cleaning_indices], 
               color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Reconstruction Error')
ax1.set_title('VAE Reconstruction Error Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: KL Divergence over time
ax2 = axes[1]
ax2.plot(df.index, all_scores['kl_divergence'], alpha=0.7, 
         label='KL Divergence', color='green')
ax2.axhline(y=thresholds['kl_divergence'], color='r', linestyle='--', 
            label=f'Threshold ({threshold_percentile}th percentile)')
if df['is_cleaning'].sum() > 0:
    ax2.scatter(cleaning_indices, all_scores['kl_divergence'][cleaning_indices], 
               color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('KL Divergence')
ax2.set_title('VAE KL Divergence Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Cleaning probability over time
ax3 = axes[2]
ax3.plot(df.index, cleaning_probabilities, alpha=0.7, color='orange', 
         label='Cleaning Probability (Combined)')
ax3.plot(df.index, cleaning_probabilities_recon, alpha=0.5, color='purple',
         label='Cleaning Probability (Recon Only)')
ax3.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold (0.5)')
if df['is_cleaning'].sum() > 0:
    ax3.scatter(cleaning_indices, cleaning_probabilities[cleaning_indices], 
               color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Probability')
ax3.set_title('Predicted Cleaning Probability Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Distribution comparison
ax4 = axes[3]
ax4.hist(train_scores['combined_score'], bins=50, alpha=0.5, 
         label='Normal Operations (Training)', density=True, color='blue')
if df['is_cleaning'].sum() > 0:
    cleaning_scores = all_scores['combined_score'][df['is_cleaning'] == 1]
    ax4.hist(cleaning_scores, bins=30, alpha=0.5, 
             label='Actual Cleaning Events', density=True, color='red')
ax4.axvline(x=thresholds['combined_score'], color='darkred', linestyle='--', 
            label='Threshold', linewidth=2)
ax4.set_xlabel('Combined Anomaly Score')
ax4.set_ylabel('Density')
ax4.set_title('Distribution of Combined Anomaly Scores')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vae_cleaning_detection_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize latent space (2D projection if latent_dim > 2)
print("\n" + "="*80)
print("LATENT SPACE VISUALIZATION")
print("="*80)

z_mean, z_log_var, z = encoder.predict(X_all_scaled)

if latent_dim == 2:
    # Direct visualization
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], 
                         c=cleaning_probabilities, 
                         cmap='RdYlBu_r', 
                         alpha=0.6, 
                         s=50)
    
    if df['is_cleaning'].sum() > 0:
        cleaning_mask = df['is_cleaning'] == 1
        plt.scatter(z_mean[cleaning_mask, 0], z_mean[cleaning_mask, 1],
                   c='red', marker='x', s=200, linewidths=3,
                   label='Actual Cleaning Events', zorder=5)
    
    plt.colorbar(scatter, label='Cleaning Probability')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space (Colored by Cleaning Probability)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('vae_latent_space_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    # Use PCA for visualization
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    z_mean_2d = pca.fit_transform(z_mean)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(z_mean_2d[:, 0], z_mean_2d[:, 1],
                         c=cleaning_probabilities,
                         cmap='RdYlBu_r',
                         alpha=0.6,
                         s=50)
    
    if df['is_cleaning'].sum() > 0:
        cleaning_mask = df['is_cleaning'] == 1
        plt.scatter(z_mean_2d[cleaning_mask, 0], z_mean_2d[cleaning_mask, 1],
                   c='red', marker='x', s=200, linewidths=3,
                   label='Actual Cleaning Events', zorder=5)
    
    plt.colorbar(scatter, label='Cleaning Probability')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'VAE Latent Space PCA Projection (Colored by Cleaning Probability)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('vae_latent_space_pca.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot uncertainty (variance) in latent space
latent_std = np.exp(0.5 * z_log_var)
latent_uncertainty = np.mean(latent_std, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Uncertainty over time
ax1 = axes[0]
ax1.plot(df.index, latent_uncertainty, alpha=0.7, color='purple')
if df['is_cleaning'].sum() > 0:
    ax1.scatter(cleaning_indices, latent_uncertainty[cleaning_indices],
               color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Mean Latent Uncertainty')
ax1.set_title('Latent Space Uncertainty Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Uncertainty vs probability
ax2 = axes[1]
ax2.scatter(latent_uncertainty, cleaning_probabilities, alpha=0.5, s=30)
if df['is_cleaning'].sum() > 0:
    cleaning_mask = df['is_cleaning'] == 1
    ax2.scatter(latent_uncertainty[cleaning_mask], 
               cleaning_probabilities[cleaning_mask],
               color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
ax2.set_xlabel('Mean Latent Uncertainty')
ax2.set_ylabel('Cleaning Probability')
ax2.set_title('Relationship: Uncertainty vs Cleaning Probability')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vae_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Export results
output_df = df[['open_start_time', 'close_end_time', 
                'reconstruction_error', 'mc_reconstruction_error',
                'kl_divergence', 'combined_score',
                'cleaning_probability', 'cleaning_probability_recon',
                'predicted_cleaning', 'label', 
                'operational_phase', 'cycle_count']].copy()

# Save high-probability cleaning events
high_prob_cleaning = output_df[output_df['cleaning_probability'] > 0.5].sort_values(
    'cleaning_probability', ascending=False)

print(f"\n{len(high_prob_cleaning)} cycles identified with >50% cleaning probability")

# Save to CSV
output_df.to_csv('vae_cleaning_predictions_all.csv', index=False)
high_prob_cleaning.to_csv('vae_cleaning_predictions_high_probability.csv', index=False)

print("\nResults saved to:")
print("  - vae_cleaning_predictions_all.csv")
print("  - vae_cleaning_predictions_high_probability.csv")

# Save model
vae.save_weights('vae_cleaning_detector_weights.h5')
print("\nModel weights saved to: vae_cleaning_detector_weights.h5")

# Feature importance based on reconstruction error contribution
X_all_pred = vae.predict(X_all_scaled, verbose=0)
feature_errors = np.mean(np.square(X_all_scaled - X_all_pred), axis=0)
feature_importance = pd.DataFrame({
    'feature': feature_names[:len(feature_errors)],
    'avg_squared_error': feature_errors
}).sort_values('avg_squared_error', ascending=False)

print("\n" + "="*80)
print("TOP 15 FEATURES CONTRIBUTING TO RECONSTRUCTION ERROR")
print("="*80)
print(feature_importance.head(15).to_string(index=False))

feature_importance.to_csv('vae_feature_importance.csv', index=False)

# Summary comparison of score types
print("\n" + "="*80)
print("ANOMALY SCORE COMPARISON")
print("="*80)

score_comparison = pd.DataFrame({
    'Score Type': ['Reconstruction', 'MC Reconstruction', 'KL Divergence', 'Combined'],
    'Mean (Normal)': [
        train_scores['reconstruction_error'].mean(),
        train_scores['mc_reconstruction_error'].mean(),
        train_scores['kl_divergence'].mean(),
        train_scores['combined_score'].mean()
    ],
    'Std (Normal)': [
        train_scores['reconstruction_error'].std(),
        train_scores['mc_reconstruction_error'].std(),
        train_scores['kl_divergence'].std(),
        train_scores['combined_score'].std()
    ],
    'Threshold': [
        thresholds['reconstruction_error'],
        thresholds['mc_reconstruction_error'],
        thresholds['kl_divergence'],
        thresholds['combined_score']
    ]
})

if df['is_cleaning'].sum() > 0:
    score_comparison['Mean (Cleaning)'] = [
        all_scores['reconstruction_error'][df['is_cleaning'] == 1].mean(),
        all_scores['mc_reconstruction_error'][df['is_cleaning'] == 1].mean(),
        all_scores['kl_divergence'][df['is_cleaning'] == 1].mean(),
        all_scores['combined_score'][df['is_cleaning'] == 1].mean()
    ]
    score_comparison['Separation'] = (
        score_comparison['Mean (Cleaning)'] - score_comparison['Mean (Normal)']
    ) / score_comparison['Std (Normal)']

print(score_comparison.to_string(index=False))
score_comparison.to_csv('vae_score_comparison.csv', index=False)

print("\n" + "="*80)
print("VAE CLEANING DETECTION COMPLETE!")
print("="*80)