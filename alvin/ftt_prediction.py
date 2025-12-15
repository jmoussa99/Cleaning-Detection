import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import ast
import warnings
warnings.filterwarnings('ignore')

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess data
def preprocess_data(df):
    """
    Convert string representations of arrays to actual arrays and flatten all features
    """
    # Parse array columns
    array_columns = ['open_raw_fft_normalized', 'close_raw_fft_normalized']
    
    for col in array_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Flatten array features
    features_list = []
    feature_names = []
    feature_types = []  # numerical/categorical
    
    # Add FFT features
    for col in array_columns:
        if col in df.columns:
            arr = np.array(df[col].tolist())
            for i in range(arr.shape[1]):
                features_list.append(arr[:, i])
                feature_names.append(f'{col}_{i}')
                feature_types.append('numerical')
    
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
            feature_types.append('numerical')
    
    # Combine all features
    X = np.column_stack(features_list)
    
    return X, feature_names, feature_types

# FTTransformer Implementation
class FeatureTokenizer(nn.Module):
    """Converts numerical features into embeddings"""
    def __init__(self, n_features, d_token, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias = nn.Parameter(torch.randn(n_features, d_token)) if bias else None
        
    def forward(self, x):
        # x: [batch_size, n_features]
        # output: [batch_size, n_features, d_token]
        x = x.unsqueeze(-1) * self.weight.unsqueeze(0)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)
        return x

class MultiheadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_token, n_heads, dropout=0.1):
        super().__init__()
        assert d_token % n_heads == 0
        
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        self.qkv = nn.Linear(d_token, 3 * d_token)
        self.out = nn.Linear(d_token, d_token)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, n_tokens, d_token = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, n_tokens, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, n_tokens, d_token)
        
        output = self.out(attn_output)
        
        return output, attn_weights.mean(dim=1)  # Return average attention weights

class TransformerBlock(nn.Module):
    """Transformer block with attention and feedforward"""
    def __init__(self, d_token, n_heads, d_ffn, dropout=0.1):
        super().__init__()
        self.attention = MultiheadAttention(d_token, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_token)
        
    def forward(self, x):
        # Attention + residual
        attn_output, attn_weights = self.attention(self.norm1(x))
        x = x + attn_output
        
        # FFN + residual
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x, attn_weights

class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data"""
    def __init__(self, n_features, d_token=96, n_blocks=3, n_heads=8, 
                 d_ffn=256, dropout=0.1, use_reconstruction=True):
        super().__init__()
        self.use_reconstruction = use_reconstruction
        self.n_features = n_features
        
        # Feature tokenizer
        self.feature_tokenizer = FeatureTokenizer(n_features, d_token)
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout)
            for _ in range(n_blocks)
        ])
        
        if use_reconstruction:
            # Reconstruction head (for each feature)
            self.reconstruction_head = nn.Linear(d_token, 1)
        else:
            # Classification head (normal vs anomaly)
            self.classification_head = nn.Sequential(
                nn.Linear(d_token, d_ffn),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ffn, 1)
            )
        
    def forward(self, x, return_attention=False):
        batch_size = x.shape[0]
        
        # Tokenize features
        tokens = self.feature_tokenizer(x)  # [batch, n_features, d_token]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [batch, n_features+1, d_token]
        
        # Pass through transformer blocks
        attention_weights = []
        for block in self.blocks:
            tokens, attn = block(tokens)
            attention_weights.append(attn)
        
        if self.use_reconstruction:
            # Reconstruct original features
            feature_tokens = tokens[:, 1:, :]  # Exclude CLS token
            reconstructed = self.reconstruction_head(feature_tokens).squeeze(-1)
            
            if return_attention:
                return reconstructed, attention_weights
            return reconstructed
        else:
            # Use CLS token for classification
            cls_output = tokens[:, 0, :]
            logits = self.classification_head(cls_output).squeeze(-1)
            
            if return_attention:
                return logits, attention_weights
            return logits

# Custom Dataset
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# Training function for reconstruction
def train_reconstruction_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_X in train_loader:
            if isinstance(batch_X, list):
                batch_X = batch_X[0]
            batch_X = batch_X.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(batch_X)
            loss = F.mse_loss(reconstructed, batch_X)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_X in val_loader:
                if isinstance(batch_X, list):
                    batch_X = batch_X[0]
                batch_X = batch_X.to(device)
                
                reconstructed = model(batch_X)
                loss = F.mse_loss(reconstructed, batch_X)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_fttransformer_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_fttransformer_model.pt'))
    return model, history

# Calculate reconstruction errors and attention anomaly scores
def calculate_anomaly_scores(model, data_loader, original_data):
    model.eval()
    all_reconstruction_errors = []
    all_attention_scores = []
    
    with torch.no_grad():
        for batch_X in data_loader:
            if isinstance(batch_X, list):
                batch_X = batch_X[0]
            batch_X = batch_X.to(device)
            
            # Get reconstruction and attention
            reconstructed, attention_weights = model(batch_X, return_attention=True)
            
            # Reconstruction error
            recon_error = torch.mean((batch_X - reconstructed) ** 2, dim=1)
            all_reconstruction_errors.extend(recon_error.cpu().numpy())
            
            # Attention anomaly score (variance in attention patterns)
            # High variance might indicate unusual feature interactions
            avg_attention = torch.stack(attention_weights).mean(dim=0)
            # Focus on CLS token attention to features
            cls_attention = avg_attention[:, 0, 1:]  # [batch, n_features]
            attention_entropy = -torch.sum(
                cls_attention * torch.log(cls_attention + 1e-10), dim=1
            )
            all_attention_scores.extend(attention_entropy.cpu().numpy())
    
    return np.array(all_reconstruction_errors), np.array(all_attention_scores)

# Main execution
def main():
    # Load your data
    print("Loading data...")
    df = pd.read_csv('nose_cap_14-247-labeled.csv')
    
    # Create binary label
    df['is_cleaning'] = df['label'].isin(['cleaning_start', 'cleaning_end']).astype(int)
    
    # Preprocess features
    print("Preprocessing features...")
    X, feature_names, feature_types = preprocess_data(df)
    
    # Handle NaN and inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Data shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Cleaning events: {df['is_cleaning'].sum()}")
    print(f"Normal operations: {(df['is_cleaning'] == 0).sum()}")
    
    # Separate normal and cleaning data
    X_normal = X[df['is_cleaning'] == 0]
    X_cleaning = X[df['is_cleaning'] == 1] if df['is_cleaning'].sum() > 0 else None
    
    # Normalize the data
    print("Normalizing data...")
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_all_scaled = scaler.transform(X)
    
    # Split normal data for training
    X_train, X_val = train_test_split(X_normal_scaled, test_size=0.2, random_state=42)
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Create data loaders
    train_dataset = TabularDataset(X_train)
    val_dataset = TabularDataset(X_val)
    all_dataset = TabularDataset(X_all_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    all_loader = DataLoader(all_dataset, batch_size=64, shuffle=False)
    
    # Build FTTransformer
    print("\nBuilding FTTransformer model...")
    n_features = X_train.shape[1]
    
    model = FTTransformer(
        n_features=n_features,
        d_token=96,
        n_blocks=3,
        n_heads=8,
        d_ffn=256,
        dropout=0.1,
        use_reconstruction=True
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    print("\nTraining FTTransformer...")
    model, history = train_reconstruction_model(
        model, train_loader, val_loader, epochs=100, lr=1e-3
    )
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('FTTransformer Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fttransformer_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate anomaly scores
    print("\nCalculating anomaly scores...")
    reconstruction_errors, attention_scores = calculate_anomaly_scores(
        model, all_loader, X_all_scaled
    )
    
    # Normalize attention scores (lower entropy = more anomalous)
    attention_anomaly = 1 - (attention_scores - attention_scores.min()) / (
        attention_scores.max() - attention_scores.min() + 1e-10
    )
    
    # Combined anomaly score
    combined_score = 0.7 * reconstruction_errors + 0.3 * attention_anomaly
    
    # Calculate thresholds
    threshold_percentile = 95
    recon_threshold = np.percentile(reconstruction_errors[:len(X_normal)], threshold_percentile)
    combined_threshold = np.percentile(combined_score[:len(X_normal)], threshold_percentile)
    
    print(f"\nAnomaly Score Statistics:")
    print(f"Reconstruction Error - Mean: {reconstruction_errors[:len(X_normal)].mean():.6f}, "
          f"Std: {reconstruction_errors[:len(X_normal)].std():.6f}")
    print(f"Reconstruction Threshold ({threshold_percentile}th percentile): {recon_threshold:.6f}")
    print(f"Combined Threshold ({threshold_percentile}th percentile): {combined_threshold:.6f}")
    
    # Convert to probabilities
    def anomaly_score_to_probability(scores, threshold):
        normalized_score = scores / threshold
        probabilities = 1 / (1 + np.exp(-5 * (normalized_score - 1)))
        return probabilities
    
    cleaning_probabilities = anomaly_score_to_probability(combined_score, combined_threshold)
    
    # Add results to dataframe
    df['reconstruction_error'] = reconstruction_errors
    df['attention_anomaly_score'] = attention_anomaly
    df['combined_anomaly_score'] = combined_score
    df['cleaning_probability'] = cleaning_probabilities
    df['predicted_cleaning'] = (cleaning_probabilities > 0.5).astype(int)
    
    # Identify most likely cleaning times
    df_sorted = df.sort_values('cleaning_probability', ascending=False)
    
    print("\n" + "="*80)
    print("TOP 20 MOST LIKELY CLEANING EVENTS (FTTransformer)")
    print("="*80)
    
    top_results = df_sorted.head(20)[['open_start_time', 'close_end_time', 
                                       'reconstruction_error', 'attention_anomaly_score',
                                       'combined_anomaly_score', 'cleaning_probability', 
                                       'label', 'operational_phase']]
    
    for idx, row in top_results.iterrows():
        rank = list(top_results.index).index(idx) + 1
        print(f"\nRank {rank}:")
        print(f"  Time Period: {row['open_start_time']} to {row['close_end_time']}")
        print(f"  Reconstruction Error: {row['reconstruction_error']:.6f}")
        print(f"  Attention Anomaly: {row['attention_anomaly_score']:.6f}")
        print(f"  Combined Score: {row['combined_anomaly_score']:.6f}")
        print(f"  Cleaning Probability: {row['cleaning_probability']:.2%}")
        print(f"  Actual Label: {row['label']}")
        print(f"  Operational Phase: {row['operational_phase']}")
    
    # Summary statistics by actual label
    print("\n" + "="*80)
    print("STATISTICS BY ACTUAL LABEL")
    print("="*80)
    
    label_stats = df.groupby('label').agg({
        'reconstruction_error': ['mean', 'std', 'min', 'max'],
        'combined_anomaly_score': ['mean', 'std', 'min', 'max'],
        'cleaning_probability': ['mean', 'std', 'min', 'max'],
        'label': 'count'
    })
    label_stats.columns = ['_'.join(col).strip() for col in label_stats.columns]
    label_stats = label_stats.rename(columns={'label_count': 'count'})
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
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - FTTransformer Cleaning Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('fttransformer_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Visualizations
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    # Plot 1: Reconstruction error
    ax1 = axes[0]
    ax1.plot(df.index, reconstruction_errors, alpha=0.7, label='Reconstruction Error', linewidth=1)
    ax1.axhline(y=recon_threshold, color='r', linestyle='--', 
                label=f'Threshold ({threshold_percentile}th percentile)', linewidth=2)
    if df['is_cleaning'].sum() > 0:
        cleaning_indices = df[df['is_cleaning'] == 1].index
        ax1.scatter(cleaning_indices, reconstruction_errors[cleaning_indices], 
                   color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Reconstruction Error')
    ax1.set_title('Reconstruction Error Over Time (FTTransformer)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Attention anomaly score
    ax2 = axes[1]
    ax2.plot(df.index, attention_anomaly, alpha=0.7, color='purple', 
             label='Attention Anomaly Score', linewidth=1)
    if df['is_cleaning'].sum() > 0:
        ax2.scatter(cleaning_indices, attention_anomaly[cleaning_indices], 
                   color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Attention Anomaly Score')
    ax2.set_title('Attention-based Anomaly Score Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Combined anomaly score
    ax3 = axes[2]
    ax3.plot(df.index, combined_score, alpha=0.7, color='green', 
             label='Combined Anomaly Score', linewidth=1)
    ax3.axhline(y=combined_threshold, color='r', linestyle='--', 
                label=f'Threshold ({threshold_percentile}th percentile)', linewidth=2)
    if df['is_cleaning'].sum() > 0:
        ax3.scatter(cleaning_indices, combined_score[cleaning_indices], 
                   color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Combined Anomaly Score')
    ax3.set_title('Combined Anomaly Score Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cleaning probability
    ax4 = axes[3]
    ax4.plot(df.index, cleaning_probabilities, alpha=0.7, color='orange', 
             label='Cleaning Probability', linewidth=1)
    ax4.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold (0.5)', linewidth=2)
    if df['is_cleaning'].sum() > 0:
        ax4.scatter(cleaning_indices, cleaning_probabilities[cleaning_indices], 
                   color='red', s=100, marker='x', label='Actual Cleaning Events', zorder=5)
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Probability')
    ax4.set_title('Predicted Cleaning Probability Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fttransformer_cleaning_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    train_recon_errors = reconstruction_errors[:len(X_normal)]
    
    # Reconstruction error distribution
    ax1 = axes[0]
    ax1.hist(train_recon_errors, bins=50, alpha=0.6, label='Normal Operations', 
             density=True, color='blue')
    if df['is_cleaning'].sum() > 0:
        cleaning_recon_errors = reconstruction_errors[df['is_cleaning'] == 1]
        ax1.hist(cleaning_recon_errors, bins=30, alpha=0.6, label='Actual Cleaning', 
                density=True, color='red')
    ax1.axvline(x=recon_threshold, color='black', linestyle='--', label='Threshold', linewidth=2)
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Reconstruction Errors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Attention anomaly distribution
    ax2 = axes[1]
    train_attention = attention_anomaly[:len(X_normal)]
    ax2.hist(train_attention, bins=50, alpha=0.6, label='Normal Operations', 
             density=True, color='blue')
    if df['is_cleaning'].sum() > 0:
        cleaning_attention = attention_anomaly[df['is_cleaning'] == 1]
        ax2.hist(cleaning_attention, bins=30, alpha=0.6, label='Actual Cleaning', 
                density=True, color='red')
    ax2.set_xlabel('Attention Anomaly Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Attention Anomaly Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Combined score distribution
    ax3 = axes[2]
    train_combined = combined_score[:len(X_normal)]
    ax3.hist(train_combined, bins=50, alpha=0.6, label='Normal Operations', 
             density=True, color='blue')
    if df['is_cleaning'].sum() > 0:
        cleaning_combined = combined_score[df['is_cleaning'] == 1]
        ax3.hist(cleaning_combined, bins=30, alpha=0.6, label='Actual Cleaning', 
                density=True, color='red')
    ax3.axvline(x=combined_threshold, color='black', linestyle='--', 
                label='Threshold', linewidth=2)
    ax3.set_xlabel('Combined Anomaly Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Combined Anomaly Scores')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fttransformer_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Export results
    output_df = df[['open_start_time', 'close_end_time', 'reconstruction_error',
                    'attention_anomaly_score', 'combined_anomaly_score',
                    'cleaning_probability', 'predicted_cleaning', 'label', 
                    'operational_phase', 'cycle_count']].copy()
    
    # Save high-probability cleaning events
    high_prob_cleaning = output_df[output_df['cleaning_probability'] > 0.75].sort_values(
        'cleaning_probability', ascending=False)
    
    print(f"\n{len(high_prob_cleaning)} cycles identified with >50% cleaning probability")
    
    # Save to CSV
    output_df.to_csv('fttransformer_cleaning_predictions_all.csv', index=False)
    high_prob_cleaning.to_csv('fttransformer_cleaning_predictions_high_probability.csv', index=False)
    
    print("\nResults saved to:")
    print("  - fttransformer_cleaning_predictions_all.csv")
    print("  - fttransformer_cleaning_predictions_high_probability.csv")
    print("  - best_fttransformer_model.pt")
    
    # Analyze attention patterns for top anomalies
    print("\n" + "="*80)
    print("ANALYZING ATTENTION PATTERNS FOR TOP ANOMALIES")
    print("="*80)
    
    # Get attention for top 5 anomalies
    top_5_indices = df_sorted.head(5).index.tolist()
    top_5_data = torch.FloatTensor(X_all_scaled[top_5_indices]).to(device)
    
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(top_5_data, return_attention=True)
    
    # Average across layers and heads
    avg_attention = torch.stack(attention_weights).mean(dim=0)
    cls_attention = avg_attention[:, 0, 1:].cpu().numpy()  # [5, n_features]
    
    # Find top attended features for each anomaly
    for i, idx in enumerate(top_5_indices):
        top_features_idx = np.argsort(cls_attention[i])[-10:][::-1]
        print(f"\nTop Anomaly #{i+1} (Index {idx}):")
        print(f"  Time: {df.loc[idx, 'open_start_time']}")
        print(f"  Cleaning Probability: {df.loc[idx, 'cleaning_probability']:.2%}")
        print(f"  Top 10 Attended Features:")
        for rank, feat_idx in enumerate(top_features_idx, 1):
            if feat_idx < len(feature_names):
                print(f"    {rank}. {feature_names[feat_idx]}: {cls_attention[i, feat_idx]:.4f}")
    
    return model, df, scaler, feature_names

if __name__ == "__main__":
    model, df, scaler, feature_names = main()
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)