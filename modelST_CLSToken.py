import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1250):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device)
        pos = self.pos_embedding(positions)
        return x + pos.unsqueeze(0)


class EEGTransformerEncoder(nn.Module):
    def __init__(self, in_channels=32, d_model=128, nhead=4, num_layers=8, dim_feedforward=256):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = LearnablePositionalEmbedding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # x: (B, C, T)
        x = x.permute(0, 2, 1)       # (B, T, C)
        x = self.input_proj(x)       # (B, T, d_model)
        x = self.pos_enc(x)
        z = self.encoder(x)          # (B, T, d_model)
        return z.permute(0, 2, 1)    # (B, d_model, T)


class ChannelWiseTransformerEncoder(nn.Module):
    def __init__(self, time_len=128, d_model=128, nhead=4, num_layers=8, dim_feedforward=256):
        super().__init__()
        self.input_proj = nn.Linear(time_len, d_model)
        self.pos_enc = LearnablePositionalEmbedding(d_model, max_len=64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # x: (B, C, T)
        x = self.input_proj(x)        # (B, C, d_model)
        x = self.pos_enc(x)           # (B, C, d_model)
        z = self.encoder(x)           # (B, C, d_model)
        return z.permute(0, 2, 1)     # (B, d_model, C)

class SpatioTemporalEncoder(nn.Module):
    def __init__(self, in_channels=32, time_len=128, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.temporal_encoder = EEGTransformerEncoder(
            in_channels=in_channels, d_model=d_model,
            nhead=nhead, num_layers=num_layers
        )
        self.spatial_encoder = ChannelWiseTransformerEncoder(
            time_len=time_len, d_model=d_model,
            nhead=nhead, num_layers=num_layers
        )

        self.fusion = nn.Linear(2 * d_model, d_model)

    def forward(self, x):  # x: (B, C, T)
        z_temp = self.temporal_encoder(x)       # (B, d_model, T)
        z_spat = self.spatial_encoder(x)        # (B, d_model, C)

        # Προσαρμογή για σωστό fusion: spatial → (B, d_model, T)
        if z_spat.shape[-1] != z_temp.shape[-1]:
            z_spat = F.interpolate(z_spat, size=z_temp.shape[-1], mode='linear', align_corners=False)

        z_fused = torch.cat([z_temp, z_spat], dim=1)   # (B, 2*d_model, T)
        z_fused = z_fused.permute(0, 2, 1)             # (B, T, 2*d_model)
        z_out = self.fusion(z_fused)                   # (B, T, d_model)
        return z_out.permute(0, 2, 1)                  # (B, d_model, T)




class EEGDecoder(nn.Module):
    def __init__(self, hidden_dim=128, out_channels=32):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, z):
        return self.decoder(z)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        sim = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        return F.cross_entropy(sim, labels)



class PreHeadAttentionBlock(nn.Module):
    def __init__(self, d_model,nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout = dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,z):
        z = z.permute(0,2,1)
        attn_output,_ = self.attn(z,z,z)
        z = self.norm(z + self.dropout(attn_output))
        return z.permute(0,2,1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query_seq, context_seq):
        # query_seq: (B, Tq, d)
        # context_seq: (B, Tc, d)
        attn_output, _ = self.cross_attn(query_seq, context_seq, context_seq)
        return self.norm(attn_output + query_seq)  # residual + norm



class MultiHeadEEGModel(nn.Module):
    def __init__(self, in_channels=32, d_model=128, H=10, temperature=0.1, num_layers=2, num_classes=4, time_len=125):
        super().__init__()

        self.time_len = time_len  # ΝΕΟ

        self.encoder = SpatioTemporalEncoder(
            in_channels=in_channels,
            time_len=self.time_len,
            d_model=d_model
        )

        

        self.reconstruction_head = EEGDecoder(hidden_dim=d_model, out_channels=in_channels)
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.BatchNorm1d(d_model),
            nn.Flatten(),
            nn.Linear(d_model, num_classes)
        )

        # self.contrastive_loss = ContrastiveLoss(temperature)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.pre_head_attention = PreHeadAttentionBlock(d_model=d_model)  # Optional
        self.deep_cca = DeepCCAProjector(in_dim_s=H, target_dim = d_model)

    def forward(self, x):
        z = self.encoder(x)                  # (B, d_model, T)
        z = z.permute(0, 2, 1)               # (B, T, d_model)
        z = self.encoder_norm(z)             # LayerNorm over d_model
        z = z.permute(0, 2, 1)               # (B, d_model, T)
        z = self.pre_head_attention(z)
        
        z = nn.functional.relu(z)
        
        x_hat = self.reconstruction_head(z)  # (B, C, T)
        y_hat = self.classification_head(z)  # (B, num_classes)
        return x_hat, y_hat, z

    # def contrastive_loss_forward(self, x1, x2):
    #     z1 = self.encoder(x1).mean(dim=2)   # (B, d_model)
    #     z2 = self.encoder(x2).mean(dim=2)
    #     return self.contrastive_loss(z1, z2)

class MultiHeadEEGModelCLS(nn.Module):
    def __init__(self, in_channels=32, d_model=128, H=10, temperature=0.1, num_layers=2,
                 num_classes=4, time_len=125, num_heads=4):
        super().__init__()

        self.time_len = time_len

        # --- Encoders ---
        self.encoder = SpatioTemporalEncoder(
            in_channels=in_channels,
            time_len=self.time_len,
            d_model=d_model
        )

        self.context_encoder = SpatioTemporalEncoder(
            in_channels= H,
            time_len=self.time_len,
            d_model=d_model
        )


        # --- Heads ---
        self.reconstruction_head = EEGDecoder(hidden_dim=d_model, out_channels=in_channels)
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.BatchNorm1d(d_model),
            nn.Dropout(p=0.3),
            nn.Flatten(),
            nn.Linear(d_model, num_classes)
        )

        self.cls_classification_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(p=0.3),
            nn.Linear(d_model, num_classes)
        )

        # --- Normalization ---
        self.encoder_norm = nn.LayerNorm(d_model)

        # --- Cross-Attention ---
        self.cross_attention = CrossAttentionBlock(d_model=d_model, num_heads=num_heads)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # Learnable [CLS] token

        # Optional Deep CCA or projection head
        self.deep_cca = DeepCCAProjector(in_dim_s=H, target_dim=d_model)

    

    def forward(self, x, context_input=None):
        B = x.size(0)
    
        # --- Encode κύρια είσοδο ---
        z = self.encoder(x)            # (B, d_model, T)
        z = z.permute(0, 2, 1)         # (B, T, d_model)
        z = self.encoder_norm(z)
    
        cls_vector = None  # θα ενημερωθεί αν υπάρχει context
    
        # --- Αν υπάρχει context_input ---
        if context_input is not None:
            # Encode context
            context_z = self.encoder(context_input)      # (B, d_model, T)
            context_z = context_z.permute(0, 2, 1)       # (B, T, d_model)
            context_z = self.encoder_norm(context_z)
    
            # Προσθήκη learnable CLS token
            cls = self.cls_token.expand(B, -1, -1)       # (B, 1, d_model)
            context_with_cls = torch.cat([cls, context_z], dim=1)  # (B, T+1, d_model)
    
            # --- Self-attention μέσα στο context για να ενημερωθεί ο CLS token ---
            context_with_cls = self.cross_attention(context_with_cls, context_with_cls)  # Q=K=V
    
            # Παίρνουμε τον ενημερωμένο [CLS] token
            cls_vector = context_with_cls[:, 0, :]       # (B, d_model)
    
            # --- Cross-attention: το z βλέπει το ενημερωμένο context ---
            z = self.cross_attention(z, context_with_cls)  # Q=z, K,V=context_with_cls
    
        else:
            # Self-attention fallback (χωρίς context)
            z = self.cross_attention(z, z)
    
        # --- Post-processing ---
        z = F.relu(z)
        z = F.dropout(z, p=0.95, training=self.training)
        z = z.permute(0, 2, 1)   # (B, d_model, T)
    
        # --- Reconstruction head ---
        x_hat = self.reconstruction_head(z)
    
        # --- Sequence classification head ---
        y_hat_seq = self.classification_head(z)  # (B, num_classes)
    
        # --- Classification via [CLS] token ---
        if cls_vector is not None:
            y_hat_cls = self.cls_classification_head(cls_vector)  # (B, num_classes)
        else:
            # dummy output in case context is not provided
            y_hat_cls = torch.zeros(B, self.classification_head[-1].out_features, device=x.device)
    
        return x_hat, y_hat_seq, y_hat_cls




class MultiHeadEEGModelFlexible(nn.Module):
    def __init__(
        self,
        in_channels=32,
        time_len=128,
        d_model=128,
        num_classes=4,
        temperature=0.1,
        use_temporal=True,
        use_spatial=True,
        spatial_mode='batch'
    ):
        super().__init__()

        self.encoder = SpatioTemporalEncoder(
            in_channels=in_channels,
            time_len=time_len,
            d_model=d_model,
            use_temporal=use_temporal,
            use_spatial=use_spatial,
            spatial_mode=spatial_mode
        )

        self.reconstruction_head = EEGDecoder(hidden_dim=d_model, out_channels=in_channels)

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.BatchNorm1d(d_model),
            nn.Flatten(),
            nn.Linear(d_model, num_classes)
        )

        # self.contrastive_loss = ContrastiveLoss(temperature)
        self.encoder_norm = nn.LayerNorm(d_model)
        

    def forward(self, x):
        z = self.encoder(x)                  # (B, d_model, T)
        z = z.permute(0, 2, 1)               # (B, T, d_model)
        z = self.encoder_norm(z)             # LayerNorm over d_model
        z = z.permute(0, 2, 1)               # (B, d_model, T)

        x_hat = self.reconstruction_head(z)  # (B, C, T)
        y_hat = self.classification_head(z)  # (B, num_classes)
        return x_hat, y_hat, z

    def contrastive_loss_forward(self, x1, x2):
        z1 = self.encoder(x1).mean(dim=2)    # (B, d_model)
        z2 = self.encoder(x2).mean(dim=2)
        return self.contrastive_loss(z1, z2)






import mlflow
import mlflow.pytorch
import torch
import torch.nn.functional as F

def train_with_mlflow(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs=10,
    device='cuda',
    experiment_name="EEG-SpatioTemporal"
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        model.to(device)

        # Log hyperparams
        mlflow.log_params({
            "d_model": model.encoder.fusion.in_features // 2 if hasattr(model.encoder, 'fusion') else 128,
            "use_temporal": model.encoder.use_temporal,
            "use_spatial": model.encoder.use_spatial,
            "spatial_mode": getattr(model.encoder, 'spatial_mode', 'n/a'),
            "epochs": num_epochs
        })

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()

                x_hat, y_hat, _ = model(xb)
                loss_rec = F.l1_loss(x_hat, xb)
                loss_cls = F.cross_entropy(y_hat, yb)
                loss = loss_rec + loss_cls

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (y_hat.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)

            train_acc = correct / total
            avg_loss = total_loss / len(train_loader)

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)

            print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | Accuracy: {train_acc:.4f}")

        # Optionally validate
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, y_hat, _ = model(xb)
                correct += (y_hat.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
            val_acc = correct / total
            mlflow.log_metric("val_acc", val_acc)
            print(f"✅ Validation Accuracy: {val_acc:.4f}")

        # Log the model
        mlflow.pytorch.log_model(model, "model")


class DeepCCAProjector(nn.Module):
    def __init__(self, in_dim_s, target_dim):
        """
        in_dim_s: αριθμός καναλιών στο ref_template (H)
        target_dim: τελική διάσταση για προβολή (π.χ. d_model)
        """
        super().__init__()

        self.proj_s = nn.Sequential(
            nn.Linear(in_dim_s, target_dim),
            nn.LayerNorm(target_dim),
            nn.ReLU(),
            nn.Linear(target_dim, target_dim),
            nn.LayerNorm(target_dim)
        )

    def forward(self, x_encoded, ref_template):
        """
        x_encoded: (B, d_model, T) → αφήνεται ως έχει
        ref_template: (B, H, T) → γίνεται (B, d_model, T)
        """
        s_proj = self.apply_proj(self.proj_s, ref_template)  # (B, d_model, T)
        x_proj = F.normalize(x_encoded.permute(0, 2, 1), dim=-1)  # (B, T, d_model)
        s_proj = F.normalize(s_proj.permute(0, 2, 1), dim=-1)     # (B, T, d_model)

        return x_proj, s_proj

    def apply_proj(self, proj, x):
        """
        x: (B, H, T)
        -> transpose to (B, T, H) so each timestep has H features
        -> project H → d_model
        -> return as (B, d_model, T)
        """
        x = x.permute(0, 2, 1)       # (B, T, H)
        x = proj(x)                  # (B, T, d_model)
        return x.permute(0, 2, 1)    # (B, d_model, T)



def deep_cca_loss(x_proj, s_proj, mask=None):
    """
    x_proj, s_proj: (B, T, proj_dim)
    mask: (B, T) — 1 για έγκυρα σημεία, 0 για masked
    """
    sim = F.cosine_similarity(x_proj, s_proj, dim=-1)  # (B, T)

    if mask is not None:
        mask = mask.float()
        sim = sim * mask                     # μηδενίζουμε masked σημεία
        loss = - sim.sum() / mask.sum().clamp(min=1e-6)  # μέσος όρος μόνο στα έγκυρα
    else:
        loss = - sim.mean()

    return loss

import warnings

def deep_cca_loss_svd(x_proj, s_proj, mask=None, out_dim=10, reg=1e-4):
    """
    x_proj, s_proj: (B, T, proj_dim)
    mask: (B, T) optional — 1 for valid time steps, 0 for masked.
    out_dim: number of canonical correlations to keep.
    reg: regularization for numerical stability.
    """
    B, T, D = x_proj.shape

    # Check: Batch Size vs Projection Dimension
    if B <= D:
        warnings.warn(f"Batch size (B={B}) is not greater than projection dimension (D={D}). "
                      "Covariance matrices may be singular. Increase batch size or decrease proj_dim.", UserWarning)

    # Mask Handling and Effective Batch Size
    if mask is not None:
        effective_B = mask.sum().item() / T
        if effective_B <= D:
            warnings.warn(f"Effective batch size after masking is {effective_B:.2f}, which is <= proj_dim ({D}). "
                          "Consider reducing the masking ratio or increasing batch size.", UserWarning)
        mask = mask.float().unsqueeze(-1)  # (B, T, 1)
        x_sum = (x_proj * mask).sum(dim=1)
        s_sum = (s_proj * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        x_mean = x_sum / counts
        s_mean = s_sum / counts
    else:
        x_mean = x_proj.mean(dim=1)
        s_mean = s_proj.mean(dim=1)

    # Center the data
    x_mean_c = x_mean - x_mean.mean(dim=0, keepdim=True)
    s_mean_c = s_mean - s_mean.mean(dim=0, keepdim=True)

    # Covariance Matrices
    Sigma_xx = (x_mean_c.T @ x_mean_c) / (B - 1) + reg * torch.eye(D, device=x_proj.device)
    Sigma_ss = (s_mean_c.T @ s_mean_c) / (B - 1) + reg * torch.eye(D, device=s_proj.device)
    Sigma_xs = (x_mean_c.T @ s_mean_c) / (B - 1)

    # Inverse square root via Cholesky
    def inv_sqrt(mat):
        try:
            chol = torch.linalg.cholesky(mat)
            inv_chol = torch.cholesky_inverse(chol)
            return inv_chol @ inv_chol.T
        except RuntimeError as e:
            raise ValueError(f"Cholesky decomposition failed. Matrix might not be positive definite.\n"
                             f"Try increasing reg (current: {reg}).\nOriginal Error: {e}")

    Sigma_xx_inv_sqrt = inv_sqrt(Sigma_xx)
    Sigma_ss_inv_sqrt = inv_sqrt(Sigma_ss)

    # Compute T matrix and SVD
    T = Sigma_xx_inv_sqrt @ Sigma_xs @ Sigma_ss_inv_sqrt
    U, S, V = torch.linalg.svd(T)

    max_possible_out_dim = min(B - 1, D)
    if out_dim > max_possible_out_dim:
        warnings.warn(f"Requested out_dim={out_dim} exceeds maximum possible value ({max_possible_out_dim}). "
                      f"Using out_dim={max_possible_out_dim} instead.", UserWarning)
        out_dim = max_possible_out_dim

    corr = S[:out_dim]
    loss = - torch.sum(corr)

    return loss

def deep_cca_loss_svd_debug(x_proj, s_proj, mask=None, out_dim=10, reg=1e-4):
    B, T, D = x_proj.shape

    # Normalize inputs (πολύ σημαντικό!)
    x_proj = F.normalize(x_proj, dim=-1)
    s_proj = F.normalize(s_proj, dim=-1)

    if mask is not None:
        mask = mask.float().unsqueeze(-1)
        x_mean = (x_proj * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        s_mean = (s_proj * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    else:
        x_mean = x_proj.mean(dim=1)
        s_mean = s_proj.mean(dim=1)

    # === Debug print: before centering
    # print(f"[Debug] x_mean stats: mean={x_mean.mean().item():.4f}, std={x_mean.std().item():.4f}")
    # print(f"[Debug] s_mean stats: mean={s_mean.mean().item():.4f}, std={s_mean.std().item():.4f}")

    # Center
    x_mean_c = x_mean - x_mean.mean(dim=0, keepdim=True)
    s_mean_c = s_mean - s_mean.mean(dim=0, keepdim=True)

    # Debug norm
    # print(f"[Debug] ||X_centered||_F = {torch.norm(x_mean_c).item():.4f}")
    # print(f"[Debug] ||S_centered||_F = {torch.norm(s_mean_c).item():.4f}")

    # Covariance
    Sigma_xx = (x_mean_c.T @ x_mean_c) / (B - 1) + reg * torch.eye(D, device=x_proj.device)
    Sigma_ss = (s_mean_c.T @ s_mean_c) / (B - 1) + reg * torch.eye(D, device=x_proj.device)
    Sigma_xs = (x_mean_c.T @ s_mean_c) / (B - 1)

    # Inverse square root (whitening)
    # def inv_sqrt(mat):
    #     chol = torch.linalg.cholesky(mat)
    #     inv_chol = torch.cholesky_inverse(chol)
    #     return inv_chol @ inv_chol.T

    def inv_sqrt(mat, eps=1e-6):
        # Συμμετρική ιδιοτιμική αποσύνθεση
        eigvals, eigvecs = torch.linalg.eigh(mat)
        # Εξασφάλιση θετικότητας
        eigvals = torch.clamp(eigvals, min=eps)
        D_inv_sqrt = torch.diag(eigvals.rsqrt())
        return eigvecs @ D_inv_sqrt @ eigvecs.T


    Tmat = inv_sqrt(Sigma_xx) @ Sigma_xs @ inv_sqrt(Sigma_ss)

    # SVD
    U, S, V = torch.linalg.svd(Tmat)
    corrs = S[:out_dim]

    # Debug: show canonical correlations
    # print(f"[Debug] Canonical correlations (top-{out_dim}): {[round(c.item(), 4) for c in corrs]}")

    # If something exploded
    if torch.any(corrs > 1.0 + 1e-3):
        print("[Warning] Canonical correlations exceed 1.0! Possible instability.")

    loss = -torch.sum(corrs)

    return loss


def deep_cca_loss_nowhiten(x_proj, s_proj, mask=None, out_dim=10):
    """
    CCA loss approximation using dot-product between normalized centered projections (no whitening).
    """
    B, T, D = x_proj.shape

    # L2 normalize per time step
    x_proj = F.normalize(x_proj, dim=-1)
    s_proj = F.normalize(s_proj, dim=-1)

    # Mean pooling across time
    if mask is not None:
        mask = mask.float().unsqueeze(-1)
        x_mean = (x_proj * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        s_mean = (s_proj * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    else:
        x_mean = x_proj.mean(dim=1)
        s_mean = s_proj.mean(dim=1)

    # Center (zero-mean across batch)
    x_mean_c = x_mean - x_mean.mean(dim=0, keepdim=True)
    s_mean_c = s_mean - s_mean.mean(dim=0, keepdim=True)

    # Normalize centered features across feature dim
    x_mean_c = F.normalize(x_mean_c, dim=-1)  # (B, D)
    s_mean_c = F.normalize(s_mean_c, dim=-1)  # (B, D)

    # Compute correlation matrix T ~ (D × D)
    T = x_mean_c.T @ s_mean_c / (B - 1)

    # SVD
    U, S, V = torch.linalg.svd(T)
    corrs = S[:out_dim]

    # Debug info (προαιρετικό)
    if torch.any(corrs > 1.0 + 1e-3):
        print("[Warning] Some canonical correlations exceed 1.0 — still unstable.")

    loss = -torch.sum(corrs)

    return loss











