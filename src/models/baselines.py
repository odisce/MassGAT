import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FourierScalarEncoder(nn.Module):
    """
    Fourier feature encoding for scalar inputs (e.g. meta features such as into, imax, etc.)
    """
    def __init__(self, input_dim=8, embed_dim=64, sigma=10.0):
        super().__init__()
        # Fourier Features projection
        self.register_buffer('B', torch.randn(input_dim, embed_dim // 2) * sigma) # buffer is used to store non-parameter tensors
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # x: [Batch, input_dim]
        # Fourier Projection:
        proj = 2 * math.pi * (x @ self.B) 
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.output_mlp(ff)

class ProfileTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=2, num_layers=1):
        super().__init__()
        
        # Project Intensity (scalar) to Vector
        self.input_proj = nn.Linear(1, d_model)
        
        # Project Time (scalar) to Vector (This replaces fixed PositionalEncoding)
        self.time_proj = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2, 
            batch_first=True, 
            dropout=0.3,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.Linear(d_model, 1)

    def forward(self, x_intensity, x_time):
        """
        x_intensity: [Batch, 32] - Normalized Intensities
        x_time:      [Batch, 32] - Relative RT (in seconds or normalized units)
        """
        # embed intensity and time separately
        h_int = self.input_proj(x_intensity.unsqueeze(-1))
        h_time = self.time_proj(x_time.unsqueeze(-1))
        
        # combine
        h = h_int + h_time
        
        # padding mask
        padding_mask = (x_intensity == 0)
        
        # transformer Pass
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        
        # attention pooling
        scores = self.attention_pool(h).squeeze(-1) # [Batch, 32]
        scores = scores.masked_fill(padding_mask, -1e9) # Ignore padding
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        
        # Weighted Sum
        return torch.sum(h * weights, dim=1) # [Batch, d_model]


class CNNPeakEncoder(nn.Module):
    """
    Standard 1D-CNN baseline for peak profile encoding.
    """
    def __init__(self, output_dim=64, conv_dropout=0.3):
        super().__init__()
        self.conv_dropout = conv_dropout
        

        # layer 1
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        # layer 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        # layer 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # global max pooling (for translation invariant property)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # final projection to match the Transformer output dimension
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x_intensity, x_time):
        """
        x_intensity: [Batch, 32]
        x_time:      [Batch, 32]
        """

        x = torch.stack([x_intensity, x_time], dim=1) # [Batch, 2, 32]
        
        # convolutional Layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.dropout(x, p=self.conv_dropout, training=self.training)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.dropout(x, p=self.conv_dropout, training=self.training)

        x = F.relu(self.bn3(self.conv3(x))) # Output: [Batch, 64, 32]
        
        # global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # final projection
        return self.fc(x)
    


class IndependentPeakClassifier(nn.Module):
    def __init__(self, num_features=72, hidden_dim=48, 
                 encoder_type='transformer',
                 fusion_dropout=0.3,
                 classifier_dropout=0.5,
                 conv_dropout=0.3):
        super().__init__()
        
        self.signal_embed_dim = 32
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.fusion_dropout = fusion_dropout
        self.classifier_dropout = classifier_dropout
        self.conv_dropout = conv_dropout
        
        # 1. Encoders (Must match GNN)
        self.scalar_encoder = FourierScalarEncoder(input_dim=8, embed_dim=self.signal_embed_dim)
        
        if self.encoder_type == 'cnn':
               self.signal_encoder = CNNPeakEncoder(output_dim=self.signal_embed_dim, conv_dropout=conv_dropout)
        elif self.encoder_type == 'transformer':
             self.signal_encoder = ProfileTransformerEncoder(d_model=self.signal_embed_dim)
        else:
            # use mlp as default
            self.signal_encoder = nn.Sequential(
                nn.Linear(2, self.signal_embed_dim),
                nn.ReLU(),
                nn.Linear(self.signal_embed_dim, self.signal_embed_dim)
            )
             
        # 2. Fusion (Combining scalars + profile)
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.signal_embed_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout)
        )
        
        # 3. Classifier Head (MLP)
        # No GNN layers here. Just direct prediction.
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1) # Output 1 logit
        )

    def forward(self, x):
        # x shape: [Batch, 72]
        # Scalars: indices 0-8
        scalars = x[:, 0:8] 
        
        # Profiles
        rt_seq = x[:, 8:40]
        int_seq = x[:, 40:72]
        
        # Encode
        emb_scalar = self.scalar_encoder(scalars)
        emb_profile = self.signal_encoder(int_seq, rt_seq)
        
        # Concatenate and Project
        combined = torch.cat([emb_profile, emb_scalar], dim=1)
        h = self.fusion_proj(combined)
        
        # Classify
        logits = self.classifier(h)
        return logits