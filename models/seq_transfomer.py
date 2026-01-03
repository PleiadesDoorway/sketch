import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """经典 Transformer 位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x):
        # x: [B, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]

class SketchTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=4, num_classes=345, dropout=0.1):
        super().__init__()
        self.input_dim = 3  # coordinate(2) + flag_bits(1)
        self.embed_dim = embed_dim

        # 线性投影到 embed_dim
        self.input_proj = nn.Linear(self.input_dim, embed_dim)

        # 位置编码
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=100)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, coordinate, flag_bits, padding_mask=None, attention_mask=None):
        """
        coordinate: [B, 100, 2]
        flag_bits: [B, 100, 1]
        padding_mask: [B, 100] 1 for keep, 0 for pad (optional)
        attention_mask: [B, 100, 100] (optional)
        """
        # 拼接输入
        x = torch.cat([coordinate, flag_bits], dim=-1)  # [B,100,3]

        # 投影到 embed_dim
        x = self.input_proj(x)  # [B,100,embed_dim]

        # 加位置编码
        x = self.pos_encoding(x)  # [B,100,embed_dim]

        # Transformer Encoder
        # 注意: PyTorch TransformerEncoder 支持 padding_mask: True->mask掉
        # padding_mask: 0 for pad positions, 1 for real tokens -> 需要转换
        if padding_mask is not None:
            key_padding_mask = (padding_mask.squeeze(-1) == 0)  # True 表示 pad -> mask
        else:
            key_padding_mask = None

        # x: [B, seq_len, embed_dim]
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # 取序列池化（mean pooling）
        x = x.mean(dim=1)  # [B, embed_dim]

        logits = self.classifier(x)  # [B, num_classes]
        return logits
