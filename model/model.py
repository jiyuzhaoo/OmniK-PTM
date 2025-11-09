import torch
import torch.nn as nn
import torch.nn.init as init
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super(CrossAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # (Batch, Seq_Len, Channels)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        output = query + self.dropout(attn_output)
        output = self.layer_norm(output)
        return output

class CAFN(nn.Module):
    def __init__(self, input_dim=46, num_classes=4, hidden_size=128):
        super(CAFN, self).__init__()
        self.conv_layer11 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_layer12 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.Residual = FeatureExtractor()
        self.embed_dim = 64
        self.num_heads = 8
        self.cross_attn_1_to_2 = CrossAttentionBlock(self.embed_dim, self.num_heads)
        self.cross_attn_2_to_1 = CrossAttentionBlock(self.embed_dim, self.num_heads)
        self.hidden_size = 64
        self.biGRU = nn.GRU(
            input_size=self.embed_dim* 2,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        mlp_input_dim = self.hidden_size * 2
        mlp_hidden_dim = 64
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(mlp_hidden_dim, num_classes)
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif type(m) == nn.GRU:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, x1, x2):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        x1 = x1.to(device).unsqueeze(1)
        x1_conv = self.conv_layer11(x1)
        _, w1 = self.Residual(x1_conv)  # w1 shape: (B, 64, L1)
        x2 = x2.to(device).transpose(1, 2)
        x2_conv = self.conv_layer12(x2)
        _, w2 = self.Residual(x2_conv)  # w2 shape: (B, 64, L2)
        w1_p = w1.permute(0, 2, 1)  # Shape: (B, L, 64)
        w2_p = w2.permute(0, 2, 1)  # Shape: (B, L, 64)

        fused_w1 = self.cross_attn_1_to_2(query=w1_p, key=w2_p, value=w2_p)  # Shape: (B, L, 64)
        fused_w2 = self.cross_attn_2_to_1(query=w2_p, key=w1_p, value=w1_p)  # Shape: (B, L, 64)
        x = torch.cat((fused_w1, fused_w2), dim=2)  # Shape: (B, L, 128)

        self.biGRU.flatten_parameters()
        output, _ = self.biGRU(x)  # output shape: (B, L, hidden_size * 2)
        forward_out = output[:, -1, :self.hidden_size]
        backward_out = output[:, 0, self.hidden_size:]

        x = torch.cat((forward_out, backward_out), dim=1)
        x = self.mlp_head(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=46, num_classes=4):
        super(FeatureExtractor, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2)
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

    def forward(self, x):
        x1 = self.conv_layer1(x)
        x2 = self.conv_layer2(x1)
        w1 = x2
        x3 = self.conv_layer3(x2)
        return x3, w1