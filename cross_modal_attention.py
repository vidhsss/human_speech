import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, audio_dim, video_dim, hidden_dim):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
    
    def forward(self, audio_feats, video_feats):
        # audio_feats: [batch, seq_len, audio_dim]
        # video_feats: [batch, seq_len, video_dim] (repeat/copy video along seq_len)
        audio_proj = self.audio_proj(audio_feats)
        video_proj = self.video_proj(video_feats)
        # Cross-attend: audio queries, video keys/values
        attn_output, _ = self.attn(audio_proj, video_proj, video_proj)
        # Fuse: sum attended video and audio
        fused = attn_output + audio_proj
        return fused 