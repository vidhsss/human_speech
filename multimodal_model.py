import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel, ViTModel
from cross_modal_attention import CrossModalAttention

class MultimodalSpeechModel(nn.Module):
    def __init__(self, vision_model_name='google/vit-base-patch16-224', audio_model_name='facebook/wav2vec2-base-960h', fusion_dim=768, num_classes=30):
        super().__init__()
        self.vision_model = ViTModel.from_pretrained(vision_model_name)
        self.audio_model = Wav2Vec2Model.from_pretrained(audio_model_name)
        self.cross_attn = CrossModalAttention(
            audio_dim=self.audio_model.config.hidden_size,
            video_dim=self.vision_model.config.hidden_size,
            hidden_dim=fusion_dim
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, audio, video):
        # audio: [batch, time] or [batch, 1, time]
        if audio.dim() == 3:
            audio = audio[:, 0, :]
        audio_feats = self.audio_model(audio).last_hidden_state  # [batch, seq_len, hidden]
        # video: [batch, frames, C, H, W]
        b, f, c, h, w = video.shape
        video = video.view(-1, c, h, w)
        vision_outputs = self.vision_model(pixel_values=video)
        vision_feats = vision_outputs.pooler_output.view(b, f, -1).mean(dim=1)  # [batch, hidden]
        # Repeat vision_feats along time to match audio_feats
        seq_len = audio_feats.size(1)
        vision_feats_seq = vision_feats.unsqueeze(1).expand(-1, seq_len, -1)
        # Cross-modal attention fusion
        fused = self.cross_attn(audio_feats, vision_feats_seq)  # [batch, seq_len, fusion_dim]
        logits = self.classifier(fused)  # [batch, seq_len, num_classes]
        return logits 