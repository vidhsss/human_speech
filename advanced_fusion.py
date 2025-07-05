import torch
import torch.nn as nn

class CrossModalTransformerDecoder(nn.Module):
    def __init__(self, fusion_dim, num_classes, num_layers=2, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=fusion_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, encoder_outputs, tgt_embeds=None, tgt_mask=None, memory_mask=None):
        # encoder_outputs: [batch, seq_len, fusion_dim]
        # tgt_embeds: [batch, tgt_seq_len, fusion_dim] or None (use zeros if None)
        if tgt_embeds is None:
            # Use a single zero vector as start token
            batch_size = encoder_outputs.size(0)
            tgt_embeds = torch.zeros(batch_size, 1, encoder_outputs.size(2), device=encoder_outputs.device)
        decoded = self.transformer_decoder(
            tgt=tgt_embeds,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        logits = self.classifier(decoded)
        return logits 