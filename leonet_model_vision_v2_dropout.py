
import torch
import torch.nn as nn
from torchvision import models

class VisionEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Linear(512, output_dim)

    def forward(self, image):
        with torch.no_grad():  # Freeze vision encoder
            features = self.backbone(image)
        return self.proj(features)

class LeoNetVisionV2(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, n_layers=4):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, hidden_dim)
        self.vision_encoder = VisionEncoder(output_dim=hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.text_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, vocab_size)
        )

        self.motor_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 8 * 128)
        )

    def forward(self, input_ids, image_tensor):
        text_embeds = self.text_embed(input_ids)           # [B, 8, D]
        vision_embeds = self.vision_encoder(image_tensor).unsqueeze(1)  # [B, 1, D]
        combined = torch.cat([vision_embeds, text_embeds], dim=1)       # [B, 9, D]

        encoded = self.transformer(combined)  # [B, 9, D]
        text_out = encoded[:, 1:, :]          # [B, 8, D]
        vision_out = encoded[:, 0]            # [B, D]

        logits = self.text_head(text_out)                         # [B, 8, vocab]
        motor_out = self.motor_head(vision_out).view(-1, 8, 128)  # [B, 8, 128]
        return logits, motor_out
