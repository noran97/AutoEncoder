# model.py
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),              # تحويل الصورة إلى vector
            nn.Linear(28*28, 256),     # طبقة fully connected
            nn.ReLU(),                 # تفعيل ReLU
            nn.Linear(256, latent_dim) # إخراج latent vector
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return z
