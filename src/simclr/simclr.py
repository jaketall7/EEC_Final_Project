import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def nt_xent_loss(z_i, z_j, temperature=0.5):
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    similarity = torch.matmul(z, z.T)
    N = z_i.shape[0]

    mask = (~torch.eye(2*N, dtype=bool)).to(z.device)
    sim = similarity / temperature
    exp_sim = torch.exp(sim) * mask

    positive_sim = torch.exp(F.cosine_similarity(z_i, z_j) / temperature)
    positives = torch.cat([positive_sim, positive_sim], dim=0)

    denominator = exp_sim.sum(dim=1)
    loss = -torch.log(positives / denominator)
    return loss.mean()

class FrozenDinoSimCLR(nn.Module):
    """
    Frozen DINO backbone + trainable 1x1 input projector + trainable projection head.

    Expected input:
        x: [B, in_channels, H, W]

    Output:
        feats: backbone features before projection head
        z:     normalized projection-head output for contrastive loss
    """
    def __init__(
        self,
        dino_backbone: nn.Module,
        in_channels: int = 8,
        backbone_feat_dim: int = 768,
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 128,
        freeze_backbone: bool = True,
        input_proj = "1x1"
    ):
        super().__init__()

        # Trainable 1x1 conv to map 8 -> 3 channels
        if (input_proj == "1x1"):
            self.input_projector = nn.Conv2d(
                in_channels=in_channels,
                out_channels=3,
                kernel_size=1,
                bias=False,
            )

        else :

            self.input_projector = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 3, kernel_size=1),
            )

        self.backbone = dino_backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # Trainable SimCLR projection head
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_feat_dim, proj_hidden_dim),
            nn.GELU(), #nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_out_dim),
        )

    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        out = self.backbone(x)
        
        if out.ndim == 3:
            return out[:, 0]
        
        return out

    def forward(self, x: torch.Tensor):
        # [B, C, H, W] -> [B, 3, H, W]
        x = self.input_projector(x)

        # Keep backbone frozen even if model.train() is called
        # with torch.no_grad():
        #     feats = self.extract_backbone_features(x)
        feats = self.extract_backbone_features(x)

        z = self.projection_head(feats)
        z = F.normalize(z, dim=1)

        return feats, z


def get_simclr_transforms():
    simclr_transforms_strong = transforms.Compose([
        # transforms.Resize((500, 500)),
        transforms.RandomResizedCrop(64, scale=(0.1, 0.8)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        # transforms.ToTensor()
    ])

    simclr_transforms_weak = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=1),
        # transforms.ToTensor()
    ])

    return simclr_transforms_weak