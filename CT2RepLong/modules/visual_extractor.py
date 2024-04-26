import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, feature_model, args):
        super(VisualExtractor, self).__init__()
        self.model = feature_model
        self.avg_fnt = torch.nn.AvgPool3d(kernel_size=20, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images, return_encoded_tokens=True)
        patch_feats = patch_feats.permute(0,4,1,2,3)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
