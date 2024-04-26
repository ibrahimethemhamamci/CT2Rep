import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from ctvit import CTViT


class CT2RepLongModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(CT2RepLongModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        model = CTViT(
                            dim = 512,
                            codebook_size = 8192,
                            image_size = 480,
                            patch_size = 24,
                            temporal_patch_size = 12,
                            spatial_depth = 4,
                            temporal_depth = 4,
                            dim_head = 32,
                            heads = 8
                        )
        self.visual_extractor = VisualExtractor(model, args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.forward = self.forward_ct2replong

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_ct2replong(self, images, context,images2,targets=None,mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        att_feats2, fc_feats2 = self.visual_extractor(images2)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, context,fc_feats2, att_feats2,mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats,targets, context,fc_feats2, att_feats2,mode='sample')
        else:
            raise ValueError
        return output

