# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
    Model Architecture modified from DETR for OmniShotCut Model
"""
import torch
import torch.nn.functional as F
from torch import nn


# Import files from the local fodler
# from util import box_ops
from util.misc import NestedTensor
from .backbone import build_backbone
from .transformer import build_transformer
from datasets.utils import nested_tensor_from_tensor_list



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




class OmniShotCut(nn.Module):
    """ This is the OmniShotCut module that performs object detection """

    def __init__(self, backbone, transformer, num_intra_relation_classes, num_inter_relation_classes, num_frames, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         OmniShotCut can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_frames = num_frames
        self.aux_loss = aux_loss


        # Trainable Parameters
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.intra_relation_class_embed = nn.Linear(hidden_dim, num_intra_relation_classes)       
        self.inter_relation_class_embed = nn.Linear(hidden_dim, num_inter_relation_classes)        
        self.range_class_embed = nn.Linear(hidden_dim, num_frames + 2)          # TODO: s+2 is to add some padding 
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size = 1)          
        
        

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "intra_clip_logits": the classification logits (including no-object) for all queries.
               - "inter_clip_logits": the classification logits (including no-object) for all queries.
               - "pred_shot_logits": The normalized ranges coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)


        # Call the Backbone (ResNet18)
        features, pos = self.backbone(samples)          # Joiner outputs ResNet features and the position embedding
        pos = pos[-1]                                   # output shape is (B, F, 384(C), H, W) 
        src, mask = features[-1].decompose()            # src shape is [B * F, C, H, W]     
        assert mask is not None


        # Reshape Vision Inputs
        src_proj = self.input_proj(src)
        n, c, h, w = src_proj.shape
        # Reshape src_proj to (B, C, F, H*W);   mask to (B, F, H*W)
        src_proj = src_proj.reshape(n // self.num_frames, self.num_frames, c, h, w).permute(0, 2, 1, 3, 4).flatten(-2)          
        mask = mask.reshape(n // self.num_frames, self.num_frames, h*w)
        pos = pos.permute(0, 2, 1, 3, 4).flatten(-2)


        # Call Transformer;     output shape is (6, B, #Queries, F*H*W)
        hs = self.transformer(src_proj, mask, self.query_embed.weight, pos)[0]          # src_proj shape is (B, C, F, H*W);   mask shape is (B, F, H*W);    pos shape is (B, C, F, H*W)


        # Output Class will be modified for Cut Anything
        outputs_intra_class = self.intra_relation_class_embed(hs)              
        outputs_inter_class = self.inter_relation_class_embed(hs) 
        outputs_shot_class = self.range_class_embed(hs)             
        out = {
                    'intra_clip_logits': outputs_intra_class[-1], 
                    'inter_clip_logits': outputs_inter_class[-1], 
                    'pred_shot_logits': outputs_shot_class[-1]
               }           # Last Layer
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_intra_class, outputs_inter_class, outputs_shot_class)           # Fetch Central Layer (Except the last layer)
            
        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_intra_class, outputs_inter_class, outputs_shot_class):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'intra_clip_logits': a1, 'inter_clip_logits': a2, 'pred_shot_logits': b}
                for a1, a2, b in zip(outputs_intra_class[:-1], outputs_inter_class[:-1], outputs_shot_class[:-1])]





def build_model(args):

    # num_intra_relation_classes is preset in argument
    num_intra_relation_classes = args.num_intra_relation_classes
    num_inter_relation_classes = args.num_inter_relation_classes


    # Init Model
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = OmniShotCut(
                            backbone,
                            transformer,
                            num_intra_relation_classes = num_intra_relation_classes,
                            num_inter_relation_classes = num_inter_relation_classes,
                            num_frames = args.max_process_window_length, 
                            num_queries = args.num_queries,
                            aux_loss = args.aux_loss,
                        )


    # Return
    return model
