import torch
import torch.nn as nn
import timm
from transformers import BertForSequenceClassification
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
class TitleBERT(torch.nn.Module):
    def __init__(self, MODEL, odim, cachedir):
        torch.nn.Module.__init__(self)
        # loaded_model \
        self.BERT = BertForSequenceClassification.from_pretrained(MODEL, cache_dir=cachedir,
                                                                       num_labels=odim,
                                                                       output_hidden_states=True,
                                                                       ignore_mismatched_sizes=True)
    def forward(self, inputs):
        nodes = inputs[0]
        mask = inputs[1]
        x = self.BERT(nodes, mask)
        feats = x.hidden_states[-1][:, 0, :]
        return x.logits, feats


class ImageViT(torch.nn.Module):
    def __init__(self, MODEL, odim, cachedir):
        torch.nn.Module.__init__(self)
        # loaded_model \
        self.ViT = timm.create_model(MODEL, pretrained=True, num_classes=odim)

    def forward(self, inputs, ifreturnfeats=False):
        x = self.ViT(inputs)
        if ifreturnfeats is True:
            self.ViT.head = Identity()
            feats = self.ViT(inputs)
            return x, feats
        else:
            return x


