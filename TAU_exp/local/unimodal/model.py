import torch
import math
from transformers import ViTForImageClassification, ASTForAudioClassification

#https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial

class Audiospectrumtransformer(torch.nn.Module):
   def __init__(self, odim=4, cachedir=None):
       super(Audiospectrumtransformer, self).__init__()
       self.AST = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", cache_dir=cachedir)
       self.classifier = torch.nn.Sequential(torch.nn.Linear(527, 527, bias=True),
                                              torch.nn.Dropout(p=0.1, inplace=True),
                                              torch.nn.Linear(527, odim, bias=True))
   def forward(self, audio_feat):
       feats = self.AST(audio_feat).logits
       x = self.classifier(feats)
       return x, feats


       return output, attention_embeddings

class E2EViTsingle(torch.nn.Module):
    def __init__(self, odim=2, cachedir=None):
        torch.nn.Module.__init__(self)
        # loaded_model \
        self.ViT = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224', cache_dir=cachedir)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(1000, 1000, bias=True),
                                              torch.nn.Dropout(p=0.1, inplace=True),
                                              torch.nn.Linear(1000, odim, bias=True))

    def forward(self, image):
        BS, N, C, H, W = image.size()
        image = image.view(-1, C, H, W)
        feats = self.ViT(image).logits
        feats = feats.view(BS, N, -1)
        feats = torch.mean(feats, dim=1)
        x = self.classifier(feats)
        return x, feats
