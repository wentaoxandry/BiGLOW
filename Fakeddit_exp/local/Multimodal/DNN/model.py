import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class DSW(nn.Module):
    def __init__(self, iadim=1024, itdim=1024, hiddendim=16, odim=7):
        super(DSW, self).__init__()
        self.repre = torch.nn.Linear(itdim, iadim)
        self.weightfeatext = torch.nn.Sequential(
        torch.nn.Linear(iadim + itdim, iadim),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(iadim),
        torch.nn.Linear(iadim, iadim),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(iadim),
        torch.nn.Linear(iadim, iadim),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(iadim),
        torch.nn.Linear(iadim, 2),
        torch.nn.Sigmoid()
        )

    def forward(self, input):
        '''
        input_shape: (batch_size, utt_max_lens, featureExtractor_dim)
        utt_mask_shape: (batch_size, utt_max_lens)
        '''
        audiofeat, audioprob, textfeat, textprob = input
        #print(audiofeat.size())
        #print(textfeat.size())

        weightfeat = torch.cat([audiofeat, textfeat], dim=-1)
        weights = self.weightfeatext(weightfeat)
        x = weights[:, 0].unsqueeze(1) * audioprob + weights[:, 1].unsqueeze(1) * textprob
        return x, weights

class Representation(nn.Module):
    def __init__(self, iadim=1024, itdim=1024, hiddendim=16, odim=7):
        super(Representation, self).__init__()
        # loaded_model \
        self.classifier = torch.nn.Sequential(torch.nn.Linear(iadim + itdim, iadim, bias=True),
                                                torch.nn.LayerNorm(iadim),
                                                torch.nn.GELU(),
                                              torch.nn.Dropout(p=0.1, inplace=True),
                                              torch.nn.Linear(iadim, odim, bias=True))

    def forward(self, input):
        audiofeat, audioprob, textfeat, textprob = input

        weightfeat1 = torch.cat([audiofeat, textfeat], dim=-1)
        logits = self.classifier(weightfeat1)
        return logits, None

