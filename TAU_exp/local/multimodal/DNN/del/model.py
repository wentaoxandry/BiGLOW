import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from attentions import *


class DSW(nn.Module):
    def __init__(self, iadim=1024, itdim=1024, hiddendim=16, odim=7):
        super(DSW, self).__init__()
        self.repre = torch.nn.Linear(itdim, iadim)
        self.weightfeatext1 = torch.nn.Sequential(
        torch.nn.Linear(iadim + itdim, iadim),
        torch.nn.LayerNorm(iadim),
        torch.nn.GELU(),
        torch.nn.Linear(iadim, hiddendim),
        )
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.weightpred = torch.nn.Sequential(torch.nn.Linear(37, hiddendim, bias=True),
                                              torch.nn.GELU(),
                                              torch.nn.Linear(hiddendim, int(hiddendim/2), bias=True),
                                              torch.nn.GELU(),
                                              torch.nn.Linear(int(hiddendim/2), 2, bias=True),
                                              torch.nn.Sigmoid())
    def forward(self, input):
        '''
        input_shape: (batch_size, utt_max_lens, featureExtractor_dim)
        utt_mask_shape: (batch_size, utt_max_lens)
        '''
        audiofeat, audioprob, textfeat, textprob = input
        #print(audiofeat.size())
        #print(textfeat.size())

        weightfeat1 = torch.cat([audiofeat, textfeat], dim=-1)
        weightfeat1 = self.weightfeatext1(weightfeat1)
        #cosfeat = self.cossim(audiofeat, self.repre(textfeat)).unsqueeze(1)
        cosprob = self.cossim(audioprob, textprob).unsqueeze(1)
        diff = audioprob - textprob
        summ = audioprob + textprob
        weightfeat2 = torch.cat([cosprob, diff, summ], dim=-1)

        weightfeat = torch.cat([weightfeat2, weightfeat1], dim=-1)
        weights = self.weightpred(weightfeat)
        x = weights[:, 0].unsqueeze(1) * audioprob + weights[:, 1].unsqueeze(1) * textprob
        return x, weights
class DSW_classic(nn.Module):
    def __init__(self, iadim=1024, itdim=1024, hiddendim=16, odim=7):
        super(DSW_classic, self).__init__()
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
        self.repre = torch.nn.Linear(itdim, iadim)
        self.weightfeatext1 = torch.nn.Sequential(
        torch.nn.Linear(iadim + itdim, iadim),
        torch.nn.LayerNorm(iadim),
        torch.nn.GELU(),
        torch.nn.Linear(iadim, hiddendim),
        )
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.weightpred = torch.nn.Sequential(torch.nn.Linear(37, 16, bias=True),
                                              torch.nn.GELU(),
                                              torch.nn.Linear(16, 8, bias=True),
                                              torch.nn.GELU(),
                                              torch.nn.Linear(8, 2, bias=True),
                                              torch.nn.Sigmoid())
        self.classifier = torch.nn.Sequential(torch.nn.Linear(iadim + itdim, iadim, bias=True),
                                                torch.nn.LayerNorm(iadim),
                                                torch.nn.GELU(),
                                              torch.nn.Dropout(p=0.1, inplace=True),
                                              torch.nn.Linear(iadim, odim, bias=True))
    def forward(self, input):
        '''
        input_shape: (batch_size, utt_max_lens, featureExtractor_dim)
        utt_mask_shape: (batch_size, utt_max_lens)
        '''
        audiofeat, audioprob, textfeat, textprob = input

        weightfeat1 = torch.cat([audiofeat, textfeat], dim=-1)
        weightfeat1 = self.weightfeatext1(weightfeat1)
        cosfeat = self.cossim(audiofeat, self.repre(textfeat)).unsqueeze(1)
        cosprob = self.cossim(audioprob, textprob).unsqueeze(1)
        diff = audioprob - textprob
        summ = audioprob + textprob
        weightfeat2 = torch.cat([cosprob, diff, summ], dim=-1)

        weightfeat = torch.cat([weightfeat2, weightfeat1], dim=-1)
        weights = self.weightpred(weightfeat)
        x = torch.cat([weights[:, 0].unsqueeze(1) * audiofeat, weights[:, 1].unsqueeze(1) * textfeat], dim=-1)
        #x = weights[:, 0].unsqueeze(1) * audiofeat + weights[:, 1].unsqueeze(1) * textfeat

        return self.classifier(x), None

class Representation_classic(nn.Module):
    def __init__(self, iadim=1024, itdim=1024, hiddendim=16, odim=7):
        super(Representation_classic, self).__init__()
        # loaded_model \
        self.classifier = torch.nn.Sequential(torch.nn.Linear(iadim + itdim, iadim, bias=True),
                                                torch.nn.LayerNorm(iadim),
                                                torch.nn.GELU(),
                                              torch.nn.Dropout(p=0.1, inplace=True),
                                              torch.nn.Linear(iadim, odim, bias=True))

    def forward(self, input):
        '''
        input_shape: (batch_size, utt_max_lens, featureExtractor_dim)
        utt_mask_shape: (batch_size, utt_max_lens)
        '''
        audiofeat, audioprob, textfeat, textprob = input

        weightfeat1 = torch.cat([audiofeat, textfeat], dim=-1)
        logits = self.classifier(weightfeat1)
        return logits, None

class Embeddingfusion(nn.Module):
    def __init__(self, iadim=1024, itdim=1024, hiddendim=16, n_head=4, dropout_rate=0.0, num_blocks=6, odim=7, eunits=2048):
        super(Embeddingfusion, self).__init__()
        self.hiddendim = hiddendim
        # subsampling text and audio embeddings, if don't want, it can be removed
        self.textsubsampling = torch.nn.Sequential(torch.nn.Linear(itdim, hiddendim),
                                                   torch.nn.LayerNorm(hiddendim))
        self.audiosubsampling = torch.nn.Sequential(torch.nn.Linear(iadim, hiddendim),
                                                   torch.nn.LayerNorm(hiddendim))

        # creat attention encoders
        encoder_selfattn_layer = MultiHeadedAttention
        encoder_selfattn_layer_args1024 = [
                                          (   n_head,
                                              hiddendim,
                                              dropout_rate,
                                          )
                                      ] * num_blocks
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args1024 = (hiddendim, eunits, dropout_rate)
        self.normalize_before = True
        concat_after = False
        stochastic_depth_rate = 0.0
        self.after_norm = LayerNorm(hiddendim)
        self.encoder = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                hiddendim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args1024[lnum]),
                positionwise_layer(*positionwise_layer_args1024),
                dropout_rate,
                self.normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks,
            ))
        self.norm1 = LayerNorm(hiddendim)
        self.BiLSTM = torch.nn.LSTM(input_size=hiddendim, hidden_size=hiddendim, num_layers=2, batch_first=True, bidirectional=True)
        self.norm2 = LayerNorm(hiddendim * 2)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(2 * hiddendim, hiddendim, bias=True),
                                          torch.nn.GELU(),
                                          torch.nn.Linear(hiddendim, odim, bias=True))

    def forward(self, input):
        '''
        input_shape: (batch_size, utt_max_lens, featureExtractor_dim)
        utt_mask_shape: (batch_size, utt_max_lens)
        '''
        audiofeat, audioprob, textfeat, textprob = input

        audiofeat = self.audiosubsampling(audiofeat)
        textfeat = self.textsubsampling(textfeat)

        features = torch.cat([audiofeat, textfeat], dim=1)

        features, _ = self.encoder(features, None)
        features = self.norm1(features)
        blstmfeats, _ = self.BiLSTM(features)

        blstmfeatsforwards = blstmfeats[:, :, :self.hiddendim]
        blstmfeatbackwards = blstmfeats[:, :, self.hiddendim:]

        classification_feat = torch.cat([blstmfeatsforwards[:, -1, :], blstmfeatbackwards[:, 0, :]], dim=-1)
        classification_feat = self.norm2(classification_feat)
        return self.classifier(classification_feat)