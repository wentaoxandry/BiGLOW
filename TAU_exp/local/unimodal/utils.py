import torch
import numpy as np
import kaldiio
import cv2
import random
import librosa
from transformers import ViTFeatureExtractor, AutoProcessor
from torch.utils.data import Dataset

def pad_video_sequence(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>: A sequence with a length of 4, representing the node sets sequence in index 0, neighbor sets sequence in index 1, public edge mask sequence in index 2 and label sequence in index 3.
                          And the length of each sequences are same as the batch size.
                          sequences: [node_sets_sequence, neighbor_sets_sequence, public_edge_mask_sequence, label_sequence]
    Return:
        node_sets_sequence <torch.LongTensor>: The padded node sets sequence (works with batch_size >= 1).
        neighbor_sets_sequence <torch.LongTensor>: The padded neighbor sets sequence (works with batch_size >= 1).
        public_edge_mask_sequence <torch.BoolTensor>: The padded public edge mask sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
    '''
    image_sequence = []
    label_sequence = []
    filename_sequence = []
    for imagearray, label, filename in sequences:
        image_sequence.append(imagearray)
        label_sequence.append(label)
        filename_sequence.append(filename)
    image_sequence = torch.nn.utils.rnn.pad_sequence(image_sequence, batch_first=True)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return image_sequence, label_sequence, filename_sequence#, imagelist_sequence

def pad_audio_sequence(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>: A sequence with a length of 4, representing the node sets sequence in index 0, neighbor sets sequence in index 1, public edge mask sequence in index 2 and label sequence in index 3.
                          And the length of each sequences are same as the batch size.
                          sequences: [node_sets_sequence, neighbor_sets_sequence, public_edge_mask_sequence, label_sequence]
    Return:
        node_sets_sequence <torch.LongTensor>: The padded node sets sequence (works with batch_size >= 1).
        neighbor_sets_sequence <torch.LongTensor>: The padded neighbor sets sequence (works with batch_size >= 1).
        public_edge_mask_sequence <torch.BoolTensor>: The padded public edge mask sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
    '''
    audio_sets_sequence = []
    label_sequence = []
    filename_sequence = []
    #imagelist_sequence = []
    for audio_sets, label, filename in sequences:
        audio_sets_sequence.append(audio_sets.squeeze(0))
        label_sequence.append(label)
        filename_sequence.append(filename)
        #imagelist_sequence.append(imagelist)
    audio_sets_sequence = torch.nn.utils.rnn.pad_sequence(audio_sets_sequence, batch_first=True, padding_value=0)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return audio_sets_sequence, label_sequence, filename_sequence#, imagelist_sequence

class Audiofeatdatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, dev_file, test_file, max_len=2500):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.max_len = max_len
        self.train_dataset, self.dev_dataset, self.test_dataset = self.prepare_dataset()

    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        train_dataset = Audiofeatloader(self.train_file, self.max_len)
        dev_dataset = Audiofeatloader(self.dev_file, self.max_len)
        test_dataset = Audiofeatloader(self.test_file, self.max_len)
        return train_dataset, dev_dataset, test_dataset

class Audiofeatloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, max_len):
        super(Audiofeatloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.max_len = max_len
        self.datadict = datadict
        self.processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        filename = self.datakeys[index]
        audiodir = self.datadict[filename]['audiodir']#.replace('./', './../../')
        y, sr = librosa.load(audiodir, sr=22050)
        SPEECH_WAVEFORM = librosa.resample(y, orig_sr=sr, target_sr=16000)
        inputs = self.processor(SPEECH_WAVEFORM, sampling_rate=16000, return_tensors="pt")
        audiofeat = torch.FloatTensor(inputs['input_values'])
        label = self.datadict[filename]['label']

        if label == 'airport':
            label = 0
        elif label == 'shopping_mall':
            label = 1
        elif label == 'metro_station':
            label = 2
        elif label == 'street_pedestrian':
            label = 3
        elif label == 'public_square':
            label = 4
        elif label == 'street_traffic':
            label = 5
        elif label == 'tram':
            label = 6
        elif label == 'bus':
            label = 7
        elif label == 'metro':
            label = 8
        elif label == 'park':
            label = 9
        label = torch.LongTensor([label])


        return audiofeat, label, filename

class Videofeatdatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, dev_file, test_file, cashedir):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.cashedir = cashedir
        self.train_dataset, self.dev_dataset, self.test_dataset = self.prepare_dataset()

    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        train_dataset = videodatasetclass(self.train_file, self.cashedir)
        dev_dataset = videodatasetclass(self.dev_file, self.cashedir)
        test_dataset = videodatasetclass(self.test_file, self.cashedir)
        return train_dataset, dev_dataset, test_dataset
class videodatasetclass(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, cashedir):
        super(videodatasetclass).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.cashedir = cashedir
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224', cache_dir=cashedir)

    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        #print(self.datakeys[index])
        #print(index)

        label = self.datadict[self.datakeys[index]]['label']
        filename = self.datakeys[index]
        videodir = self.datadict[self.datakeys[index]]['videodir']#.replace('./', './../../')
        video = torch.load(videodir)

        if label == 'airport':
            label = 0
        elif label == 'shopping_mall':
            label = 1
        elif label == 'metro_station':
            label = 2
        elif label == 'street_pedestrian':
            label = 3
        elif label == 'public_square':
            label = 4
        elif label == 'street_traffic':
            label = 5
        elif label == 'tram':
            label = 6
        elif label == 'bus':
            label = 7
        elif label == 'metro':
            label = 8
        elif label == 'park':
            label = 9
        label = torch.LongTensor([label])


        return video, label, filename
