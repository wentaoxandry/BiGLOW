import os
import numpy as np
import torch
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask

def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    return ~make_pad_mask(lengths, xs, length_dim)

def pad_title(sequences):
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
    node_sets_sequence = []
    mask_sequence = []
    label_sequence = []
    filename_sequence = []
    for node_sets, mask, label, filename in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        mask_sequence.append(mask.squeeze(0))
        label_sequence.append(label)
        filename_sequence.append(filename)
        #imagelist_sequence.append(imagelist)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=1)
    mask_sequence = torch.nn.utils.rnn.pad_sequence(mask_sequence, batch_first=True, padding_value=0)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return node_sets_sequence, mask_sequence, label_sequence, filename_sequence#, imagelist_sequence

def pad_BERT_title(sequences):
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
    node_sets_sequence = []
    mask_sequence = []
    label_sequence = []
    filename_sequence = []
    for node_sets, mask, label, filename in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        mask_sequence.append(mask.squeeze(0))
        label_sequence.append(label)
        filename_sequence.append(filename)
        #imagelist_sequence.append(imagelist)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=0)
    mask_sequence = torch.nn.utils.rnn.pad_sequence(mask_sequence, batch_first=True, padding_value=0)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return node_sets_sequence, mask_sequence, label_sequence, filename_sequence#, imagelist_sequence

def pad_image(sequences):
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
    for image, label, filename in sequences:
        image_sequence.append(image.squeeze(0))
        label_sequence.append(label)
        filename_sequence.append(filename)
        #imagelist_sequence.append(imagelist)
    image_sequence = torch.nn.utils.rnn.pad_sequence(image_sequence, batch_first=True)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return image_sequence, label_sequence, filename_sequence#, imagelist_sequence

def pad_image_LRP(sequences):
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
    org_image_sequence = []
    for image, label, filename, org_image in sequences:
        image_sequence.append(image.squeeze(0))
        label_sequence.append(label)
        filename_sequence.append(filename)
        org_image_sequence.append(org_image)
    image_sequence = torch.nn.utils.rnn.pad_sequence(image_sequence, batch_first=True)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return image_sequence, label_sequence, filename_sequence, org_image_sequence#, imagelist_sequence

class Titledatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, dev_file, test_file, tokenizer, max_len=2500):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.train_dataset, self.dev_dataset, self.test_dataset = self.prepare_dataset()

    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        [self.train_file[id].update(
            {'encode': self.tokenizer(self.train_file[id]['title'], return_tensors='pt', truncation=True,
                                      max_length=self.max_len)})
            for id in list(self.train_file.keys())]
        [self.test_file[id].update(
            {'encode': self.tokenizer(self.test_file[id]['title'], return_tensors='pt', truncation=True,
                                      max_length=self.max_len)})
            for id in list(self.test_file.keys())]
        [self.dev_file[id].update(
            {'encode': self.tokenizer(self.dev_file[id]['title'], return_tensors='pt', truncation=True,
                                      max_length=self.max_len)})
            for id in list(self.dev_file.keys())]

        # preparing self.train_dataset
        train_dataset = Titleloader(self.train_file)
        dev_dataset = Titleloader(self.dev_file)
        test_dataset = Titleloader(self.test_file)
        return train_dataset, dev_dataset, test_dataset

class Titleloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict): #, max_len):
        super(Titleloader).__init__()
        self.datakeys = self._get_keys(datadict)
        #self.max_len = max_len
        self.datadict = datadict
    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        filename = self.datakeys[index]
        ids = self.datadict[filename]['encode'].data['input_ids']
        masks = self.datadict[filename]['encode'].data['attention_mask']
        label = self.datadict[filename]['6_way_label']

        label = torch.LongTensor([label])



        return ids, masks, label, filename


class Imagedatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, dev_file, test_file, tokenizer):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.train_dataset, self.dev_dataset, self.test_dataset = self.prepare_dataset()

    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        train_dataset = Imageloader(self.train_file, self.tokenizer)
        dev_dataset = Imageloader(self.dev_file, self.tokenizer)
        test_dataset = Imageloader(self.test_file, self.tokenizer)
        return train_dataset, dev_dataset, test_dataset

class Imageloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, tokenizer): #, max_len):
        super(Imageloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.tokenizer = tokenizer
        self.datadict = datadict
    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        filename = self.datakeys[index]
        imagedir = self.datadict[filename]['imgdir']
        img = Image.open(imagedir.replace('./', '/data/wentao/Fakeddit/')).convert('RGB')
        #img = Image.open(imagedir.replace('./', './../../')).convert('RGB')
        img = img.resize((256, 256))
        inputs = self.tokenizer(images=img, return_tensors="pt")

        imgdata = inputs.data['pixel_values']
        label = self.datadict[filename]['6_way_label']

        label = torch.LongTensor([label])



        return imgdata, label, filename
    

class Imagedatasetclass_LRP:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, dev_file, test_file, transformimg=None):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.transformimg = transformimg
        self.train_dataset, self.dev_dataset, self.test_dataset = self.prepare_dataset()

    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        train_dataset = Imageloader_LRP(self.train_file, self.transformimg)
        dev_dataset = Imageloader_LRP(self.dev_file, self.transformimg)
        test_dataset = Imageloader_LRP(self.test_file, self.transformimg)
        return train_dataset, dev_dataset, test_dataset
class Imageloader_LRP(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, transformimg): #, max_len):
        super(Imageloader_LRP).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.transformimg = transformimg
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            normalize,
                        ])
    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        filename = self.datakeys[index]
        imagedir = self.datadict[filename]['imgdir']
        img = Image.open(imagedir.replace('./', '/data/wentao/Fakeddit/')).convert('RGB')
        #img = Image.open(imagedir.replace('./', './../../')).convert('RGB')
        imgdata = self.transform(img)
        #print(imgdata.size())
        #img = img.resize((256, 256))
        #inputs = self.tokenizer(images=img, return_tensors="pt")

        #imgdata = inputs.data['pixel_values']
        label = self.datadict[filename]['6_way_label']

        label = torch.LongTensor([label])



        return imgdata, label, filename, img
