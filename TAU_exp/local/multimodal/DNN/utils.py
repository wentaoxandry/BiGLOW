import os
import numpy as np
import torch
import kaldiio
from collections import OrderedDict
from torch.utils.data import Dataset
#from transformers import ViTFeatureExtractor
#from torchvision.transforms import Compose, RandomAffine, RandomHorizontalFlip, ToTensor, Resize, ToPILImage

def _get_from_loader(filepath, filetype):
    """Return ndarray

    In order to make the fds to be opened only at the first referring,
    the loader are stored in self._loaders

    >>> ndarray = loader.get_from_loader(
    ...     'some/path.h5:F01_050C0101_PED_REAL', filetype='hdf5')

    :param: str filepath:
    :param: str filetype:
    :return:
    :rtype: np.ndarray
    """
    if filetype in ['mat', 'vec']:
        # e.g.
        #    {"input": [{"feat": "some/path.ark:123",
        #                "filetype": "mat"}]},
        # In this case, "123" indicates the starting points of the matrix
        # load_mat can load both matrix and vector
        #filepath = filepath.replace('/home/wentao', '.')
        return kaldiio.load_mat(filepath)
    elif filetype == 'scp':
        # e.g.
        #    {"input": [{"feat": "some/path.scp:F01_050C0101_PED_REAL",
        #                "filetype": "scp",
        filepath, key = filepath.split(':', 1)
        loader = self._loaders.get(filepath)
        if loader is None:
            # To avoid disk access, create loader only for the first time
            loader = kaldiio.load_scp(filepath)
            self._loaders[filepath] = loader
        return loader[key]
    else:
        raise NotImplementedError(
            'Not supported: loader_type={}'.format(filetype))


def Padding(sequences):
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
    audiofeat_sequence = []
    audioprob_sequence = []
    videofeat_sequence = []
    videoprob_sequence = []
    label_sequence = []
    filename_sequence = []
    for audiofeat, audioprob, videofeat, videoprob, label, filename in sequences:
        audiofeat_sequence.append(audiofeat.squeeze(0))
        audioprob_sequence.append(audioprob.squeeze(0))
        videofeat_sequence.append(videofeat.squeeze(0))
        videoprob_sequence.append(videoprob.squeeze(0))
        label_sequence.append(label)
        filename_sequence.append(filename)
    audiofeat_sequence = torch.nn.utils.rnn.pad_sequence(audiofeat_sequence, batch_first=True, padding_value=0)
    audioprob_sequence = torch.nn.utils.rnn.pad_sequence(audioprob_sequence, batch_first=True, padding_value=0)
    videofeat_sequence = torch.nn.utils.rnn.pad_sequence(videofeat_sequence, batch_first=True, padding_value=0)
    videoprob_sequence = torch.nn.utils.rnn.pad_sequence(videoprob_sequence, batch_first=True, padding_value=0)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return audiofeat_sequence, audioprob_sequence, videofeat_sequence, videoprob_sequence, label_sequence, filename_sequence#, imagelist_sequence

class Loaddatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, dev_file, test_file, train_file=None):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.train_dataset, self.dev_dataset, self.test_dataset = self.prepare_dataset()

    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        if self.train_file == None:
            train_dataset = None
        else:
            train_dataset = featuredataloader(self.train_file)
        dev_dataset = featuredataloader(self.dev_file)
        test_dataset = featuredataloader(self.test_file)
        return train_dataset, dev_dataset, test_dataset




class featuredataloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict):
        super(featuredataloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        filename = self.datakeys[index]


        audiofeatdir = self.datadict[filename]['audiofeat']
        audioprobdir = self.datadict[filename]['audioprob']
        videofeatdir = self.datadict[filename]['videofeat']
        videoprobdir = self.datadict[filename]['videoprob']
        labeldata = _get_from_loader(filepath=self.datadict[filename]['label'], filetype='mat')



        '''audiofeatdir = self.datadict[filename]['audiofeat'].replace('./', './../../../')
        audioprobdir = self.datadict[filename]['audioprob'].replace('./', './../../../')
        videofeatdir = self.datadict[filename]['videofeat'].replace('./', './../../../')
        videoprobdir = self.datadict[filename]['videoprob'].replace('./', './../../../')
        labeldata = _get_from_loader(filepath=self.datadict[filename]['label'].replace('./', './../../../'),
                                     filetype='mat')'''


        audiofeat = _get_from_loader(
                filepath=audiofeatdir,
                filetype='mat')
        audioprob = _get_from_loader(
            filepath=audioprobdir,
            filetype='mat')
        videofeat = _get_from_loader(
            filepath=videofeatdir,
            filetype='mat')
        videoprob = _get_from_loader(
            filepath=videoprobdir,
            filetype='mat')



        audiofeat = torch.FloatTensor(audiofeat)
        audioprob = torch.FloatTensor(audioprob)
        videofeat = torch.FloatTensor(videofeat)
        videoprob = torch.FloatTensor(videoprob)

        label = torch.LongTensor([int(labeldata[0][0])])

        return audiofeat, audioprob, videofeat, videoprob, label, filename  # , imagelist  # twtfsingdata.squeeze(0), filename
