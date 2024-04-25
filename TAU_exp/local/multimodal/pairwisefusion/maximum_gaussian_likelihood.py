import os
import torch
from tqdm import tqdm
import json
import time
import kaldiio
import numpy as np
import pingouin as pg
import argparse
from scipy.stats import multivariate_normal
import logging

# = MulticlassF1Score(num_classes=7, average='weighted')
logsoft = torch.nn.LogSoftmax(dim=-1)

def Henze_Zirkler_normality_test(data):
    """
    Args:
        data: the datato test if it is multivariant gaussian distribution

    Returns: p-value, if p-value > 0.05, it is gaussian distribution

    """
    data = data.numpy()
    mardia_test = pg.multivariate_normality(data)
    return mardia_test.pval, mardia_test.normal
def acc(pred, label):
    return (np.sum((pred.data.numpy() == label.data.numpy()), axis=-1) / pred.size())[0]

def find_best(accdict):
    Keymax = max(zip(accdict.values(), accdict.keys()))[1]
    return Keymax
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

def parameter_estimate(trainaudiodata, trainvideodata, trainlabel, i):
    # split pair i and j samples based on the ground truth labels
    audio_i = trainaudiodata[trainlabel == i]#[:, i]
    video_i = trainvideodata[trainlabel == i]#[:, i]

    y = torch.cat([audio_i, video_i], dim=1)

    mean = np.mean(y.numpy(), axis=0) #torch.mean(betai, dim=0)
    cov = np.cov(y, rowvar=False)

    return mean, cov

def gaussiandis(Gaussian_param_dict, audioprob, videoprob, i):
    y = np.concatenate((audioprob, videoprob), axis=0)
    gaussianparm = Gaussian_param_dict[i]
    mv_normal = multivariate_normal(mean=gaussianparm[0], cov=gaussianparm[1])
    pdf = mv_normal.pdf(y)
    return pdf


def Gaussian_test(trainaudiodata, trainvideodata, trainlabel, i):
    # split pair i and j samples based on the ground truth labels
    audio_i = trainaudiodata[trainlabel == i]  # [:, i]
    video_i = trainvideodata[trainlabel == i]  # [:, i]

    y = torch.cat([audio_i, video_i], dim=1)

    skpval, sknormal = Henze_Zirkler_normality_test(y)
    logging.info(f"p-value of the Henze-Zirkler normality test for class {i} is {skpval}.")


def GML(traindict, devdict, savedir, N, ifgaussiantest):
    '''
    Gaussian maximal likelihood
    '''
    logging.basicConfig(filename=os.path.join(savedir, 'train.log'), level=logging.INFO)
    start_time = time.time()
    trainaudiodata = []
    trainvideodata = []
    trainlabel = []
    for uttid in list(traindict.keys()):
        trainaudiodata.append(torch.FloatTensor(traindict[uttid]['audioprob'][0]).unsqueeze(0))
        trainvideodata.append(torch.FloatTensor(traindict[uttid]['videoprob'][0]).unsqueeze(0))
        trainlabel.append(torch.LongTensor(traindict[uttid]['label'][0]))
    trainaudiodata = torch.concatenate(trainaudiodata, dim=0)
    trainvideodata = torch.concatenate(trainvideodata, dim=0)
    trainlabel = torch.concatenate(trainlabel, dim=0)
    Gaussian_param_dict = {}
    for i in range(N):
        mean, cov = parameter_estimate(trainaudiodata, trainvideodata, trainlabel, i)
        Gaussian_param_dict.update({i: [mean, cov]})
        #print(mean)
    end_time = time.time()
    time_usage = end_time - start_time
    logging.info(f"Training stage takes {time_usage} seconds to complete.")
    if ifgaussiantest is True:
        for i in range(N):
            Gaussian_test(trainaudiodata, trainvideodata, trainlabel, i)

    
    start_time = time.time()
    # ML Classification
    videopreds = []
    audiopreds = []
    multipreds = []
    labels = []
    output = {}
    for _, uttid in enumerate(tqdm(list(devdict.keys()))):
        gaussianprob = []
        audiopred = torch.argmax(torch.softmax(torch.FloatTensor(devdict[uttid]['audioprob'][0]), dim=-1), dim=-1)
        videopred = torch.argmax(torch.softmax(torch.FloatTensor(devdict[uttid]['videoprob'][0]), dim=-1), dim=-1)
        audioprob = devdict[uttid]['audioprob'][0]
        videoprob = devdict[uttid]['videoprob'][0]
        label = devdict[uttid]['label'][0]
        for i in range(N):
            gaussianprob.append(gaussiandis(Gaussian_param_dict, audioprob, videoprob, i))
        
        multipreds.append(torch.argmax(torch.FloatTensor(gaussianprob)))
        videopreds.append(videopred)
        audiopreds.append(audiopred)
        labels.append(label.tolist()[0])

        output.update({uttid: {}})
        output[uttid].update({'prob': torch.softmax(torch.FloatTensor(gaussianprob), dim=0).tolist()[0]})
        output[uttid].update({'predict': int(np.argmax(np.asarray(gaussianprob)))})
        output[uttid].update({'label': int(label[0])})

    #print(videopreds)
    videopreds = torch.LongTensor(videopreds)
    audiopreds = torch.LongTensor(audiopreds)
    multipreds = torch.LongTensor(multipreds)
    labels = torch.LongTensor(labels)#.squeeze(1)

    videoscore = acc(videopreds, labels)
    audioscore = acc(audiopreds, labels)
    multiscore = acc(multipreds, labels)

    with open(os.path.join(savedir, 'videoacc_' + str(videoscore) + '_audioacc' + str(
                audioscore) + '_multiacc' + str(multiscore) + ".json"), 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    end_time = time.time()
    time_usage = end_time - start_time
    logging.info(f"Test stage takes {time_usage} seconds to complete.")

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='MGL', type=str, help='which data stream')
    parser.add_argument('--audiomodal', default='audio', type=str, help='which data stream')
    parser.add_argument('--videomodal', default='video', type=str, help='single or multi images')
    parser.add_argument('--savedir', default='./../../../trained', type=str, help='Dir to save trained model and results')
    parser.add_argument('--ifgaussiantest', default='true', type=str, help='if using Henze-Zirkler multivariate normality test')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasdir = args.datasdir
    modal = args.modal
    audiomodal = args.audiomodal
    videomodal = args.videomodal
    savedir = args.savedir
    ifgaussiantest = args.ifgaussiantest
    savedir = os.path.join(savedir, 'Multimodal', modal)
    if ifgaussiantest == 'true':
        ifgaussiantest = True
    else:
        ifgaussiantest = False

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    N = 10  # the number of classes
    with open(os.path.join(datasdir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datasdir, "dev.json"), encoding="utf8") as json_file:
        devdict = json.load(json_file)
    with open(os.path.join(datasdir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    prior = {}
    for i in range(N):
        prior.update({i: []})
    for workdict, dset in zip([devdict, testdict], ["dev", "test"]):
        videofeatscpdir = os.path.join(args.datasdir, 'data', videomodal, dset + 'feats.scp')
        videoprobscpdir = os.path.join(args.datasdir, 'data', videomodal, dset + 'prob.scp')
        audiofeatscpdir = os.path.join(args.datasdir, 'data', audiomodal, dset + 'feats.scp')
        audioprobscpdir = os.path.join(args.datasdir, 'data', audiomodal, dset + 'prob.scp')
        labelscpdir = os.path.join(args.datasdir, 'data', audiomodal, dset + 'label.scp')

        audiofeat, audioprob, videofeat, videoprob, label = {}, {}, {}, {}, {}
        for scpfiles in [videofeatscpdir, videoprobscpdir, audiofeatscpdir, audioprobscpdir, labelscpdir]:
            with open(scpfiles) as f:
                srcdata = f.readlines()
            datadict = {}
            for j in srcdata:
                fullname = j.split(' ')[0]
                # subnames = fullname.split('_')
                datadir = j.split(' ')[1].strip('\n')
                datadict.update({fullname: datadir})
            if scpfiles == videofeatscpdir:
                videofeat = datadict
            elif scpfiles == videoprobscpdir:
                videoprob = datadict
            elif scpfiles == audiofeatscpdir:
                audiofeat = datadict
            elif scpfiles == audioprobscpdir:
                audioprob = datadict
            elif scpfiles == labelscpdir:
                label = datadict

        savedict = {}
        for filename in list(workdict.keys()):
            try:
                savedict.update({filename: {}})
                savedict[filename].update({'audioprob': _get_from_loader(
                    filepath=audioprob[filename].replace('./', './../../../'), filetype='mat')})
                savedict[filename].update({'videoprob': _get_from_loader(
                    filepath=videoprob[filename].replace('./', './../../../'), filetype='mat')})
                labeldata = _get_from_loader(filepath=label[filename].replace('./', './../../../'), filetype='mat')
                if dset == 'test':
                    prior[labeldata[0][0]].append(filename)
                savedict[filename].update(
                    {'label': labeldata})


            except:
                del savedict[filename]

        if dset == 'dev':
            devdict = savedict
        elif dset == 'test':
            testdict = savedict

    priorlist = [len(prior[i]) / len(devdict.keys()) for i in range(N)]

    GML(devdict, testdict, savedir, N, ifgaussiantest)









