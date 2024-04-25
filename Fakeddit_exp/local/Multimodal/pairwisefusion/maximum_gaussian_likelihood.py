import os
import json
import torch
from tqdm import tqdm
import json
import time
import kaldiio
import pingouin as pg
import numpy as np
import argparse
from scipy.stats import multivariate_normal
from torchmetrics.classification import MulticlassF1Score
import logging

f1score = MulticlassF1Score(num_classes=7, average='weighted')
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

def parameter_estimate(trainimagedata, trainvideodata, trainlabel, i):
    # split pair i and j samples based on the ground truth labels
    image_i = trainimagedata[trainlabel == i]#[:, i]
    video_i = trainvideodata[trainlabel == i]#[:, i]

    y = torch.cat([image_i, video_i], dim=1)

    mean = np.mean(y.numpy(), axis=0) #torch.mean(betai, dim=0)
    cov = np.cov(y, rowvar=False)

    return mean, cov

def gaussiandis(Gaussian_param_dict, imageprob, videoprob, i):
    y = np.concatenate((imageprob, videoprob), axis=0)
    gaussianparm = Gaussian_param_dict[i]
    mv_normal = multivariate_normal(mean=gaussianparm[0], cov=gaussianparm[1])
    pdf = mv_normal.pdf(y)
    return pdf


def Gaussian_test(trainimagedata, traintextdata, trainlabel, i):
    # split pair i and j samples based on the ground truth labels
    image_i = trainimagedata[trainlabel == i]#[:, i]
    text_i = traintextdata[trainlabel == i]#[:, i]

    y = torch.cat([image_i, text_i], dim=1)
    
    skpval, sknormal = Henze_Zirkler_normality_test(y)
    logging.info(f"p-value of the Henze-Zirkler normality test for class {i} is {skpval}.")


def MGL(traindict, devdict, testdict, savedir, N, ifgaussiantest):
    logging.basicConfig(filename=os.path.join(savedir, 'train.log'), level=logging.INFO)
    #for i in list(devdict.keys()):
    #    traindict.update({i + '_dev': devdict[i]})
    traindict = devdict
    start_time = time.time()
    trainimagedata = []
    traintextdata = []
    trainlabel = []
    for uttid in list(traindict.keys()):
        trainimagedata.append(torch.FloatTensor(traindict[uttid]['imageprob'][0]).unsqueeze(0))
        traintextdata.append(torch.FloatTensor(traindict[uttid]['textprob'][0]).unsqueeze(0))
        trainlabel.append(torch.LongTensor(traindict[uttid]['label'][0]))
    trainimagedata = torch.concatenate(trainimagedata, dim=0)
    traintextdata = torch.concatenate(traintextdata, dim=0)
    trainlabel = torch.concatenate(trainlabel, dim=0)
    Gaussian_param_dict = {}
    for i in range(N):
        mean, cov = parameter_estimate(trainimagedata, traintextdata, trainlabel, i)
        Gaussian_param_dict.update({i: [mean, cov]})
    end_time = time.time()
    time_usage = end_time - start_time
    logging.info(f"Training stage takes {time_usage} seconds to complete.")
    # Gaussian test
    for i in range(N):
        Gaussian_test(trainimagedata, traintextdata, trainlabel, i)

    start_time = time.time()
    # ML Classification
    textpreds = []
    imagepreds = []
    multipreds = []
    labels = []
    output = {}
    for _, uttid in enumerate(tqdm(list(testdict.keys()))):
        gaussianprob = []
        imagepred = torch.argmax(torch.softmax(torch.FloatTensor(testdict[uttid]['imageprob'][0]), dim=-1), dim=-1)
        textpred = torch.argmax(torch.softmax(torch.FloatTensor(testdict[uttid]['textprob'][0]), dim=-1), dim=-1)
        imageprob = testdict[uttid]['imageprob'][0]
        textprob = testdict[uttid]['textprob'][0]
        label = int(testdict[uttid]['label'][0][0])
        for i in range(N):
            gaussianprob.append(gaussiandis(Gaussian_param_dict, imageprob, textprob, i))

        multipreds.append(torch.argmax(torch.softmax(torch.FloatTensor(gaussianprob), dim=0)))
        textpreds.append(textpred)
        imagepreds.append(imagepred)
        labels.append(label)

        output.update({uttid: {}})
        output[uttid].update({'prob': torch.softmax(torch.FloatTensor(gaussianprob), dim=0).tolist()[0]})
        output[uttid].update({'predict': int(np.argmax(np.asarray(gaussianprob)))})
        output[uttid].update({'label': int(label)})

    #print(videopreds)
    textpreds = torch.LongTensor(textpreds)
    imagepreds = torch.LongTensor(imagepreds)
    multipreds = torch.LongTensor(multipreds)
    labels = torch.LongTensor(labels)

    textscore = acc(textpreds, labels)
    imagescore = acc(imagepreds, labels)
    multiscore = acc(multipreds, labels)

    with open(os.path.join(savedir, 'imageacc_' + str(imagescore) + '_textacc' + str(
            textscore) + '_multiacc' + str(multiscore) + ".json"), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    end_time = time.time()
    time_usage = end_time - start_time
    logging.info(f"Test stage takes {time_usage} seconds to complete.")

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='MGL', type=str, help='Evaluation type, possible are Delta_representation and Pairwise')
    parser.add_argument('--textmodal', default='Title', type=str, help='which data stream')
    parser.add_argument('--imagemodal', default='Image', type=str, help='single or multi images')
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
    textmodal = args.textmodal
    imagemodal = args.imagemodal
    savedir = args.savedir
    ifgaussiantest = args.ifgaussiantest
    savedir = os.path.join(savedir, 'Multimodal', modal)
    if ifgaussiantest == 'true':
        ifgaussiantest = True
    else:
        ifgaussiantest = False

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    N = 6  # the number of classes
    with open(os.path.join(datasdir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datasdir, "val.json"), encoding="utf8") as json_file:
        devdict = json.load(json_file)
    with open(os.path.join(datasdir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    prior = {}
    for i in range(N):
        prior.update({i: []})
    for workdict, dset in zip([devdict, testdict, traindict], ["dev", "test", "train"]):
        textfeatscpdir = os.path.join(args.datasdir, 'data', textmodal, dset + 'feats.scp')
        textprobscpdir = os.path.join(args.datasdir, 'data', textmodal, dset + 'prob.scp')
        imagefeatscpdir = os.path.join(args.datasdir, 'data', imagemodal, dset + 'feats.scp')
        imageprobscpdir = os.path.join(args.datasdir, 'data', imagemodal, dset + 'prob.scp')
        labelscpdir = os.path.join(args.datasdir, 'data', imagemodal, dset + 'label.scp')

        textfeat, textprob, imagefeat, imageprob, label = {}, {}, {}, {}, {}
        for scpfiles in [textfeatscpdir, textprobscpdir, imagefeatscpdir, imageprobscpdir, labelscpdir]:
            with open(scpfiles) as f:
                srcdata = f.readlines()
            datadict = {}
            for j in srcdata:
                fullname = j.split(' ')[0]
                # subnames = fullname.split('_')
                datadir = j.split(' ')[1].strip('\n')
                datadict.update({fullname: datadir})
            if scpfiles == textfeatscpdir:
                textfeat = datadict
            elif scpfiles == textprobscpdir:
                textprob = datadict
            elif scpfiles == imagefeatscpdir:
                imagefeat = datadict
            elif scpfiles == imageprobscpdir:
                imageprob = datadict
            elif scpfiles == labelscpdir:
                label = datadict

        savedict = {}
        for filename in list(workdict.keys()):
            try:
                savedict.update({filename: {}})
                savedict[filename].update({'imageprob': _get_from_loader(
                    filepath=imageprob[filename].replace('./', './../../../'), filetype='mat')}) 
                savedict[filename].update({'textprob': _get_from_loader(
                    filepath=textprob[filename].replace('./', './../../../'),
                    filetype='mat')}) 
                labeldata = _get_from_loader(filepath=label[filename].replace('./', './../../../'), filetype='mat')

                if dset == 'train':
                    prior[labeldata[0][0]].append(filename)
                savedict[filename].update(
                    {'label': labeldata})


            except:
                del savedict[filename]

        if dset == 'dev':
            devdict = savedict
        elif dset == 'test':
            testdict = savedict
        elif dset == 'train':
            traindict = savedict

    priorlist = [len(prior[i]) / len(traindict.keys()) for i in range(N)]

    #print(priorlist)
    MGL(traindict, devdict, testdict, savedir, N, ifgaussiantest)









