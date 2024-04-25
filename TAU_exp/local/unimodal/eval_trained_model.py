import json
import argparse
from tqdm import tqdm
from model import *
from utils import *
from kaldiio import WriteHelper
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import random
import logging

SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def Evaluation(config):
    modeldir = config["modeldir"]
    traindict = config["traindict"]
    devdict = config["devdict"]
    testdict = config["testdict"]

    if 'audio' in config["modal"]:
        dataset = Audiofeatdatasetclass(train_file=traindict,
                                        dev_file=devdict,
                               test_file=testdict,
                                max_len=5000)
        padding = pad_audio_sequence
        model = Audiospectrumtransformer(odim=10, cachedir=config["cashedir"])
    elif 'video' in config["modal"]:
        dataset = Videofeatdatasetclass(train_file=traindict,
                                        dev_file=devdict,
                                        test_file=testdict,
                                        cashedir=config["cashedir"])
        padding = pad_video_sequence
        model = E2EViTsingle(odim=10, cachedir=config["cashedir"])

    parms = torch.load(modeldir)
    updated_parameters = {key.replace('module.module.', ''): value for key, value in parms.items()}

    model.load_state_dict(updated_parameters)
    model = model.to(config["device"])
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The model has {trainable_parameters} trainable parameters')
    logging.info(f'Train set contains {len(dataset.train_dataset)} samples')
    logging.info(f'Dev set contains {len(dataset.dev_dataset)} samples')
    logging.info(f'Test set contains {len(dataset.test_dataset)} samples')


    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True,
                                                        batch_size=config["batch_size"],
                                                        num_workers=config["NWORKER"],
                                                        collate_fn=padding)

    data_loader_dev = torch.utils.data.DataLoader(dataset.dev_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=padding)
    data_loader_test = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                  collate_fn=padding)
    for dset, dataloader in zip(['dev', 'test', 'train'], [data_loader_dev, data_loader_test, data_loader_train]): #'dev', 'test' , data_loader_dev, data_loader_test
        probarksavedir = 'ark,scp:' + os.path.join(config["savefeaturesdir"], dset + 'prob.ark') + ',' + os.path.join(
            config["savefeaturesdir"],
            dset + 'prob.scp')
        featsarksavedir = 'ark,scp:' + os.path.join(config["savefeaturesdir"], dset + 'feats.ark') + ',' + os.path.join(
            config["savefeaturesdir"], dset + 'feats.scp')
        labelarksavedir = 'ark,scp:' + os.path.join(config["savefeaturesdir"], dset + 'label.ark') + ',' + os.path.join(
            config["savefeaturesdir"], dset + 'label.scp')
        model.eval()
        outpre = {}
        for i, data in enumerate(tqdm(dataloader), 0):
            with torch.no_grad():
                if 'audio' in config["modal"]:
                    audio_feat = data[0].to(config["device"])
                    label = data[1].to(config["device"])
                    labels = label.squeeze(-1)
                    filename = data[2]
                    outputs, feats = model(audio_feat)
                elif 'video' in config["modal"]:
                    image = data[0].to(config["device"])
                    label = data[1].to(config["device"])
                    labels = label.squeeze(-1)
                    filename = data[2]
                    outputs, feats = model(image)
                for i in range(len(filename)):
                    logit = np.expand_dims(outputs[i, :].cpu().data.numpy(), axis=0)
                    feat = np.expand_dims(feats[i, :].cpu().data.numpy(), axis=0)
                    label = np.expand_dims(np.expand_dims(labels[i].cpu().data.numpy(), axis=0), axis=0)
                    outpre.update({filename[i]: {}})
                    outpre[filename[i]].update({'logit': logit})
                    outpre[filename[i]].update({'feat': feat})
                    outpre[filename[i]].update({'label': label})
        with WriteHelper(probarksavedir, compression_method=2) as writer1:
            with WriteHelper(featsarksavedir, compression_method=2) as writer2:
                with WriteHelper(labelarksavedir, compression_method=2) as writer3:
                    for ids in tqdm(list(outpre.keys())):
                        writer1(ids, outpre[ids]['logit'])
                        writer2(ids, outpre[ids]['feat'])
                        writer3(ids, outpre[ids]['label'])



def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--featdir', default='./../../Dataset/Features', type=str, help='which data stream')
    parser.add_argument('--modelsavedir', default='./../../unimodal_models', type=str, help='which data stream')
    parser.add_argument('--modal', default='video', type=str, help='which data stream')
    parser.add_argument('--cachedir', default='./../../CACHE', type=str, help='which data stream')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.datadir
    vfeatdir = args.featdir
    modelsavedir = args.modelsavedir
    modal = args.modal
    cashedir = args.cachedir

    savefeaturesdir = os.path.join(datadir, 'data', modal)

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'
    modeldir = os.path.join(modelsavedir, modal + '.pkl')

    for makedir in [savefeaturesdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)


    with open(os.path.join(datadir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "dev.json"), encoding="utf8") as json_file:
        devdict = json.load(json_file)
    with open(os.path.join(datadir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)
    if modal == 'video':
        del devdict['airport-lyon-1095-40158']

        for workdict, dset in zip([traindict, devdict, testdict], ["train", "dev", "test"]):
            for i in list(workdict.keys()):
                videodir = os.path.join(vfeatdir, dset, i + '.pt')
                workdict[i]['videodir'] = videodir
            if dset == 'train':
                traindict = workdict
            elif dset == 'dev':
                devdict = workdict
            elif dset == "test":
                testdict = workdict

    '''traindict = {k: traindict[k] for k in random.sample(list(traindict), 20)}
    devdict = {k: devdict[k] for k in random.sample(list(devdict), 20)}'''
    #testdict = {k: testdict[k] for k in random.sample(list(testdict), 20)}

    if 'audio' in modal:
        BS = 8
    elif 'video' in modal:
        BS = 4

    config = {
            "NWORKER": 0,
            "device": device,
            "batch_size": BS,
            "savefeaturesdir": savefeaturesdir,
            "traindict": traindict,
            "devdict": devdict,
            "testdict": testdict,
            "modal": modal,
            "modeldir": modeldir,
            "cashedir": cashedir
        }
    Evaluation(config)






