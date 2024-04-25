import json
import argparse
from tqdm import tqdm
from kaldiio import WriteHelper
from model import *
from utils import *
from transformers import BertTokenizer, AutoTokenizer, get_linear_schedule_with_warmup, AdamW, ViTFeatureExtractor
from torchmetrics.classification import MulticlassF1Score
import random
import logging

SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def Evaluation(config, dataset=None):
    modal = config["modal"]
    modeldir = config["modeldir"]

    #model = torch.load(modeldir, map_location='cpu')
    if 'Title' in modal:
        model = TitleBERT(odim=6, MODEL=config['MODEL'], cachedir=config["cachedir"])
        padding = pad_title
    elif 'Image' in modal:
        model = ImageViT(odim=6, MODEL=config['MODEL'], cachedir=config["cachedir"])
        padding = pad_image_LRP

    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = torch.nn.DataParallel(model)
    parms = torch.load(modeldir)
    updated_parameters = {key.replace('module.module.', ''): value for key, value in parms.items()}

    model.load_state_dict(updated_parameters)
    model = model.to(config["device"])

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
    #for dset, dataloader in zip(['test'], [data_loader_test]):
    for dset, dataloader in zip(['dev', 'test', 'train'], [data_loader_dev, data_loader_test, data_loader_train]):
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
                if 'Title' in modal:
                    token_ids = data[0].to(config["device"])
                    mask = data[1].to(config["device"])
                    labels = data[2].squeeze(-1).to(config["device"])
                    filename = data[3]
                    input = [token_ids, mask]
                    outputs, feats = model(input)
                elif 'Image' in modal:
                    image = data[0].to(config["device"])
                    labels = data[1].squeeze(-1).to(config["device"])
                    filename = data[2]
                    input = image
                    outputs, feats = model(input, ifreturnfeats=True)

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
    parser.add_argument('--datasdir', default='./../../Dataset', type=str, help='Dir saves the meta information')
    parser.add_argument('--modal', default='Image', type=str, help='which data modality. It could be Text and Image')
    parser.add_argument('--modelsavedir', default='./../../unimodal_models', type=str, help='Dir to save trained model and results')
    parser.add_argument('--BS', default=32, type=float, help='The batch size')
    parser.add_argument('--cachedir', default='./../../CACHE', type=str, help='Cache dir to save the downloaded pretrained transformer model')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]


    datasetdir = args.datasdir
    modal = args.modal
    modal = modal
    BS = int(args.BS)
    modelsavedir = args.modelsavedir
    cachedir = args.cachedir

    savefeaturesdir = os.path.join(datasetdir, 'data', modal)

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'
    modeldir = os.path.join(modelsavedir, modal + '.pkl')
    max_uttlen = 500

    for makedir in [modeldir, savefeaturesdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)


    with open(os.path.join(datasetdir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datasetdir, "val.json"), encoding="utf8") as json_file:
        devdict = json.load(json_file)
    with open(os.path.join(datasetdir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    '''traindict = {k: traindict[k] for k in random.sample(list(traindict), 100)}
    devdict = {k: devdict[k] for k in random.sample(list(devdict), 100)}
    testdict = {k: testdict[k] for k in random.sample(list(testdict), 100)}'''

    if 'Title' in modal:
        MODEL = "bert-large-uncased"
        tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=cachedir)
        dataset = Titledatasetclass(train_file=traindict,
                               dev_file=devdict,
                               test_file=testdict,
                                tokenizer=tokenizer,
                               max_len=max_uttlen)
    elif 'Image' in modal:
        MODEL = 'vit_large_patch16_384.augreg_in21k_ft_in1k'
        dataset = Imagedatasetclass_LRP(train_file=traindict,
                                    dev_file=devdict,
                                    test_file=testdict)

    config = {
        "NWORKER": 0,
        "device": device,
        "batch_size": BS,
        "MODEL": MODEL,
        "modal": modal,
        "savefeaturesdir": savefeaturesdir,
        "cachedir": cachedir,
        "modeldir": modeldir,
    }
    Evaluation(config, dataset=dataset)



