import os, sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from model import *
from utils import *
from transformers import get_linear_schedule_with_warmup, AdamW
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

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)
def training(config):
    logging.basicConfig(filename=os.path.join(config["modeldir"], 'train.log'), level=logging.INFO)
    #traindict = config["traindict"]
    devdict = config["devdict"]
    testdict = config["testdict"]

    dataset = Loaddatasetclass(dev_file=devdict,
                                test_file=testdict)  

    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]

    evalacc_best = 0
    early_wait = 10
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    f1score = MulticlassF1Score(num_classes=6, average='weighted')
    criterion = torch.nn.CrossEntropyLoss()

    if 'DSW' in config['modal']:
        model = DSW(iadim=1024, itdim=1024, hiddendim=16, odim=6)
    elif 'representation' in config['modal']:
        model = Representation(iadim=1024, itdim=1024, hiddendim=16, odim=6)
    model = model.to(config["device"])
    model.apply(weights_init_uniform)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The model has {trainable_parameters} trainable parameters')
    logging.info(f'Dev set contains {len(dataset.dev_dataset)} samples')
    logging.info(f'Test set contains {len(dataset.test_dataset)} samples')
    optimizer = AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=config["weight_decay"], no_deprecation_warning=True
                                  )
    logging.info(f'Dev set contains {len(dataset.dev_dataset)} samples')
    logging.info(f'Test set contains {len(dataset.test_dataset)} samples')
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    len(dataset.dev_dataset) / config["batch_size"]) * 10,
                                                num_training_steps=int(
                                                    len(dataset.dev_dataset) / config["batch_size"]) * config["epochs"])

    data_loader_dev = torch.utils.data.DataLoader(dataset.dev_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=Padding)
    data_loader_test = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=Padding)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        trainpredict = []
        trainprob = []
        trainlabel = []
        for i, data_all in enumerate(tqdm(data_loader_dev), 0):
            imagefeat, imageprob, textfeat, textprob, labels, filenames = data_all
            data = [i.to(config["device"]) for i in [imagefeat, imageprob, textfeat, textprob]]
            labels = labels.to(config["device"]).squeeze(-1)
            optimizer.zero_grad()
            outputs, predweights = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            nb_tr_steps += 1
            prob = torch.softmax(outputs, dim=-1)
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            trainprob.append(prob)
            trainpredict.extend(predicted.cpu().detach().tolist())
            trainlabel.extend(labels.cpu().data.numpy().tolist())
        trainlabel = torch.LongTensor(trainlabel)
        trainpredict = torch.LongTensor(trainpredict)
        trainallscore = f1score(trainpredict, trainlabel).cpu().data.numpy()
        trainallscore = float(trainallscore)
        trainallaccscore = (sum(trainpredict == trainlabel) / trainlabel.size(0)).cpu().data.numpy()
        trainallaccscore = float(trainallaccscore)
        logging.info(f'Epoch {epoch} training accuracy score is {trainallaccscore}.')


        # Validation loss
        torch.cuda.empty_cache()
        evallossvec = []
        evalpred = []
        evallabel = []
        model.eval()
        outpre = {}
        total = 0
        for i, data_all in enumerate(tqdm(data_loader_test), 0):
            with torch.no_grad():
                imagefeat, imageprob, textfeat, textprob, labels, filenames = data_all
                data = [i.to(config["device"]) for i in [imagefeat, imageprob, textfeat, textprob]]
                labels = labels.to(config["device"]).squeeze(-1)
                outputs, predweights = model(data)
                dev_loss = criterion(outputs, labels)
                #if i % 10 == 0:
                #    print(predweights)

                evallossvec.append(dev_loss.cpu().data.numpy())
                predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
                evallabel.extend(labels.cpu().data.numpy().tolist())
                evalpred.extend(predicted.cpu().detach().tolist())
                total += labels.size(0)
                for i in range(len(filenames)):
                    outpre.update({filenames[i]: {}})
                    outpre[filenames[i]].update({'label': int(labels[i].cpu().detach())})
                    outpre[filenames[i]].update({'predict': int(predicted[i].cpu().detach())})
                    outpre[filenames[i]].update({'prob': prob[i]})

        evalpred = torch.LongTensor(evalpred)
        evallabel = torch.LongTensor(evallabel)
        allscore = f1score(evalpred, evallabel).cpu().data.numpy()
        allscore = float(allscore)
        evalallaccscore = (sum(evalpred == evallabel) / evallabel.size(0)).cpu().data.numpy()
        evalallaccscore = float(evalallaccscore)
        evallossmean = np.mean(np.array(evallossvec))
        logging.info(f'Epoch {epoch} evaluation loss is {evallossmean}, evaluation accuracy score is {evalallaccscore}.')

        OUTPUT_DIR = os.path.join(modeldir,'bestmodel.pkl')

        with open(os.path.join(resultsdir, str(epoch) + '_' + str(evallossmean) + '_' + str(trainallscore)[:6] + '_' + str(
                                        allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(outpre, f, ensure_ascii=False, indent=4)


        torch.cuda.empty_cache()
        if allscore <= evalacc_best:
            stop_counter = stop_counter + 1
            print('no improvement')
            continuescore = 0
        else:
            print('new score')
            evalacc_best = allscore
            continuescore = continuescore + 1
            torch.save(model, OUTPUT_DIR)

        if continuescore >= run_wait:
            stop_counter = 0
        print(stop_counter)
        print(early_wait)
        if stop_counter < early_wait:
            pass
        else:
            break


    model = torch.load(os.path.join(modeldir,'bestmodel.pkl'), map_location=config["device"])
    testpred = []
    testlabel = []
    model.eval()
    outpre = {}
    total = 0
    for i, data_all in enumerate(tqdm(data_loader_test), 0):
        with torch.no_grad():
            imagefeat, imageprob, textfeat, textprob, labels, filenames = data_all
            data = [i.to(config["device"]) for i in [imagefeat, imageprob, textfeat, textprob]]
            labels = labels.to(config["device"]).squeeze(-1)
            outputs, _ = model(data)

            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
            testlabel.extend(labels.cpu().data.numpy().tolist())
            testpred.extend(predicted.cpu().detach().tolist())
            total += labels.size(0)
            for i in range(len(filenames)):
                outpre.update({filenames[i]: {}})
                outpre[filenames[i]].update({'label': int(labels[i].cpu().detach())})
                outpre[filenames[i]].update({'predict': int(predicted[i].cpu().detach())})
                outpre[filenames[i]].update({'prob': prob[i]})

    testpred = torch.LongTensor(testpred)
    testlabel = torch.LongTensor(testlabel)
    allscore = f1score(testpred, testlabel).cpu().data.numpy()
    testallaccscore = (sum(testpred == testlabel) / testlabel.size(0)).cpu().data.numpy()
    testallaccscore = float(testallaccscore)
    logging.info(f'Test accuracy score is {testallaccscore}.')
    with open(os.path.join(resultsdir, 'besttestacc_' + str(testallaccscore)[:6] + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='representation', type=str, help='which data stream')
    parser.add_argument('--textmodal', default='Title', type=str, help='which data stream')
    parser.add_argument('--imagemodal', default='Image', type=str, help='single or multi images')
    parser.add_argument('--savedir', default='./../../../trained', type=str, help='Dir to save trained model and results')
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

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'
    N = 6  # the number of classes
    max_num_epochs = 100

    cvfolder = os.path.join(datasdir, 'cvsplit')
    if not os.path.exists(cvfolder):
        os.makedirs(cvfolder)
    numdir = os.listdir(cvfolder)
    if len(numdir) == 0:
        with open(os.path.join(datasdir, "val.json"), encoding="utf8") as json_file:
            devdict_org = json.load(json_file)
        with open(os.path.join(datasdir, "test.json"), encoding="utf8") as json_file:
            testdict = json.load(json_file)

        labeldevdict_org = {}
        for n in range(N):
            labeldevdict_org.update({n: {}})
        for keys in list(devdict_org.keys()):
            label = devdict_org[keys]['6_way_label']
            labeldevdict_org[label].update({keys: devdict_org[keys]})

        for M in [18, 32, 64, 128, 256, 512]:  
            devdict = deepcopy(devdict_org)
            labeldevdict = deepcopy(labeldevdict_org)
            cvsubfolder = os.path.join(cvfolder, str(M))
            if not os.path.exists(cvsubfolder):
                os.makedirs(cvsubfolder)
                i = 0
                while i < 10:  # substruct 10 subsets
                    devsubdict_0 = {}
                    for n in range(N):
                        # print(len(list(labeltraindict[n].keys())))
                        selectkeys = random.sample(list(labeldevdict[n].keys()), 3)
                        for selectkey in selectkeys:
                            devsubdict_0.update({selectkey: labeldevdict[n][selectkey]})
                        # del labeltraindict[n][selectkey[0]]
                    M_new = M - 3 * N
                    devsubdict = {k: devdict[k] for k in random.sample(list(devdict.keys()), M_new)}
                    devsubdict.update(devsubdict_0)
                    filelist = list(devsubdict.keys())
                    with open(os.path.join(cvsubfolder, str(i) + ".txt"), 'w') as file:
                        for item in filelist:
                            file.write(str(item) + '\n')  # Convert item to string if necessary
                    i = i + 1
                    for subkeys in list(devsubdict.keys()):
                        label = devdict[subkeys]['6_way_label']
                        del labeldevdict[label][subkeys]
                        del devdict[subkeys]
            else:
                pass
    else:
        pass


    for M in [18, 32, 64, 128, 256, 512]:
        cvsubfolder = os.path.join(cvfolder, str(M))
        for id in range(10):
            cvsavedir = os.path.join(savedir, 'Subset', modal, str(M), str(id))
            modeldir = os.path.join(cvsavedir, 'model')
            resultsdir = os.path.join(cvsavedir, 'results')
            for makedir in [modeldir, resultsdir]:
                if not os.path.exists(makedir):
                    os.makedirs(makedir)

            with open(os.path.join(datasdir, "cvsplit", str(M), str(id) + ".txt"), 'r') as file:
                # Read all lines from the file into a list
                lines = file.readlines()
            with open(os.path.join(datasdir, "val.json"), encoding="utf8") as json_file:
                devorgdict = json.load(json_file)
            with open(os.path.join(datasdir, "test.json"), encoding="utf8") as json_file:
                testdict = json.load(json_file)
            devdict = {}
            for i in lines:
                i = i.strip('\n')
                devdict.update({i: devorgdict[i]})

            prior = {}
            for i in range(N):
                prior.update({i: []})
            for workdict, dset in zip([devdict, testdict], ["dev", "test"]):
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
                        savedict[filename].update({'textfeat': textfeat[filename]})
                        savedict[filename].update({'imageprob': imageprob[filename]})
                        savedict[filename].update({'textprob': textprob[filename]})  
                        savedict[filename].update({'imagefeat': imagefeat[filename]})

                        savedict[filename].update({'label': label[filename]})


                    except:
                        del savedict[filename]

                if dset == 'dev':
                    devdict = savedict
                elif dset == 'test':
                    testdict = savedict



            config = {
        "testdict": testdict,
        "devdict": devdict,
        "NWORKER": 0,
        "device": device,
        "modal": modal,
        "weight_decay": 0.01,
        "lr": 7e-05,  
        "batch_size": 64,
        "modeldir": modeldir,
        "resultsdir": resultsdir,
        "epochs": max_num_epochs 
            }
            training(config)




