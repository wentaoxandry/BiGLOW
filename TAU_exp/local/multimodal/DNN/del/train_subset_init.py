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

def acc(pred, label):
    return (np.sum((pred.data.numpy() == label.data.numpy()), axis=-1) / pred.size())[0]
def find_best(filename):
    filedict = {}
    for i in filename:
        score = float(i.split('_')[-1].strip('.pkl'))
        filedict.update({i: score})
    Keymax = max(zip(filedict.values(), filedict.keys()))[1]
    return Keymax, filedict[Keymax]

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
                                test_file=testdict)  # ,
    # max_len=max_uttlen)

    #print(len(dataset.train_dataset))
    print(len(dataset.dev_dataset))
    print(len(dataset.test_dataset))

    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]

    evalacc_best = 0
    early_wait = 5
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    f1score = MulticlassF1Score(num_classes=10, average='weighted')
    #loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696,
    #                                  0.84847735, 5.42461417, 1.21859721])
    #criterion = torch.nn.CrossEntropyLoss(weight=loss_weights.to(config["device"]), ignore_index=-7)
    criterion = torch.nn.CrossEntropyLoss()

    if 'DSW' in config['modal']:
        model = DSW_classic(iadim=527, itdim=1000, hiddendim=16, odim=10)
    elif 'representation' in config['modal']:
        model = Representation_classic(iadim=527, itdim=1000, hiddendim=16, odim=10)
    #elif config['modal'] == 'representation_classic':
    #    model = Representation_classic(iadim=256, itdim=1000, hiddendim=16, odim=6)
    model = model.to(config["device"])
    model.apply(weights_init_uniform)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The model has {trainable_parameters} trainable parameters')
    '''if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)'''
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

    #data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True,
    #                                                    batch_size=config["batch_size"],
    #                                                    num_workers=config["NWORKER"],
    #                                                    collate_fn=Padding)

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
            audiofeat, audioprob, videofeat, videoprob, labels, filenames = data_all
            data = [i.to(config["device"]) for i in [audiofeat, audioprob, videofeat, videoprob]]
            labels = labels.to(config["device"]).squeeze(-1)
            optimizer.zero_grad()
            outputs, predweights = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            #if i % 10 == 0:
            #    print(predweights)
            #print("\r%f" % loss, end='')
            # print statistics
            tr_loss += loss.item()
            nb_tr_steps += 1
            prob = torch.softmax(outputs, dim=-1)
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            trainprob.append(prob)
            trainpredict.extend(predicted.cpu().detach().tolist())
            trainlabel.extend(labels.cpu().data.numpy().tolist())
        trainlabel = torch.LongTensor(trainlabel)
        trainpredict = torch.LongTensor(trainpredict)
        trainallscore = acc(trainpredict, trainlabel)
        trainallscore = float(trainallscore)
        logging.info(f'Epoch {epoch} training accuracy is {trainallscore}.')

        #trainallscore = np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1) / len(trainlabel)

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
                audiofeat, audioprob, videofeat, videoprob, labels, filenames = data_all
                data = [i.to(config["device"]) for i in [audiofeat, audioprob, videofeat, videoprob]]
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
        allscore = acc(evalpred, evallabel)
        allscore = float(allscore)
        #allscore = correct / total
        # evalacc = evalacc / len(evallabel)
        evallossmean = np.mean(np.array(evallossvec))
        logging.info(f'Epoch {epoch} evaluation loss is {evallossmean}, evaluation accuracy is {allscore}.')
        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        OUTPUT_DIR = os.path.join(modeldir,'bestmodel.pkl')

        with open(os.path.join(resultsdir, str(epoch) + '_' + str(evallossmean) + '_' + str(
                                        currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
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
            audiofeat, audioprob, videofeat, videoprob, labels, filenames = data_all
            data = [i.to(config["device"]) for i in [audiofeat, audioprob, videofeat, videoprob]]
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
    allscore = acc(testpred, testlabel)
    testacc = float(allscore)
    logging.info(f'Test accuracy is {testacc}.')
    with open(os.path.join(resultsdir, 'besttestacc_' + str(testacc)[:6] + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='representation', type=str, help='which data stream')
    parser.add_argument('--videomodal', default='Title', type=str, help='which data stream')
    parser.add_argument('--audiomodal', default='Image', type=str, help='single or multi images')
    parser.add_argument('--savedir', default='./../../../trained', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasdir = args.datasdir
    modal = args.modal
    videomodal = args.videomodal
    audiomodal = args.audiomodal
    savedir = args.savedir
    modal = modal + '_initial_parm'

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'
    N = 10  # the number of classes
    max_num_epochs = 100

    #modeldir = os.path.join(savedir, modal, 'model')
    #resultsdir = os.path.join(savedir, modal, 'results')
    #for makedir in [modeldir, resultsdir]:
    #    if not os.path.exists(makedir):
    #        os.makedirs(makedir)

    cvfolder = os.path.join(datasdir, 'subsets')
    if not os.path.exists(cvfolder):
        os.makedirs(cvfolder)
    numdir = os.listdir(cvfolder)
    if len(numdir) == 0:
        # with open(os.path.join(datasdir, "train.json"), encoding="utf8") as json_file:
        #    traindict_org = json.load(json_file)
        with open(os.path.join(datasdir, "Development.json"), encoding="utf8") as json_file:
            traindict_org = json.load(json_file)
        with open(os.path.join(datasdir, "Evaluation.json"), encoding="utf8") as json_file:
            devdict = json.load(json_file)

        labeltraindict_org = {}
        for n in range(N):
            labeltraindict_org.update({n: {}})
        for keys in list(traindict_org.keys()):
            label = traindict_org[keys]['label']
            label = str2id(label)
            labeltraindict_org[label].update({keys: traindict_org[keys]})

        traindict = deepcopy(traindict_org)
        labeltraindict = deepcopy(labeltraindict_org)
        for M in [24, 32, 64, 128, 256]:  # [4, 16, 64, 256, 1024, 2048]:
            cvsubfolder = os.path.join(cvfolder, str(M))
            if not os.path.exists(cvsubfolder):
                os.makedirs(cvsubfolder)
                i = 0
                while i < 10:  # substruct 10 subsets
                    trainsubdict_0 = {}
                    for n in range(N):
                        # print(len(list(labeltraindict[n].keys())))
                        selectkeys = random.sample(list(labeltraindict[n].keys()), 2)
                        for selectkey in selectkeys:
                            trainsubdict_0.update({selectkey: labeltraindict[n][selectkey]})
                        # del labeltraindict[n][selectkey[0]]
                    M_new = M - 2 * N
                    trainsubdict = {k: traindict[k] for k in random.sample(list(traindict.keys()), M_new)}
                    trainsubdict.update(trainsubdict_0)
                    with open(os.path.join(cvsubfolder, str(i) + ".json"), 'w', encoding='utf-8') as f:
                        json.dump(trainsubdict, f, ensure_ascii=False, indent=4)
                    i = i + 1
                    for subkeys in list(trainsubdict.keys()):
                        label = traindict[subkeys]['label']
                        label = str2id(label)
                        del labeltraindict[label][subkeys]
                        del traindict[subkeys]
            else:
                pass
    else:
        pass

    #import random
    #traindict = {k: traindict[k] for k in list(random.sample(list(traindict.keys()), 20))}
    #devdict = {k: devdict[k] for k in list(random.sample(list(devdict.keys()), 20))}
    #testdict = {k: testdict[k] for k in list(random.sample(list(testdict.keys()), 20))}

    for M in [24, 32, 64, 128, 256]:
        cvsubfolder = os.path.join(cvfolder, str(M))
        for id in range(10):
            cvsavedir = os.path.join(savedir, 'Subset', modal, str(M), str(id))
            modeldir = os.path.join(cvsavedir, 'model')
            resultsdir = os.path.join(cvsavedir, 'results')
            for makedir in [modeldir, resultsdir]:
                if not os.path.exists(makedir):
                    os.makedirs(makedir)

            with open(os.path.join(datasdir, "subsets", str(M), str(id) + ".json"), encoding="utf8") as json_file:
                devdict = json.load(json_file)
            #with open(os.path.join(datasdir, "val.json"), encoding="utf8") as json_file:
            #    devdict = json.load(json_file)
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

                videofeat, videoprob, audiofeat, audioprob, label = {}, {}, {}, {}, {}
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
                        savedict[filename].update({'videofeat': videofeat[filename]})
                        savedict[filename].update({'audioprob': audioprob[filename]})
                        savedict[filename].update({'videoprob': videoprob[filename]})  # .replace('./', './../../../')
                        savedict[filename].update({'audiofeat': audiofeat[filename]})

                        savedict[filename].update({'label': label[filename]})


                    except:
                        del savedict[filename]

                if dset == 'dev':
                    devdict = savedict
                elif dset == 'test':
                    testdict = savedict
                # elif dset == 'train':
                #    traindict = savedict



            config = {
        "testdict": testdict,
        "devdict": devdict,
        "NWORKER": 0,
        "device": device,
        "modal": modal,
        "weight_decay": 0.01,
        "lr": 7e-05,  # tune.choice([5e-2, 5e-3, 5e-4, 5e-5]),
        "batch_size": 64,
        "modeldir": modeldir,
        "resultsdir": resultsdir,
        "epochs": max_num_epochs  # tune.choice([3, 5, 10, 15])
            }
            training(config)
    #"hidden_dropout": 0.1,
    #"attention_dropout": 0.1,
    #"n_atts": 2,  # 6, #
    #"max_uttlen": 174,



