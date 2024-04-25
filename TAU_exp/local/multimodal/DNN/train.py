import json
import argparse
from tqdm import tqdm
from model import *
import time
from utils import *
from transformers import AdamW
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

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)
def training(config):
    logging.basicConfig(filename=os.path.join(config["modeldir"], 'train.log'), level=logging.INFO)
    traindict = config["traindict"]
    devict = config["devdict"]

    dataset = Loaddatasetclass(dev_file=traindict,
                                test_file=devict)  # ,

    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]

    evalacc_best = 0
    early_wait = 5
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    criterion = torch.nn.CrossEntropyLoss()

    if 'DSW' in config['modal']:
        model = DSW(iadim=527, itdim=1000, hiddendim=16, odim=10)
    elif 'representation' in config['modal']:
        model = Representation(iadim=527, itdim=1000, hiddendim=16, odim=10)
    model.apply(weights_init_uniform)
    model = model.to(config["device"])
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The model has {trainable_parameters} trainable parameters')
    logging.info(f'Dev set contains {len(dataset.dev_dataset)} samples')
    logging.info(f'Test set contains {len(dataset.test_dataset)} samples')

    optimizer = AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=config["weight_decay"], no_deprecation_warning=True
                                  )

    data_loader_dev = torch.utils.data.DataLoader(dataset.dev_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=Padding)
    data_loader_test = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=Padding)
    start_time = time.time()
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
        evallossmean = np.mean(np.array(evallossvec))
        logging.info(f'Epoch {epoch} evaluation loss is {evallossmean}, evaluation accuracy is {allscore}.')
        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        OUTPUT_DIR = os.path.join(modeldir, 'bestmodel.pkl')

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

    end_time = time.time()
    time_usage = end_time - start_time
    logging.info(f'Training stage takes {time_usage} seconds to complete.')
    start_time = time.time()

    model = torch.load(os.path.join(modeldir, 'bestmodel.pkl'), map_location=config["device"])
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
    end_time = time.time()
    time_usage = end_time - start_time
    logging.info(f'Testing stage takes {time_usage} seconds to complete.')

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../../Dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='representation', type=str, help='which data stream')
    parser.add_argument('--videomodal', default='video', type=str, help='which data stream')
    parser.add_argument('--audiomodal', default='audio', type=str, help='single or multi images')
    parser.add_argument('--savedir', default='./../../../trained', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasetdir = args.datasdir
    modal = args.modal
    videomodal = args.videomodal
    audiomodal = args.audiomodal
    savedir = args.savedir

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    modeldir = os.path.join(savedir, 'Multimodal', modal, 'model')
    resultsdir = os.path.join(savedir, 'Multimodal', modal, 'results')

    max_num_epochs = 100

    for makedir in [modeldir, resultsdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)


    with open(os.path.join(datasetdir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datasetdir, "dev.json"), encoding="utf8") as json_file:
        devdict = json.load(json_file)
    with open(os.path.join(datasetdir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    for workdict, dset in zip([devdict, testdict], ["dev", "test"]):
        videofeatscpdir = os.path.join(datasetdir, 'data', videomodal, dset + 'feats.scp')
        videoprobscpdir = os.path.join(datasetdir, 'data', videomodal, dset + 'prob.scp')
        audiofeatscpdir = os.path.join(datasetdir, 'data', audiomodal, dset + 'feats.scp')
        audioprobscpdir = os.path.join(datasetdir, 'data', audiomodal, dset + 'prob.scp')
        labelscpdir = os.path.join(datasetdir, 'data', audiomodal, dset + 'label.scp')

        videofeat, videoprob, audiofeat, audioprob, label = {}, {}, {}, {}, {}
        for scpfiles in [videofeatscpdir, videoprobscpdir, audiofeatscpdir, audioprobscpdir, labelscpdir]:
            with open(scpfiles) as f:
                srcdata = f.readlines()
            datadict = {}
            for j in srcdata:
                fullname = j.split(' ')[0]
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
                savedict[filename].update({'videoprob': videoprob[filename]})
                savedict[filename].update({'audiofeat': audiofeat[filename]})

                savedict[filename].update({'label': label[filename]})


            except:
                del savedict[filename]

        if dset == 'dev':
            devdict = savedict
        elif dset == 'test':
            testdict = savedict
        elif dset == 'train':
            traindict = savedict

    config = {
        "traindict": devdict,
        "devdict": testdict,
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




