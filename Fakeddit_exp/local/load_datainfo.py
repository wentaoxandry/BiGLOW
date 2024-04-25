import os, json
import multiprocessing as mp
import pandas as pd
import argparse
from tqdm import tqdm
from PIL import Image
def checkIfImageIsAvaliable(imgdir):
    if (os.path.isfile(imgdir)):
        try:
            im = Image.open(imgdir)
            return True
        except OSError:
            return False
    else:
        return False

def createdict(item):
    datadict = {}
    #print(item['id'])

    imgdir = os.path.join(imagefiledir, item['id'] + '.jpg')
    try:
        imageexist = checkIfImageIsAvaliable(imgdir)
        if imageexist is True:
            datadict.update({item['id']: {}})
            datadict[item['id']].update({'title': item['clean_title']})
            datadict[item['id']].update({'imgdir': imgdir})
            datadict[item['id']].update({'2_way_label': item['2_way_label']})
            datadict[item['id']].update({'3_way_label': item['3_way_label']})
            datadict[item['id']].update({'6_way_label': item['6_way_label']})
            metadatas = [item['num_comments'], item['score'], item['upvote_ratio']]
            datadict[item['id']].update({'meta_data': metadatas})
            try:
                comments = dict_comments_sort[item['id']]
                if len(comments) == int(item['num_comments']):
                    pass
                else:
                    print(
                        f'Checked comments and num_comments -> mismatch! len of comments found: {len(comments)}, but should be {int(item["num_comments"])} at id {item["id"]}')
            except:
                comments = None

            datadict[item['id']].update({'comments': comments})
        else:
            pass
    except:
        pass

    return datadict


def read_csv(datasourcedir, savedir):
    for dset, dsetdir in zip(['train', 'dev', 'test'], ['train_splits', 'dev_splits_complete', 'output_repeated_splits_test']):
        labelfiledir = os.path.join(datasourcedir, 'MELD.Raw', dset + '_sent_emo.csv')
        data = pd.read_csv(labelfiledir)
        dialogue_id = data['Dialogue_ID']
        utterance_id = data['Utterance_ID']
        #starttime = data['StartTime']
        #endtime = data['EndTime']
        utterance = data['Utterance']
        emotion = data['Emotion']
        datadict = {}
        for i in range(len(dialogue_id)):
            filename = 'dia' + str(dialogue_id[i]) + '_utt' + str(utterance_id[i])
            datadict.update({filename: {}})
            videodir = os.path.join(datasourcedir, 'MELD.Raw', dsetdir, filename + '.mp4')
            saveaudiofile = split_audio(videodir, datasourcedir, filename, dset)
            datadict[filename].update({'audiodir': saveaudiofile})
            #datadict[filename].update({'endtime': endtime[i]})
            datadict[filename].update({'videodir': videodir})
            datadict[filename].update({'utterance': ftfy.fix_text(utterance[i])})
            datadict[filename].update({'emotion': emotion[i]})
        with open(os.path.join(savedir, dset + ".json"), 'w', encoding='utf-8') as f:
            json.dump(datadict, f, ensure_ascii=False, indent=4)

def product_helper(args):
    return createdict(*args)

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasourcedir', default='./../Sourcedata', type=str, help='Dir saves the Fakeddit dataset')
    parser.add_argument('--savedir', default='./../Dataset', type=str, help='Dir saves metainformation')
    parser.add_argument('--ifmulticore', default=True, type=bool, help='If use multi processor to faster the process')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #### This script creats the dataset meta information in JSON files.
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasourcedir = args.datasourcedir
    savedir = args.savedir
    ifmulticore = args.ifmulticore

    textfiledir = os.path.join(datasourcedir, 'downloaded', 'labels')
    global imagefiledir
    imagefiledir = os.path.join(datasourcedir, 'public_image_set')

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    ## load data information ##
    # Path to comments TSV
    path_to_comments_tsv = os.path.join(textfiledir,  "all_comments.tsv")
    print("Path to all comments is: " + path_to_comments_tsv)

    # Path to train data
    path_to_train_tsv = os.path.join(textfiledir, "multimodal_train.tsv")
    print("Path to train.tsv is: " + path_to_train_tsv)


    # Path to test data
    path_to_test_tsv = os.path.join(textfiledir, "multimodal_test_public.tsv")
    print("Path to test.tsv is: " + path_to_test_tsv)

    # Path to val data
    path_to_val_tsv = os.path.join(textfiledir, "multimodal_validate.tsv")
    print("Path to val.tsv is: " + path_to_val_tsv)

    global dict_comments_sort
    if os.path.exists(os.path.join(savedir, "all_comments.json")):
        print('comments already processed, loading...')
        with open(os.path.join(savedir, "all_comments.json"), encoding="utf8") as json_file:
            dict_comments_sort = json.load(json_file)
    else:
        dict_comments_sort = {}
        df_all_comments = pd.read_csv(path_to_comments_tsv, header=0, sep='\t')
        dict_comments = df_all_comments.to_dict('records')
        print('processing all comments file')
        for i in tqdm(dict_comments):
            if i['submission_id'] in list(dict_comments_sort.keys()):
                pass
            else:
                dict_comments_sort.update({i['submission_id']: []})
            dict_comments_sort[i['submission_id']].append(i['body'])
        print('finished all comments file')
        with open(os.path.join(savedir, "all_comments.json"), 'w', encoding='utf-8') as f:
            json.dump(dict_comments_sort, f, ensure_ascii=False, indent=4)


    # Excerpt from train set
    df_train_original = pd.read_csv(path_to_train_tsv, header=0, sep='\t')
    df_train_original = df_train_original.loc[:, ~df_train_original.columns.str.contains('^Unnamed')]

    # Excerpt from test set
    df_test_original = pd.read_csv(path_to_test_tsv, header=0, sep='\t')
    df_test_original = df_test_original.loc[:, ~df_test_original.columns.str.contains('^Unnamed')]

    # Excerpt from val set
    df_val_original = pd.read_csv(path_to_val_tsv, header=0, sep='\t')
    df_val_original = df_val_original.loc[:, ~df_val_original.columns.str.contains('^Unnamed')]
    df_val_original['title'] = df_val_original['title'].astype(str)

    for dsetdata, dset in zip([df_val_original, df_test_original, df_train_original], ['val', 'test', 'train']):
        dict_data = dsetdata.to_dict('records')
        import random
        dict_data = list(random.sample(dict_data, 400))
        savedict = {}
        results = []
        if ifmulticore is True:
            pool = mp.Pool()
            results.extend(pool.map(createdict, dict_data))
        else:
            for i in dict_data:
                results.append(createdict(i))
        newresults = []
        newresults = [i for i in results if i != {}]
        for i in newresults:
            savedict.update(i)
        with open(os.path.join(savedir, dset + '.json'), "w", encoding="utf-8") as f:
            json.dump(savedict, f, ensure_ascii=False, indent=4)






