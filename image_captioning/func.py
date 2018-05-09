import json
import zipfile
import re

import constant
import utils

import matplotlib.pyplot as plt
import numpy as np

from functools import reduce
from collections import Counter,defaultdict

PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"

# .json is available from https://github.com/pveerina/imgcap/tree/master/data/mscoco

def get_captions_for_fns(fns, zip_fn, zip_json_path):
    """
    get image captions based on the file names
    """
    zf = zipfile.ZipFile(zip_fn)
    j = json.loads(zf.read(zip_json_path).decode("utf8"))
    id_to_fn = {img["id"]: img["file_name"] for img in j["images"]} # extract info from json file and create a dict
    #test = {cap["image_id"]: cap["caption"] for cap in j["annotations"]}
    fn_to_caps = defaultdict(list) # we need the default factory method
    for cap in j['annotations']:
        fn_to_caps[id_to_fn[cap['image_id']]].append(cap['caption'])
    # create fn -> caption map
    fn_to_caps = dict(fn_to_caps)
    return list(map(lambda x: fn_to_caps[x], fns))

def show_training_example(train_img_fns, train_captions, example_idx = 0):
    """
    showing images with their image captions
    """
    zf = zipfile.ZipFile("train2014_sample.zip")
    captions_by_file = dict(zip(train_img_fns, train_captions))
    all_files = set(train_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist)) # the last word i.e. file name
    example = found_files[example_idx]
    img = utils.decode_image_from_buf(zf.read(example)) # example is ZipInfo necessary for zf.read() function
    plt.imshow(utils.image_center_crop(img))
    plt.title("\n".join(captions_by_file[example.filename.rsplit("/")[-1]]))
    plt.show()

def split_sentence(sentence):
    """
    split sentence into words
    """
    return list(filter(lambda x: len(x) > 0, re.split("\W+", sentence.lower())))

def generate_vocabulary(train_captions):
    # r = list(reduce(lambda x, y: x + y, train_captions))
    # s = list(map(split_sentence, r))
    # t = list(reduce(lambda x, y: x + y, s))
    # u = dict(filter(lambda x: x[1] >= 5, Counter(t).items()))
    # the algorithm above turns out to be super slow, sad story, go for numpy

    v = []
    for i in train_captions:
        for j in i:
            v += split_sentence(j)

    vocab, c = np.unique(np.array(v), return_counts = True)
    vocab = vocab[np.where(c >= 5)]

    vocab = np.hstack((vocab, [PAD, UNK, START, END]))
    return {token: index for index, token in enumerate(sorted(vocab))}

def caption_tokens_to_indices(caption, vocab):
    """
    caption shall be an list of lists:
    [
        [
            "image1 caption1",
            "image2 caption2",
            ...
        ],
        [
            "image2 caption1",
            "image2 caption2",
            ...
        ],
        ...
    ]
    Replace all tokens with vocabulary indices, use UNK for unknown words (out of vocabulary).
    Add START and END tokens to start and end of each sentence respectively.
    The result is:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    """
    res = []
    for blocks in caption:
        block = []
        for sentence in blocks:
            s = []
            s.append(vocab[START])
            for word in split_sentence(sentence):
                if word in vocab.keys():
                    s.append(vocab[word])
                else:
                    s.append(vocab[UNK])
            s.append(vocab[END])
            block.append(s)
        res.append(block)
    return res

def batch_captions_to_matrix(batch_captions, pad_idx, max_len = None):
    """
    `batch_captions` is an array of arrays:
    [
        [vocab[START], ..., vocab[END]],
        [vocab[START], ..., vocab[END]],
        ...
    ]
    Add padding with pad_idx where necessary.
    Input example: [[1, 2, 3], [4, 5]]
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=None
    Output example: np.array([[1, 2], [4, 5]]) if max_len=2
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=100
    """ 
    cols = None
    matrix = None

    if max_len is None:
        cols = max(map(len, batch_captions))
    else:
        cols = min(max_len, max(map(len, batch_captions)))
        
    matrix = np.ones((len(batch_captions), cols)) * pad_idx
    for i in range(len(batch_captions)):
        l = min(len(batch_captions[i]), cols)
        matrix[i,:l] = batch_captions[i][:l]
    return matrix
