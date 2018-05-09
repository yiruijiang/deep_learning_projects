import utils
# import func
import pickle
import numpy as np

from func import caption_tokens_to_indices

train_img_embeds = utils.read_pickle("train_img_embeds.pickle")
train_img_fns = utils.read_pickle("train_img_fns.pickle")
val_img_embeds = utils.read_pickle("val_img_embeds.pickle")
val_img_fns = utils.read_pickle("val_img_fns.pickle")
train_captions = utils.read_pickle("train_captions.pickle")
val_captions = utils.read_pickle("val_captions.pickle")
vocab = utils.read_pickle("vocabs.pickle")

# swap the key value of vocab
vocab_inverse = {value: key for key, value in vocab.items()}

train_captions_index = np.array(caption_tokens_to_indices(train_captions, vocab))
val_captions_index = np.array(caption_tokens_to_indices(val_captions, vocab))

# train_captions = func.get_captions_for_fns(train_img_fns, "captions_train-val2014.zip", 
#                                       "annotations/captions_train2014.json")

# val_captions = func.get_captions_for_fns(val_img_fns, "captions_train-val2014.zip", 
#                                        "annotations/captions_val2014.json")

# with open("val_captions.pickle", "wb") as fn:
#     pickle.dump(val_captions, fn)
                




