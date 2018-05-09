from tensorflow.contrib import keras
K = keras.backend
import utils
import zipfile
import numpy as np
import constant

def get_cnn_encoder():
    K.set_learning_phase(False) # Sets the learning phase to a fixed value
    model = keras.applications.InceptionV3(include_top = False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input # a function
    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output)) # add another layer on top of the current model
    return model, preprocess_for_model

def load_encoder():
    # load pre-trained model
    K.clear_session()
    encoder, preprocess_for_model = get_cnn_encoder()

    # extract train features
    train_img_embeds, train_img_fns = utils.apply_model(
        "train2014.zip", encoder, preprocess_for_model, input_shape=(constant.IMG_SIZE, constant.IMG_SIZE))
    # we can download the zip from http://msvocds.blob.core.windows.net/coco2014/train2014.zip

    utils.save_pickle(train_img_embeds, "train_img_embeds.pickle")
    utils.save_pickle(train_img_fns, "train_img_fns.pickle")

    # extract validation features
    val_img_embeds, val_img_fns = utils.apply_model(
        "val2014.zip", encoder, preprocess_for_model, input_shape=(constant.IMG_SIZE, constant.IMG_SIZE))
    utils.save_pickle(val_img_embeds, "val_img_embeds.pickle")
    utils.save_pickle(val_img_fns, "val_img_fns.pickle")

    # sample images for learners
    def sample_zip(fn_in, fn_out, rate = 0.01, seed=42):
        np.random.seed(seed)
        with zipfile.ZipFile(fn_in) as fin, zipfile.ZipFile(fn_out, "w") as fout:
            sampled = filter(lambda _: np.random.rand() < rate, fin.filelist)
            for zInfo in sampled:
                fout.writestr(zInfo, fin.read(zInfo))

    sample_zip("train2014.zip", "train2014_sample.zip")
    sample_zip("val2014.zip", "val2014_sample.zip")
