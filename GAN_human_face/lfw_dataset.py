# labeld face in the wild

import numpy as np
import os
import cv2
import pandas as pd
import tarfile
import tqdm

ATTRS_NAME = "lfw_attributes.txt"
IMAGES_NAME = "lfw-deepfunneled.tgz"
RAW_IMAGES_NAME = "lfw.tgz"

def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype = np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_lfw_dataset(
    use_raw = False,
    dx = 80, dy = 80,
    dimx = 45, dimy = 45):

    # read attr
    df_attrs = pd.read_csv(ATTRS_NAME, sep = '\t', skiprows = 1)
    df_attrs = pd.DataFrame(df_attrs.iloc[:,:-1].values, columns = df_attrs.columns[1:])    
    imgs_with_attrs = set(map(tuple, df_attrs[["person", "imagenum"]].values))

    # read photos
    all_photos = []
    photo_ids = []

    with tarfile.open(RAW_IMAGES_NAME if use_raw else IMAGES_NAME) as f:
        for m in tqdm.tqdm(f.getmembers()):
            if m.isfile() and m.name.endswith(".jpg"):
                # crop and resize image
                img = decode_image_from_raw_bytes(f.extractfile(m).read())
                img = img[dy:-dy, dx:-dx]
                img = cv2.resize(img, (dimx, dimy))
                # parse person
                fname = os.path.split(m.name)[-1]
                # this code is not clean..., I take it anyway
                fname_splitted = fname[:-4].replace('_',' ').split()
                person_id = ' '.join(fname_splitted[:-1])
                photo_number = int(fname_splitted[-1])
                if (person_id, photo_number) in imgs_with_attrs:
                    all_photos.append(img)
                    photo_ids.append({'person': person_id,
                    'imagenum':photo_number})

    # person and imagenum become column
    # pd.DataFrame take list of dicts, and take key values as columns
    photo_ids = pd.DataFrame(photo_ids)
    all_photos = np.stack(all_photos).astype('uint8')

    # preserve photo_ids order
    # default is left_on so df_attrs order is preserved
    # drop name and number of images
    all_attrs = photo_ids.merge(df_attrs, on = ('person', 'imagenum')).drop(["person", "imagenum"], axis = 1)
    return all_photos, all_attrs

