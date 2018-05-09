import generator
import zipfile
import utils
import matplotlib.pyplot as plt
import numpy as np
from constant import IMG_SIZE
from generator import generator_model, generate_caption
from preparation import val_img_fns

zf = zipfile.ZipFile("val2014_sample.zip")

def apply_model_to_image_raw_bytes(raw):
    img = utils.decode_image_from_buf(raw)
    plt.figure(figsize = (7,7))
    plt.grid('off')
    plt.axis('off')
    plt.imshow(img)
    img = utils.crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), generator_model.process_for_model)
    img = np.expand_dims(img, axis = 0)
    title = generate_caption(img)[1:-1]
    plt.title(' '.join(title))
    plt.show()

def show_valid_example(val_img_fns, example_idx = 0):
    all_files = set(val_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    apply_model_to_image_raw_bytes(zf.read(example))

def get_caption(img):
    img = utils.crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), generator_model.process_for_model)
    img = np.expand_dims(img, axis = 0)
    return generate_caption(img)[1:-1]
    

def show_valid_example_series(val_img_fns, index_array):
    all_files = set(val_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
    #plt.figure()
    index = 1
    for row in range(index_array.shape[0]):
        for col in range(index_array.shape[1]):
            fn = found_files[index_array[row,col]]
            img = utils.decode_image_from_buf(zf.read(fn))
            subplot = plt.subplot(index_array.shape[0], index_array.shape[1], index)
            subplot.axis('off')
            subplot.grid('off')
            plt.imshow(img)
            img = utils.crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), generator_model.process_for_model)
            img_s = np.expand_dims(img, axis = 0)
            title = generate_caption(img_s, sample = False)[1:-1]
            subplot.set_title(" ".join(title))    
            index += 1      
    plt.show()

sel = np.random.choice(len(zf.filelist), 9).reshape(3,3)

show_valid_example_series(val_img_fns, sel)