import os
import queue
import threading
import zipfile
from tqdm import tqdm
import cv2
import numpy as np
import pickle

# Note: python supports dynamic type inference
def image_center_crop(img):
    """
    crop the image so that it has square size
    """
    h, w = img.shape[0], img.shape[1]
    pad_left = pad_right = pad_top = pad_bottom = 0
    diff = abs(h - w)
    half_diff = diff // 2
    if h > w:
        pad_top = diff - half_diff
        pad_bottom = half_diff
    else:
        pad_left = diff - half_diff
        pad_right = half_diff
    return img[pad_top : h - pad_bottom, pad_left : h - pad_right]

def decode_image_from_buf(buf):
    """
    decode the img from raw data
    """
    img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def crop_and_preprocess(img, input_shape, preprocess_for_model):
    """
    centralize image and preprocess
    """
    img = image_center_crop(img)
    img = cv2.resize(img, input_shape)
    img = img.astype("float32")
    img = preprocess_for_model(img)
    return img

def apply_model(zip_fn, model, preprocess_for_model, extensions = (".jpg",), input_shape = (224, 224), batch_size = 32):

    # queue for cropped images
    q = queue.Queue(maxsize = batch_size * 10) 

    # when read thread put all images in queue
    read_thread_completed = threading.Event()

    # kill read thread
    kill_read_thread = threading.Event()

    # fn is file name
    def reading_thread(zip_fn):

        zf = zipfile.ZipFile(zip_fn)
        for fn in tqdm(zf.namelist()): # every file in the zip file
            if kill_read_thread.is_set():
                break
            if os.path.splitext(fn)[-1] in extensions: #get extension from path
                buf = zf.read(fn)
                img = decode_image_from_buf(buf)
                img = crop_and_preprocess(img, input_shape, preprocess_for_model)
                while True:
                    try:
                        q.put((os.path.split(fn)[-1], img), timeout = 1) # similar to unordered_map< string, img> in c++
                    except:
                        if kill_read_thread.is_set():
                            break
                        continue # dont care if the img is not put into the queue, we have enough images already
                    break
            
        read_thread_completed.set() # all images have been read
    
    t = threading.Thread(target = reading_thread, args = (zip_fn, )) # similar to thread(std::bind(&X::Func, args, )) in c++
    t.daemon = True # similar to thread.detach in c++
    t.start()

    img_fns = [] # image file names
    img_embeddings = [] # image embedded features

    batch_imgs = [] # batch images

    def process_batch(batch_imgs):
        batch_imgs = np.stack(batch_imgs, axis = 0) # convert list to array of shape [batch_size, image_shape]
        batch_embeddings = model.predict(batch_imgs) # compute the feature out of batch images
        img_embeddings.append(batch_embeddings)
    
    try:
        while True:
            try:
                fn, img = q.get(timeout = 1)
            except queue.Empty:
                if read_thread_completed.is_set():
                    break
                continue # wait until the new image comes into the queue
            img_fns.append(fn)
            batch_imgs.append(img)
            if len(batch_imgs) == batch_size: # else we dont care
                process_batch(batch_imgs)
                batch_imgs = [] # clear buffer
            q.task_done() # tells the queue that processing on the task is complete

        #process last batch
        if len(batch_imgs):
            process_batch(batch_imgs)
    finally:
        kill_read_thread.set() # now it is time to stop the 
        t.join() # block the reading image thread till it is finished
    
    q.join() #blocks untill all items in the queue have been gotten and processed

    img_embeddings = np.vstack(img_embeddings)# convert from list to array of shape [batch_size, img_embed_size]

    return img_embeddings, img_fns

def save_pickle(obj, fn):
    with open(fn, "wb") as f:
        pickle.dump(obj, f, protocol = pickle.HIGHEST_PROTOCOL)

def read_pickle(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)






