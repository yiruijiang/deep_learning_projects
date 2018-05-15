
import utils
# import lfw_dataset
# import numpy as np
# data, attr = lfw_dataset.load_lfw_dataset(dimx = 36, dimy = 36)

# data = np.float32(data) / 255.

# utils.save_pickle(data, "data.pickle")
# utils.save_pickle(data, "attr.pickle")

data = utils.read_pickle("data.pickle")
attr = utils.read_pickle("attr.pickle")

IMG_SHAPE = data.shape[1:]
CODE_SIZE = 256