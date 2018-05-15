import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import tqdm
from sklearn.datasets import load_digits
import generator
import discriminator
import data_sample

gpu_options = tf.GPUOptions(allow_growth = True, per_process_gpu_memory_fraction = 0.333)

sess = tf.InteractiveSession(config = tf.ConfigProto(gpu_options = gpu_options))

generator_model = generator.create_generator_model()

discriminator_model = discriminator.create_discriminator_model()

#define placeholder

noise = tf.placeholder(dtype = tf.float32, shape = [None, data_sample.CODE_SIZE], name = 'input_noise')

real_data = tf.placeholder(dtype = tf.float32, shape = [None,] + list(data_sample.IMG_SHAPE), name = 'input_real_img')

# the output shape of the discriminator is [batch_size, 2]
# the first entry is log-possibility of being fake and second entry is being real
logp_real = discriminator_model(real_data)

generated_data = generator_model(noise)

# the output shape of the generator is [batch_size, IMG_SHAPE]
logp_gen = discriminator_model(generated_data)

# define loss for the discriminator
# maximize log(D) and minimize log(G)
d_loss = -tf.reduce_mean(logp_real[:, 1] + logp_gen[:, 0])

# trick: regularize discriminatro output weights to prevent explosion
d_loss += tf.reduce_mean(discriminator_model.layers[-1].kernel ** 2)

# define loss for the generator
# minimize log(1 - G) -> maximize log(G)
g_loss = -tf.reduce_mean(logp_gen[:, 1])

d_optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-3).minimize(
    d_loss, var_list = discriminator_model.trainable_weights
)

g_optimizer = tf.train.AdamOptimizer(1e-4).minimize(
    g_loss, var_list = generator_model.trainable_weights
)

sess.run(tf.global_variables_initializer())

def sample_noise_batch(batch_size):
    return np.random.normal(size = (batch_size, data_sample.CODE_SIZE)).astype('float32')

def sample_data_batch(batch_size):
    idxs = np.random.choice(np.arange(data_sample.data.shape[0]), size = batch_size)
    return data_sample.data[idxs]

def sample_images(nrow, ncol, sharp = False):

    images = generator_model.predict(sample_noise_batch(batch_size = nrow * ncol))
    
    if np.var(images) != 0:
        
        images = images.clip(np.min(data_sample.data), np.max(data_sample.data))

    for i in range(nrow*ncol):

        plt.subplot(nrow,ncol,i+1)

        if sharp:
            plt.imshow(images[i].reshape(data_sample.IMG_SHAPE),cmap="gray", interpolation="none")
        else:
            plt.imshow(images[i].reshape(data_sample.IMG_SHAPE),cmap="gray")

    plt.show()

def sample_probas(bsize):
    plt.title('Generated vs real data')
    plt.hist(np.exp(discriminator_model.predict(sample_data_batch(bsize)))[:,1],
             label='D(x)', alpha=0.5,range=[0,1])
    plt.hist(np.exp(discriminator_model.predict(generator_model.predict(sample_noise_batch(bsize))))[:,1],
             label='D(G(z))',alpha=0.5,range=[0,1])
    plt.legend(loc='best')
    plt.show()


# Training Phase

for epoch in tqdm.tqdm(range(50000)):
    feed_dict = {
        real_data: sample_data_batch(100),
        noise: sample_noise_batch(100)
    }

    for i in range(5):
        sess.run(d_optimizer, feed_dict = feed_dict)
    
    sess.run(g_optimizer, feed_dict = feed_dict)

    # if epoch % 100 == 0 and epoch > 0:
    #     sample_images(2,3, True)
    #     sample_probas(1000)
#The network was trained for about 15k iterations. 
#Training for longer yields MUCH better results
plt.figure(figsize=[16,24])
sample_images(16,8)