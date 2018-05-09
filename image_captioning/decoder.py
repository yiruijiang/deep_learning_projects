import sys
import os
sys.path.append("..")
import tensorflow as tf
import numpy as np
import func
import tqdm
from tensorflow.contrib import keras
from preparation import train_img_embeds,train_captions,train_captions_index, \
                        val_img_embeds,val_captions, val_captions_index, \
                        vocab
from hyperparameters \
import IMG_EMBED_BOTTLENECK, WORD_EMBED_SIZE, LSTM_UNITS, LOGIT_BOTTLENECK ,\
MAX_LEN, n_epoch, n_training_batch_per_epoch, n_validation_batches, batch_size

from constant import PAD

L = keras.layers
K = keras.backend

IMG_SIZE = 299
IMG_EMBED_SIZE = train_img_embeds.shape[1]

tf.reset_default_graph()
# tf.set_random_seed(42)

s = tf.InteractiveSession()

class decoder:

    # #[Batch_size, IMG_EMBED_SIZE] of CNN image features
    img_embeds = tf.placeholder('float32', [None, IMG_EMBED_SIZE])

    # #[Batch_size, time steps] fo word ids
    sentences = tf.placeholder('int32', [None, None])

    # # we use bottleneck here to reduce the number of parameters
    # # image embedding -> bottleneck
    img_embed_to_bottleneck_layer = L.Dense(IMG_EMBED_BOTTLENECK, input_shape = \
    (None, IMG_EMBED_SIZE), activation = 'elu')

    # # image embedding bottleneck -> lstm initial state
    img_embed_bottleneck_to_initialize_state_layer = L.Dense(LSTM_UNITS, input_shape = \
    (None, IMG_EMBED_BOTTLENECK), activation = 'elu')

    # # word -> embedding
    word_embed_layer = L.Embedding(len(vocab), WORD_EMBED_SIZE)

    # # lstm cell (tensorflow)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)

    # # we use bottleneck here to reduce model complexity
    # # lstm output -> logits bottleneck
    token_logits_bottleneck_layer = L.Dense(LOGIT_BOTTLENECK, input_shape = \
    (None, LSTM_UNITS), activation = 'elu')

    # # logits bottleneck -> logits for next token prediction
    # # final layer before softmax layer
    token_logits_layer = L.Dense(len(vocab), input_shape = (None, LOGIT_BOTTLENECK))

    # # img_embeds are features from the encoder
    img_embed_to_bottleneck = img_embed_to_bottleneck_layer(img_embeds)

    # # initial lstm cell state of shape (None, LSTM)
    c0 = h0 = img_embed_bottleneck_to_initialize_state_layer(img_embed_to_bottleneck)

    # # embed all tokens but the last for lstm output
    # # we give sentence till the last word
    word_embeds = word_embed_layer(sentences[:, :-1])

    # # during training we use ground truth tokens 'word_embeds' as context for next token predictions
    # # hidden_states have shape of [batch_size, max_time, cell.output_size]
    hidden_states, _ = tf.nn.dynamic_rnn(lstm_cell, word_embeds, \
    initial_state = tf.nn.rnn_cell.LSTMStateTuple(c0, h0))

    # # now we need to calculate token logits for all the hidden states
    flat_hidden_states = tf.reshape(hidden_states, shape = (-1, LSTM_UNITS))

    # # output from lstm to bottleneck layer 
    token_logits_bottleneck = token_logits_bottleneck_layer(flat_hidden_states)

    # # bottleneck layer to final layer before prediction
    # # token_logits has shape of [time_step, len(vocab)]
    token_logits = token_logits_layer(token_logits_bottleneck)

    # # define ground truth, we use current token to predict the next
    ground_truth = tf.reshape(
        tf.where(sentences[:, 1:] != vocab[PAD], sentences[:, 1:],\
        tf.zeros_like(sentences[:, 1:])), \
        shape = [-1]
    )
    
    # # calculate prediction via softmax
    # # softmax will be performed on the logits
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits( \
        labels = ground_truth, \
        logits = token_logits \
        )

    # # define loss
    loss = tf.reduce_sum(losses) / tf.count_nonzero(ground_truth, dtype = tf.float32)

optimizer = tf.train.AdamOptimizer()

train_step = optimizer.minimize(decoder.loss)

saver = tf.train.Saver()

s.run(tf.global_variables_initializer())


def generate_batch(image_embeddings, indexed_captions, batch_size, max_len = None):
    """
    `images_embeddings` is a np.array of shape [number of images, IMG_EMBED_SIZE].
    `indexed_captions` holds 5 vocabulary indexed captions for each image:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    Generate a random batch of size `batch_size`.
    Take random images and choose one random caption for each image.
    Remember to use `batch_captions_to_matrix` for padding and respect `max_len` parameter.
    Return feed dict {decoder.img_embeds: ..., decoder.sentences: ...}.
    """
    select = np.random.choice(len(image_embeddings), batch_size)
    img = image_embeddings[select]
    c = np.array(indexed_captions)[select]
    caps = []
    for block in c:
        sel = np.random.choice(len(block))
        caps.append(block[sel])

    caps = func.batch_captions_to_matrix(caps, vocab[PAD], max_len)
    return {decoder.img_embeds: img, decoder.sentences: caps}

last_epoch = n_epoch
# saver.restore(s, os.path.abspath("weights_{}".format(last_epoch)))

for epoch in range(last_epoch, n_epoch):

    print("epoch: {}".format(epoch))
    pbar = tqdm.tqdm(range(n_training_batch_per_epoch))
    count = 0

    train_loss = 0
    for _ in pbar:

        train_loss += s.run([decoder.loss, train_step], \
                            generate_batch(train_img_embeds, \
                                           train_captions_index, \
                                           batch_size, \
                                           MAX_LEN))[0]
        count += 1
        pbar.set_description("Training loss: %f" % (train_loss / count))

 
    train_loss /= n_training_batch_per_epoch

    val_loss = 0
    for _ in range(n_validation_batches):
        val_loss += s.run(decoder.loss, \
        generate_batch(val_img_embeds, val_captions_index, batch_size, MAX_LEN))

    val_loss /= n_validation_batches
    print('Epoch: {}, train loss: {}, val loss: {}'.format(epoch, train_loss, val_loss))

    # save weights after finishing epoch
    saver.save(s, os.path.abspath("weights_{}".format(epoch)))

print("Finished!")
