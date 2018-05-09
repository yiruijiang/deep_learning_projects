import encoder
import os
import tensorflow as tf
import numpy as np
from decoder import s, saver, last_epoch, decoder
from hyperparameters import LSTM_UNITS
from constant import IMG_SIZE, START, END
from preparation import vocab, vocab_inverse

load_weight = 4

class generator_model:

    # CNN encoder
    encoder, process_for_model = encoder.get_cnn_encoder()
    # restore the weights
    saver.restore(s, os.path.abspath("weights_{}".format(load_weight)))

    # containers for current lstm state
    lstm_c = tf.Variable(tf.zeros([1, LSTM_UNITS]), name = "cell")
    lstm_h = tf.Variable(tf.zeros([1, LSTM_UNITS]), name = "hidden")

    input_image = tf.placeholder('float32', [1, IMG_SIZE, IMG_SIZE, 3], name = 'images')

    img_embeds = encoder(input_image)

    bottleneck = decoder.img_embed_to_bottleneck_layer(img_embeds)

    init_c = init_h = decoder.img_embed_bottleneck_to_initialize_state_layer(bottleneck)

    init_lstm = tf.assign(lstm_c, init_c), tf.assign(lstm_h, init_h)

    current_word = tf.placeholder('int32', [1], name = "current_input")

    word_embed = decoder.word_embed_layer(current_word)

    new_c, new_h = decoder.lstm_cell(word_embed, tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h))[1]

    new_logits = decoder.token_logits_layer(decoder.token_logits_bottleneck_layer(new_h))

    new_probs = tf.nn.softmax(new_logits)

    one_step = new_probs, tf.assign(lstm_c, new_c), tf.assign(lstm_h, new_h)

def generate_caption(image, t = 1, sample = False, max_len = 20):
    """
    Generate caption for given image
    if 'sample' is True, we will sample next token from predicted probability distribution.
    t is a temperature during the sampling
    higher t causes ore uniform like distribution = more chaos
    """

    # condition lstm on the image
    s.run(generator_model.init_lstm, {generator_model.input_image: image})

    caption = [vocab[START]]

    for _ in range(max_len):

        next_word_probs = s.run(generator_model.one_step, {generator_model.current_word: [caption[-1]]})[0]

        next_word_probs = next_word_probs.ravel()

        next_word_probs = next_word_probs ** (1 / t) / np.sum(next_word_probs ** (1 / t))

        if sample:
            next_word = np.random.choice(range(len(vocab)), p = next_word_probs)
        else:
            next_word = np.argmax(next_word_probs)
        caption.append(next_word)
        if next_word == vocab[END]:
            break

    return list(map(vocab_inverse.get, caption))


    

