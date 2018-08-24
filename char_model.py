import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
batch_size = 64
epochs = 10
latent_dim = 256
num_samples = 10000

input_texts = []
target_texts = []
input_chars = {}
rev_input_chars = {}
num_encoder_tokens = 0
num_decoder_tokens = 0
target_chars = {}
rev_target_chars = {}
with open("open_corpus", "r", encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    input_text, target_text = line.split('\t')
    target_text = '\t' + target_text
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_chars:
            input_chars[char] = num_encoder_tokens
            rev_input_chars[num_encoder_tokens] = char
            num_encoder_tokens += 1
    for char in target_text:
        if char not in target_chars:
            target_chars[char] = num_decoder_tokens
            rev_target_chars[num_decoder_tokens] = char
            num_decoder_tokens += 1

# Define max input length and output length

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

#print(num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length)

# Define a 3D array with one hot representation of every sentence
enc_input_data = np.zeros((num_samples, max_encoder_seq_length, num_encoder_tokens), dtype='float32')

dec_input_data = np.zeros((num_samples, max_decoder_seq_length, num_decoder_tokens), dtype='float32')

dec_target_data = np.zeros((num_samples, max_decoder_seq_length, num_decoder_tokens), dtype='float32')

# Filling my enc_input_data and dec_data one hot representation



# Defining the encoder portion

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_out, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# define the decoder part
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

num_batches = len(lines)/num_samples

for s in range(0,9):#int(num_batches)-1):
    enc_input_data = np.zeros((num_samples, max_encoder_seq_length, num_encoder_tokens), dtype='float32')

    dec_input_data = np.zeros((num_samples, max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    dec_target_data = np.zeros((num_samples, max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts[s*num_samples:s*num_samples+num_samples], target_texts[s*num_samples:s*num_samples+num_samples])):
        for t, char in enumerate(input_text):
            enc_input_data[i, t, input_chars[char]] = 1.0
        for t, char in enumerate(target_text):
            dec_input_data[i, t, target_chars[char]] = 1.0
            if t > 0:
                dec_target_data[i, t - 1, target_chars[char]] = 1.
    model.fit([enc_input_data, dec_input_data], dec_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)

#model.save('fr_eng_model.model')

# Inference

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_h = Input(shape=(latent_dim,))
decoder_state_c = Input(shape=(latent_dim,))
decoder_states_input = [decoder_state_h, decoder_state_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_input)
decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_input, [decoder_outputs] + decoder_states)


def decode_seq(input_seq):
    enc_states_values = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))

    target_seq[0, 0, target_chars['\t']] = 1.

    stop_cond = False

    decoded_sentence = ''

    while not stop_cond:
        output_tokens, h, c = decoder_model.predict([target_seq] + enc_states_values)

        # Sample a token ??
        sampled_token_ind = np.argmax(output_tokens[0, -1, :])
        sampled_char = rev_target_chars[sampled_token_ind]
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_cond = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_ind] = 1.

        enc_states_values = [h, c]

    return decoded_sentence


enc_input_data = np.zeros((num_samples, max_encoder_seq_length, num_encoder_tokens), dtype='float32')

dec_input_data = np.zeros((num_samples, max_decoder_seq_length, num_decoder_tokens), dtype='float32')

dec_target_data = np.zeros((num_samples, max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts[90000:], target_texts[90000:])):
    for t, char in enumerate(input_text):
        enc_input_data[i, t, input_chars[char]] = 1.0
    for t, char in enumerate(target_text):
        dec_input_data[i, t, target_chars[char]] = 1.0
        if t > 0:
            dec_target_data[i, t - 1, target_chars[char]] = 1.



for seq_index in range(0,1000):
    input_seq = enc_input_data[seq_index:seq_index + 1]
    decoded_sentence = decode_seq(input_seq)
    print('-')
    print('Input Sentence: ', input_texts[seq_index])
    print('Decoded Sentence: ', decoded_sentence)
