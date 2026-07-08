import json
import os

import tensorflow as tf
import numpy as np

checkpoint_dir = './training_checkpoints' # Directory to save training checkpoints
os.makedirs(checkpoint_dir, exist_ok=True) # Create the directory if it doesn't exist

text = open('input.txt', 'rb').read().decode(encoding='utf-8') # Read the input text file and decode it to a string

vocab = sorted(set(text)) # Create a sorted list of unique characters in the text
char2idx = {u: i for i, u in enumerate(vocab)} # Create a mapping from characters to indices
idx2char = np.array(vocab) # Create a mapping from indices to characters

# Persist the vocab alongside the weights: idx2char depends on the exact
# text used to build it, so a checkpoint is only reusable if this travels
# with it.
with open(os.path.join(checkpoint_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
    json.dump(vocab, f)

text_as_int = np.array([char2idx[c] for c in text]) # Convert the entire text into an array of integers using the char2idx mapping

seq_length = 100 # Length of each training sequence
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) # Create a TensorFlow dataset from the array of integers
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True) # Create batches of sequences of length seq_length + 1 (the extra character is for the target)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]   # "hello worl" -> predict "ello world"

dataset = sequences.map(split_input_target) # Map the split_input_target function to each sequence in the dataset to create input-target pairs
BATCH_SIZE = 128 # Batch size for training Larger batch sizes can lead to faster training but require more memory
dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True) # Shuffle the dataset and create batches of size BATCH_SIZE, dropping any remainder sequences that don't fit into a full batch

vocab_size = len(vocab)

embedding_dim = 256 # Dimension of the embedding layer. This is the size of the vector space in which characters will be embedded. A larger embedding dimension can capture more complex relationships between characters but also increases the number of parameters in the model.
rnn_units = 1024 # Number of units in the RNN layer. More units can capture more complex patterns in the data but also increase the number of parameters and the risk of overfitting.

# So generate.py can rebuild the exact same architecture before loading weights.
with open(os.path.join(checkpoint_dir, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump({'embedding_dim': embedding_dim, 'rnn_units': rnn_units}, f)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size): # Function to build the RNN model. It takes the vocabulary size, embedding dimension, number of RNN units, and batch size as input parameters.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=(batch_size, None)), # Input layer that specifies the shape of the input data. The batch size is fixed, but the sequence length can vary (hence None).
        tf.keras.layers.Embedding(vocab_size, embedding_dim), # Embedding layer that converts integer-encoded characters into dense vectors of fixed size (embedding_dim). This allows the model to learn a continuous representation of characters.
        tf.keras.layers.GRU(rnn_units, # GRU layer with rnn_units number of units. GRU (Gated Recurrent Unit) is a type of RNN that is capable of capturing temporal dependencies in sequential data.
                             return_sequences=True, # return_sequences=True means that the GRU layer will return the full sequence of outputs for each input sequence, rather than just the output at the last time step. This is important for sequence-to-sequence tasks where we want to predict the next character at each time step.
                             stateful=True, # stateful=True means that the GRU layer will maintain its state across batches. This is useful for training on sequences that are longer than the batch size, as it allows the model to remember information from previous batches.
                             recurrent_initializer='glorot_uniform', # recurrent_initializer='glorot_uniform' specifies the initializer for the recurrent kernel weights. Glorot uniform initialization is a common choice that helps with convergence during training.
                             use_cudnn="auto"), # the cluster's P100 nodes (m9g) hit a cuDNN-fused-RNN-kernel bug ("unknown cudnn status: 5003") that the A100s (cs partition) don't — sbatch_jobs/*.slurm now target cs, so the fast cuDNN path is safe to leave on. If you ever run this on a P100 node again, set this back to False.
        tf.keras.layers.Dense(vocab_size) # Dense layer that outputs a vector of size vocab_size for each time step. This layer produces the logits for each character in the vocabulary, which can then be used to compute the loss and make predictions.
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=BATCH_SIZE) # Build the model using the specified parameters.

def loss(labels, logits): # Function to compute the loss between the true labels and the predicted logits. 
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels,  # The true labels (the next character in the sequence).
        logits,  # The predicted logits (the output of the model before applying softmax).
        from_logits=True # from_logits=True indicates that the logits are raw, unnormalized scores and that the loss function should apply the softmax activation internally before computing the cross-entropy loss. 
    )

model.compile(optimizer='adam', loss=loss)

best_weights_path = os.path.join(checkpoint_dir, 'best.weights.h5')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( # Callback to save the model's weights during training.
    filepath=best_weights_path,
    save_weights_only=True,
    save_best_only=True, # only overwrite best.weights.h5 when training loss improves, so long runs don't fill the disk with a checkpoint per epoch
    monitor='loss',
    mode='min')

EPOCHS = int(os.environ.get('EPOCHS', 30)) # Overridable per-job: small corpora (e.g. runs/transformers, ~0.5MB vs 3-22MB for the authors) get few batches per epoch, so their slurm job raises this to reach a comparable number of gradient steps.
model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback], verbose=2) # verbose=2 prints one line per epoch instead of a \r-animated per-step bar, which bloats log files when stdout isn't a real terminal (as with sbatch's redirected output).