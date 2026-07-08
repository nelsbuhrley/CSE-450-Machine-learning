"""
Generate text from a trained char-RNN checkpoint (see main.py for training).

Usage:
    python generate.py --start "The " --length 500 --temperature 0.8
"""

import argparse
import contextlib
import json
import os

# Generation is a sequential, one-char-at-a-time loop — a GPU can't
# parallelize it and gains nothing. On this cluster's login node, TF can see
# a GPU device stub it can't actually claim. Default to CPU-only; override
# by setting CUDA_VISIBLE_DEVICES before running if you're on an allocated
# GPU node and want to use it anyway.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
# Silences oneDNN's INFO banner at import time.
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import tensorflow as tf

CHECKPOINT_DIR = './training_checkpoints'
DEFAULT_EMBEDDING_DIM = 256
DEFAULT_RNN_UNITS = 1024

# Each `sbatch_jobs/train_<author>.slurm` run writes its checkpoint under
# ../runs/<author>/training_checkpoints/ (see main.py). Resolved from this
# file's location rather than cwd, so --author works regardless of where
# generate.py is invoked from.
RUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'runs'))


def available_authors():
    """Author names under ../runs/ that have a trained checkpoint ready to load."""
    if not os.path.isdir(RUNS_DIR):
        return []
    authors = []
    for name in sorted(os.listdir(RUNS_DIR)):
        vocab_path = os.path.join(RUNS_DIR, name, 'training_checkpoints', 'vocab.json')
        if os.path.isfile(vocab_path):
            authors.append(name)
    return authors


@contextlib.contextmanager
def _quiet_stderr():
    """Suppress the "cuInit failed" CUDA-probe error TF logs on first GPU-
    touching op when no GPU is visible — it's below TF_CPP_MIN_LOG_LEVEL's
    reach and harmless (falls back to CPU regardless)."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(saved_stderr_fd)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    with _quiet_stderr():
        return tf.keras.Sequential([
            tf.keras.layers.Input(batch_shape=(batch_size, None)),
            tf.keras.layers.Embedding(vocab_size, embedding_dim),
            tf.keras.layers.GRU(rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])


def _filter_logits(logits, top_k, top_p):
    """1D logits over the vocab. Sets everything outside the top_k most
    likely characters, and/or outside the smallest set whose cumulative
    probability crosses top_p, to -inf so sampling can't pick them."""
    if top_k > 0:
        top_values, _ = tf.math.top_k(logits, k=min(top_k, logits.shape[-1]))
        min_value = top_values[-1]
        logits = tf.where(logits < min_value, tf.fill(tf.shape(logits), float('-inf')), logits)

    if top_p < 1.0:
        sorted_indices = tf.argsort(logits, direction='DESCENDING')
        sorted_logits = tf.gather(logits, sorted_indices)
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits))
        sorted_mask = cumulative_probs > top_p
        # Always keep the first token that crosses the threshold.
        sorted_mask = tf.concat([[False], sorted_mask[:-1]], axis=0)
        remove_indices = tf.boolean_mask(sorted_indices, sorted_mask)
        neg_inf = tf.fill(tf.shape(remove_indices), float('-inf'))
        logits = tf.tensor_scatter_nd_update(logits, tf.expand_dims(remove_indices, axis=1), neg_inf)

    return logits


def generate_text(model, char2idx, idx2char, start_string, num_generate, temperature, top_k=0, top_p=1.0):
    input_eval = tf.expand_dims([char2idx[c] for c in start_string], 0)
    temperature = tf.constant(temperature, dtype=tf.float32)

    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()

    # Tracing this once and reusing it avoids re-dispatching the whole
    # eager graph on every single character (~10x faster in practice) —
    # the plain eager loop pays Python/graph-dispatch overhead per step.
    @tf.function
    def step(inp):
        predictions = model(inp)
        logits = predictions[0, -1, :] / temperature
        logits = _filter_logits(logits, top_k, top_p)
        return tf.random.categorical(tf.expand_dims(logits, 0), num_samples=1)[0, 0]

    print(start_string, end='', flush=True)
    for _ in range(num_generate):
        predicted_id = step(input_eval)
        input_eval = tf.expand_dims([predicted_id], 0)
        print(idx2char[predicted_id.numpy()], end='', flush=True)
    print()


def main():
    authors = available_authors()
    parser = argparse.ArgumentParser(description="Generate text from a trained char-RNN checkpoint.")
    parser.add_argument('--author', choices=authors, default=None,
                         help=f"which author's model to generate from (available: {', '.join(authors) or 'none found'})")
    parser.add_argument('--list', action='store_true',
                         help='list available authors (from ../runs/) and exit')
    parser.add_argument('--start', default='The ', help='seed string to start generation from')
    parser.add_argument('--length', type=int, default=500, help='number of characters to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                         help='higher = more random/creative, lower = more conservative/repetitive')
    parser.add_argument('--top-k', type=int, default=0,
                         help='only sample from the K most likely characters each step (0 = disabled)')
    parser.add_argument('--top-p', type=float, default=1.0,
                         help='nucleus sampling: sample from the smallest set of characters whose '
                              'cumulative probability exceeds this (1.0 = disabled)')
    parser.add_argument('--checkpoint-dir', default=None,
                         help=f'explicit checkpoint dir, overrides --author (default: {CHECKPOINT_DIR})')
    parser.add_argument('--weights', default='best.weights.h5',
                         help='weights filename inside checkpoint-dir')
    args = parser.parse_args()

    if args.list:
        if authors:
            print("Available authors (../runs/):")
            for name in authors:
                print(f"  {name}")
        else:
            print(f"No trained checkpoints found under {RUNS_DIR}")
        return

    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    elif args.author:
        checkpoint_dir = os.path.join(RUNS_DIR, args.author, 'training_checkpoints')
    else:
        checkpoint_dir = CHECKPOINT_DIR
    args.checkpoint_dir = checkpoint_dir

    with open(os.path.join(args.checkpoint_dir, 'vocab.json'), 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        embedding_dim = config['embedding_dim']
        rnn_units = config['rnn_units']
    else:
        embedding_dim = DEFAULT_EMBEDDING_DIM
        rnn_units = DEFAULT_RNN_UNITS

    model = build_model(len(vocab), embedding_dim, rnn_units, batch_size=1)
    model.load_weights(os.path.join(args.checkpoint_dir, args.weights))
    model.build(tf.TensorShape([1, None]))

    generate_text(model, char2idx, idx2char, args.start, args.length, args.temperature,
                  args.top_k, args.top_p)


if __name__ == '__main__':
    main()
