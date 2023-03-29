import io
import os
import tqdm
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import styled_print
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_preprocessing_datapipeline(df, batch_size=512):
    def only_sentence(sample):
        return sample["paragraphs"]

    datagen = tf.data.Dataset.from_tensor_slices(dict(df))
    datagen = datagen.map(only_sentence, num_parallel_calls=tf.data.AUTOTUNE)

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=None,
        standardize='strip_punctuation',
        split='whitespace',
        ngrams=None,
        output_mode='int',
        output_sequence_length=None,
        pad_to_max_tokens=False
    )
    vectorize_layer.adapt(datagen.batch(batch_size))

    text_vector_datagen = datagen.batch(1024).prefetch(
        tf.data.AUTOTUNE).map(vectorize_layer, num_parallel_calls=tf.data.AUTOTUNE).unbatch()
    return datagen, text_vector_datagen, vectorize_layer


def create_skip_grams(sequence, vocabulary_size, window_size=2, sampling_table=None, only_positive_skip_grams=True):
    if only_positive_skip_grams:
        negative_samples = 0
    else:
        negative_samples = 1
    skip_grams, labels = tf.keras.preprocessing.sequence.skipgrams(
        sequence,
        vocabulary_size=vocabulary_size,
        window_size=window_size,
        sampling_table=sampling_table,
        negative_samples=negative_samples)
    return skip_grams, labels


def get_negative_sampling_candidates(context, num_ns, vocabulary_size, seed):
    context_class = tf.reshape(tf.constant(context, dtype="int64"), (1, 1))
    negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
        true_classes=context_class,
        num_true=1,
        num_sampled=num_ns,
        unique=True,
        range_max=vocabulary_size,
        seed=seed,
        name="negative_sampling"
    )
    return negative_sampling_candidates


def create_training_pairs(sequences, window_size, num_ns, vocabulary_size, seed=1):
    targets, contexts, labels = [], [], []
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(
        size=vocabulary_size)
    for sequence in tqdm.tqdm(sequences):
        skip_grams, _ = create_skip_grams(
            sequence,
            vocabulary_size,
            window_size,
            sampling_table,
            only_positive_skip_grams=True
        )

        for target_word, context_word in skip_grams:
            negative_sampling_candidates = get_negative_sampling_candidates(
                context_word, num_ns, vocabulary_size, seed
            )

            # Build context and label vectors (for one target word)
            context = tf.concat(
                [tf.constant([context_word], dtype="int64"), negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
    return targets, contexts, labels


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = tf.keras.layers.Embedding(vocab_size,
                                                          embedding_dim,
                                                          input_length=1,
                                                          name="w2v_target_embedding")
        self.context_embedding = tf.keras.layers.Embedding(vocab_size,
                                                           embedding_dim,
                                                           input_length=num_ns+1,
                                                           name="w2v_context_embedding")

    def call(self, pair):
        target, context = pair
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        return dots


def create_training_datapipeline(targets, contexts, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(targets.shape[0])
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def extract_w2v_vectors(model, artifact_dir, embedding_name="w2v_target_embedding"):
    weights = model.get_layer(embedding_name).get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    out_v = io.open(
        os.path.join(artifact_dir, 'vectors.tsv'), 'w', encoding='utf-8')
    out_m = io.open(os.path.join(artifact_dir, 'metadata.tsv'),
                    'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Custom Word2Vec')
    parser.add_argument('--csv-file-paths', type=str, nargs='+',
                        help="List of paths to data csv files.")
    parser.add_argument('--logdir', type=str, default="../data/word2vec/logs",
                        help="Path to save training logs and artifacts.")
    parser.add_argument('--window-size', type=int, default=2,
                        help="Window size for skipgram creation.")
    parser.add_argument('--num-negs', type=int, default=5,
                        help="Number of negative (target, context) pairs per positive pair.")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="Batch size for model training.")
    parser.add_argument('--epochs', type=int, default=25,
                        help="Batch size for model training.")
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help="Dimension of embedding space.")
    args = parser.parse_args()

    styled_print("Creating Word2Vec Model ... ", header=True)
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    # Data reading and concatenation for multiple sources.
    styled_print("Extracting Data ... ", header=True)
    dfs = []
    for file_path in args.csv_file_paths:
        styled_print(f"Reading data from {file_path} ...")
        df = pd.read_csv(file_path)
        styled_print(f"Found {df.shape[0]} sentences ...")
        dfs.append(df)

    dataframe = pd.concat(dfs, ignore_index=True)
    styled_print(
        f"Found total {dataframe.shape[0]} sentences across {len(dfs)} data sources ...")
    dataframe.drop(["id"], axis=1)

    # Data cleaning and preprocessing.
    styled_print(f"Apply Preprocessing Pipeline to Dataset ...", header=True)
    _, text_vector_datagen, vectorize_layer = get_preprocessing_datapipeline(
        dataframe)
    sequences = list(text_vector_datagen.as_numpy_iterator())
    styled_print(
        f"Checking the first Sequences after preprocessing ...")
    for sequence in sequences[:1]:
        styled_print(f"The length of Sequence is {len(sequence)}")
        styled_print(sequence)
        for token in sequence[:30]:
            styled_print(
                f"{token} --> {vectorize_layer.get_vocabulary()[token]}")

    # Create Training Data - Positive and Negative Skip Grams
    styled_print("Creating Training Dataset of +/- Skipgrams ...", header=True)
    targets, contexts, labels = create_training_pairs(
        sequences=sequences,
        window_size=args.window_size,
        num_ns=args.num_negs,
        vocabulary_size=vectorize_layer.vocabulary_size(),
        seed=1)

    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)

    styled_print(f"The shape of target array is: {targets.shape}")
    styled_print(f"The shape of context array is: {contexts.shape}")
    styled_print(f"The shape of label array is: {labels.shape}")

    # Create data pipeline for training
    styled_print("Building Training Data Pipeline ...", header=True)
    dataset = create_training_datapipeline(
        targets, contexts, labels, args.batch_size)

    # Create Model Architecture
    w2v_model = Word2Vec(vectorize_layer.vocabulary_size(),
                         args.embedding_dim, args.num_negs)
    w2v_model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

    # Callbacks and Training
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.logdir)
    w2v_model.fit(dataset, epochs=args.epochs,
                  callbacks=[tensorboard_callback], verbose=1)

    # Save Embeddings for Visualization
    extract_w2v_vectors(w2v_model, args.logdir,
                        embedding_name="w2v_target_embedding")

    w2v_model.save(os.path.join(args.logdir, "tf-saved-model"))
