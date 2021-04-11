import numpy as np
from itertools import chain

def translate_sequences(seqs_a, seqs_b):
    unique = np.concatenate((np.unique(seqs_a), np.unique(seqs_b)))
    mapping = dict(zip(unique, np.arange(len(unique))))

    seqs_a_encoded = np.array([mapping[e] for e in seqs_a])
    seqs_b_encoded = np.array([mapping[e] for e in seqs_b])

    return mapping, seqs_a_encoded, seqs_b_encoded


def encode_sequence(seqs):
    if isinstance(seqs, np.ndarray):
        return np.ravel(seqs), np.arange(seqs.shape[1], len(seqs), seqs.shape[1])
    else:
        return np.array(list(chain(*seqs))), np.cumsum(list(map(len, seqs)))


def sequences_preprocess(seqs_a, seqs_b):
    seqs_a_encoded, a_seqs_end = encode_sequence(seqs_a)
    seqs_b_encoded, b_seqs_end = encode_sequence(seqs_b)

    mapping, seqs_a_translated, seqs_b_translated = translate_sequences(seqs_a_encoded, seqs_b_encoded)

    return mapping, (seqs_a_translated, a_seqs_end), (seqs_b_translated, b_seqs_end)