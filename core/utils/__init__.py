import json
import os.path
import re

import numpy as np

from configs.common import BASE_DIR


class Voc:
    def __init__(self):
        pass

    def load_data(self, path):
        with open(path) as f:
            self.__dict__ = json.load(f)

voc = Voc()
voc.load_data(os.path.join(BASE_DIR, "assets", "voc.json"))

class T5InferenceHelper:
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2

    voc = voc

    @classmethod
    def normalize_string(cls, s):
        s = s.lower()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    @classmethod
    def get_batched_indices(cls, sentence):
        return [
            [
                el
                for el in [
                    cls.voc.word2index.get(word) for word in sentence.split(' ')
                ] if el is not None
            ]
            + [cls.EOS_token]
        ]

    @classmethod
    def list2numpy(cls, indices):
        return np.array(indices, dtype=np.long).transpose()

    @classmethod
    def indices2str(cls, indices):
        return ' '.join(
            [
                cls.voc.index2word[str(ind)]
                for ind in indices
            ]
        )
