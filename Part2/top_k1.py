import numpy as np
from math import sqrt, acos, pi


class Similarity:
    def __init__(self, vocab_file, vectors_file):
        self._vocab_to_idx, self._idx_to_vocab = self._read_map_file(vocab_file)
        self._vectors = np.loadtxt(vectors_file)

    @staticmethod
    def _read_map_file(map_file):
        list_words = []
        map_file = open(map_file, "rt")
        for row in map_file:
            list_words.append(row.strip())
        word_to_idx = {word: i for i, word in enumerate(list_words)}
        return word_to_idx, list_words

    def _dist(self, u, v):
        val = np.dot(u, v) / (sqrt(np.dot(u, u)) * sqrt(np.dot(v, v)))
        val = 1 if val > 1 else val
        val = -1 if val < -1 else val
        res_temp = acos(val)
        return abs(res_temp)

    def most_similar(self, word, k):
        word_vec = self._vectors[self._vocab_to_idx[word]]
        best_scores = [self._dist(word_vec, word_i_vec) for word_i_vec in self._vectors]
        best_arg = np.argsort([self._dist(word_vec, word_i_vec) for word_i_vec in self._vectors])[0:k+1]

        return [(self._idx_to_vocab[idx], best_scores[idx]) for idx in best_arg]


if __name__ == "__main__":
    import os
    sim = Similarity(vocab_file=os.path.join("..", "data", "word_embed", "embed_map"),
                     vectors_file=os.path.join("..", "data", "word_embed", "wordVectors"))
    print(sim.most_similar("explode", 20))
    words = ["explode", "england", "office", "john", "dog"]
    out = open("dafna-file", "wt")
    for w in words:
        out.write(str(sim.most_similar(w, 6)) + "\n")
    e = 0
