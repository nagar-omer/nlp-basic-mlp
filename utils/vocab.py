from utils.params import START, END, UNKNOWN, SEC_ORDER, FST_ORDER, BAR, REMEMBER_BAR
import numpy as np


class Vocabulary:
    def __init__(self, src_file=None, labeled=True, vocab_file=None):
        if vocab_file:
            self._vocab_to_idx, self._idx_to_vocab = self._read_map_file(vocab_file)
        elif src_file:
            self._vocab_to_idx, self._idx_to_vocab = self._vocab_from_train(src_file, labeled)
        # {second_order_word: {word : count}}
        self._first_order_n, self._second_order_n, self._first_order_p, self._second_order_p, \
            self._prob_vocab = (None, None, None, None, {})

    def __len__(self):
        return len(self._idx_to_vocab)

    @staticmethod
    def _vocab_from_train(src_file, labeled):
        all_words = []
        src = open(src_file, "rt")
        for i, row in enumerate(src):
            if row == "\n":
                continue
            word, pos = row.split() if labeled else (row.strip(), None)
            all_words.append(word)
        idx_to_vocab = [START, END, UNKNOWN] + [w.lower() for w in set(all_words)]
        vocab_to_idx = {word: i for i, word in enumerate(idx_to_vocab)}
        return vocab_to_idx, idx_to_vocab

    @staticmethod
    def _read_map_file(map_file):
        list_words = []
        map_file = open(map_file, "rt")
        for row in map_file:
            list_words.append(row.strip())
        word_to_idx = {word: i for i, word in enumerate(list_words)}
        return word_to_idx, list_words

    def learn_distribution(self, src_file, labeled=False):
        word_count = {}
        num_words = 0
        all_words = [END, START]
        src = open(src_file, "rt")
        for i, row in enumerate(src):
            if row == "\n":
                all_words += [END, START]
                continue

            word, pos = row.split() if labeled else (row.strip(), None)
            if word in self._vocab_to_idx:
                word_count[word] = word_count.get(word, 0) + 1
            num_words += 1
            all_words.append(word)
        src.close()

        second_order_n = {}
        second_order_p = {}
        first_order_n = {}
        first_order_p = {}

        bar = BAR * np.mean(list(word_count.values()))
        for i, word in enumerate(all_words):
            if word not in self._vocab_to_idx or word == START or word == END or word_count[word] < bar:
                continue
            second_order_p[all_words[i-2]] = second_order_p[all_words[i-2]] if all_words[i-2] in second_order_p else {}
            second_order_n[all_words[i+2]] = second_order_n[all_words[i+2]] if all_words[i+2] in second_order_n else {}
            first_order_p[all_words[i-1]] = first_order_p[all_words[i-1]] if all_words[i-1] in first_order_p else {}
            first_order_n[all_words[i+1]] = first_order_n[all_words[i+1]] if all_words[i+1] in first_order_n else {}

            curr_word_score = 1 / word_count[word]
            second_order_p[all_words[i-2]][word] = second_order_p[all_words[i-2]].get(word, 0) + SEC_ORDER * curr_word_score
            second_order_n[all_words[i+2]][word] = second_order_n[all_words[i+2]].get(word, 0) + SEC_ORDER * curr_word_score
            first_order_p[all_words[i-1]][word] = first_order_p[all_words[i-1]].get(word, 0) + FST_ORDER * curr_word_score
            first_order_n[all_words[i+1]][word] = first_order_n[all_words[i+1]].get(word, 0) + FST_ORDER * curr_word_score

        self._first_order_n = first_order_n
        self._first_order_p = first_order_p
        self._second_order_n = second_order_n
        self._second_order_p = second_order_p

    def get_best_by_distribution(self, sequence):
        pp_word, p_word, unk_word, n_word, nn_word = sequence
        unk_word = unk_word.lower()
        best_candidates = {}
        if pp_word in self._second_order_p:
            for word, score in self._second_order_p[pp_word].items():
                best_candidates[word] = best_candidates.get(word, 0) + score
        if nn_word in self._second_order_n:
            for word, score in self._second_order_n[nn_word].items():
                best_candidates[word] = best_candidates.get(word, 0) + score
        if p_word in self._first_order_p:
            for word, score in self._first_order_p[p_word].items():
                best_candidates[word] = best_candidates.get(word, 0) + score
        if n_word in self._first_order_n:
            for word, score in self._first_order_n[n_word].items():
                best_candidates[word] = best_candidates.get(word, 0) + score

        if not best_candidates:     # no options at all
            return UNKNOWN, self._vocab_to_idx[UNKNOWN], 0

        word, score = max(best_candidates.items(), key=lambda x: x[1])
        if score >= REMEMBER_BAR:
            self._prob_vocab[unk_word] = self._vocab_to_idx[word]
        return word, self._vocab_to_idx[word], score

    def vocab(self, word_idx):
        if type(word_idx) == int:
            return self._idx_to_vocab[word_idx]  # case int - idx

        word = word_idx.lower()
        # case word
        if word in self._vocab_to_idx:
            return self._vocab_to_idx[word]  # case word is embedded

        # case UNK-word that was already checked
        if word in self._prob_vocab:
            return self._prob_vocab[word]  # case word is embedded

        return -1    # no matching word


if __name__ == "__main__":
    import os
    # vc = Vocabulary(vocab_file=os.path.join("data", "embed_map"))
    vc = Vocabulary(src_file=os.path.join("..", "data", "pos", "train"))
    vc.learn_distribution(os.path.join("..", "data", "pos", "train"), labeled=True)
    vc.get_best_by_distribution(["an", "oct.", "", "review", "of"])
    vc.get_best_by_distribution(["paint", "color", "colorful", "green", "red"])
    e = 1
