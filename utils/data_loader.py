from sys import stdout
from torch.utils.data import Dataset, DataLoader
import os
import torch
from utils.loggers import PrintLogger
from utils.params import START, END, DEBUG, UNKNOWN
from utils.vocab import Vocabulary


class TrainDataLoader(Dataset):
    # src_file:         data source file - each row consist of <word POS> or <\n> for end of sentence
    # embed_map_file:   file withe ordered words separated by \n
    # win_size:         odd number for win size [ word_i-k ... word_i ... word_i+k ]
    def __init__(self, src_file, vocab=None, suf_pref=False, by_proba=True):
        self._by_proba=by_proba
        self._suf_pref = suf_pref
        self._win_size = 5                                                          # fix 5
        if not vocab:                                                               # vocabulary for any word
            self._vocab = Vocabulary(src_file, labeled=True, suffix_prefix=suf_pref)
        else:
            self._vocab = Vocabulary(vocab_file=vocab) if type(vocab) == str else vocab  # learn prob's for unknowns
        self._len, self._map_data, self._all_words, self._all_pos, self._idx_to_pos\
            = self._read_file(src_file, self._win_size)                             # read file to structured data
        self._pos_to_idx = {pos: i for i, pos in enumerate(self._idx_to_pos)}       # pos to index

    def __len__(self):
        return self._len

    def pos_to_idx(self, pos):
        if pos not in self._pos_to_idx:
            return -1
        return self._pos_to_idx[pos]

    def idx_to_pos(self, idx):
        return self._idx_to_pos[idx]

    @property
    def vocabulary(self):
        return self._vocab

    @property
    def pos_map(self):
        return self._idx_to_pos, self._pos_to_idx

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def pos_dim(self):
        return len(self._idx_to_pos)

    def _load_pos_map(self, pos_map):
        self._idx_to_pos = pos_map[0]
        self._pos_to_idx = pos_map[1]

    @property
    def win_size(self):
        return self._win_size

    @staticmethod
    def _read_file(src_file, win_size):
        blank_size = 2
        num_words = 0
        all_words = ([END] * blank_size) + ([START] * blank_size)
        all_pos = ([END] * blank_size) + ([START] * blank_size)
        pos_list = []
        src = open(src_file, "rt")
        for i, row in enumerate(src):
            if row == "\n":
                all_pos += ([END] * blank_size) + ([START] * blank_size)
                all_words += ([END] * blank_size) + ([START] * blank_size)
                continue
            num_words += 1
            word, pos = row.split()
            all_words.append(word)
            all_pos.append(pos)
            pos_list.append(pos)
        all_words += [END] * blank_size
        map_idx = 0
        map_data_dict = {}
        for i, word in enumerate(all_words):
            if word == START or word == END:
                continue
            map_data_dict[map_idx] = i
            map_idx += 1
        return num_words - 1, map_data_dict, all_words, all_pos, list(set(sorted(pos_list)))

    def __getitem__(self, item):
        logger = PrintLogger("get_item")
        idx = item if type(item) == int else self._map_data[item.item()]
        idx = self._map_data[item] if idx in self._map_data else len(self._all_words) + 1
        embed_vec = []
        for i, word in enumerate(self._all_words[idx - 2:idx + 3]):
            embed_i = self._vocab.vocab(word)
            if embed_i < 0 and self._by_proba:
                w, embed_i, s = self._vocab.get_best_by_distribution(self._all_words[idx - 4 + i:idx + 1 + i])
                if DEBUG:
                    logger.info("word by prob: seq=" + str(self._all_words[idx - 4 + i:idx + 1 + i]) + "\tpred=" + w)
            elif embed_i < 0:
                embed_i = self._vocab.vocab(UNKNOWN)
            embed_vec.append(embed_i)
        list_words = torch.Tensor(embed_vec).long()
        label = torch.Tensor([self._pos_to_idx[self._all_pos[idx]]]).long()
        if not self._suf_pref:
            return list_words, label

        pref_vec = []
        for word in self._all_words[idx - 2:idx + 3]:
            embed_i = self._vocab.pref_vocab(word)
            pref_vec.append(embed_i)

        suf_vec = []
        for word in self._all_words[idx - 2:idx + 3]:
            embed_i = self._vocab.suf_vocab(word)
            suf_vec.append(embed_i)
        return list_words, torch.Tensor(pref_vec).long(), torch.Tensor(suf_vec).long(), label


class TestDataLoader:
    # src_file:         data source file - each row consist of <word POS> or <\n> for end of sentence
    # vocab:            vocabulary object
    # labeled:          is data labeled
    def __init__(self, src_file, vocab, labeled=False, suf_pref=False, by_proba=False):
        self._by_proba = by_proba
        self._suf_pref = suf_pref
        self._win_size = 5                                                                # fixed 5
        self._len, self._map_data, self._all_words = self._read_file(src_file, labeled)   # read file to structured data
        self._vocab = vocab
        self._pos_to_idx, self._idx_to_pos = (None, None)

    @property
    def vocabulary(self):
        return self._vocab

    def __len__(self):
        return self._len

    def pos_to_idx(self, pos):
        if pos not in self._pos_to_idx:
            return -1
        return self._pos_to_idx[pos]

    def idx_to_pos(self, idx):
        return self._idx_to_pos[idx]

    def load_pos_map(self, pos_map):
        self._idx_to_pos = pos_map[0]
        self._pos_to_idx = pos_map[1]

    @property
    def win_size(self):
        return self._win_size

    @staticmethod
    def _read_file(src_file, labeled):
        blank_size = 2
        num_words = 0
        all_words = ([END] * blank_size) + ([START] * blank_size)
        src = open(src_file, "rt")
        for i, row in enumerate(src):
            if row == "\n":
                all_words += ([END] * blank_size) + ([START] * blank_size)
                continue
            num_words += 1
            word, pos = row.split() if labeled else (row.strip(), None)
            all_words.append(word)
        all_words += [END] * blank_size

        map_idx = 0
        map_data_dict = {}
        for i, word in enumerate(all_words):
            if word == START or word == END:
                continue
            map_data_dict[map_idx] = i
            map_idx += 1
        return num_words - 1, map_data_dict, all_words

    def __getitem__(self, item):
        logger = PrintLogger("get_item")
        idx = item if type(item) == int else self._map_data[item.item()]
        idx = self._map_data[item] if idx in self._map_data else len(self._all_words) + 1
        embed_vec = []
        is_start = True if self._all_words[idx - 1] == START else False
        is_end = True if self._all_words[idx + 1] == END else False
        for i, word in enumerate(self._all_words[idx - 2:idx + 3]):
            embed_i = self._vocab.vocab(word)
            if embed_i < 0 and self._by_proba:
                w, embed_i, s = self._vocab.get_best_by_distribution(self._all_words[idx - 4 + i:idx + 1 + i])
                if DEBUG:
                    logger.info("word by prob: seq=" + str(self._all_words[idx - 4 + i:idx + 1 + i]) + "\tpred=" + w)
            elif embed_i < 0:
                embed_i = self._vocab.vocab(UNKNOWN)
            # else:
            #     if DEBUG:
            #         print(word)
            embed_vec.append(embed_i)
        list_words = torch.Tensor(embed_vec).long()
        if not self._suf_pref:
            return self._all_words[idx], list_words, (is_start, is_end)

        pref_vec = []
        for word in self._all_words[idx - 2:idx + 3]:
            embed_i = self._vocab.pref_vocab(word)
            pref_vec.append(embed_i)

        suf_vec = []
        for word in self._all_words[idx - 2:idx + 3]:
            embed_i = self._vocab.suf_vocab(word)
            suf_vec.append(embed_i)
        return self._all_words[idx], list_words, torch.Tensor(pref_vec).long(), torch.Tensor(suf_vec).long(), (is_start, is_end)


if __name__ == "__main__":
    dl_train = TrainDataLoader(
        os.path.join("..", "data", "pos", "train"), suf_pref=True)  # , vocab_file=os.path.join("data", "embed_map"))
    dl_train.vocabulary.learn_distribution(os.path.join("..", "data", "pos", "dev"), labeled=True)
    # dl_dev = TrainDataLoader(os.path.join("..", "data", "pos", "dev"), vocab=dl_train.vocabulary)
    # d = [dl_train.__getitem__(i) for i in range(len(dl_train))]
    # d = [dl_dev.__getitem__(i) for i in range(len(dl_dev))]

    data_loader = DataLoader(
            dl_train,
            batch_size=64, shuffle=True
        )
    for batch_index, (data, label) in enumerate(data_loader):
        stdout.write("\r\r\r%d" % int(100 * ((batch_index + 1) / len(data_loader))) + "%")
        stdout.flush()

    # dl_test = TestDataLoader(os.path.join("..", "data", "pos", "test"), dl_train.vocabulary, labeled=False)
    # dl_test.vocabulary.learn_distribution(os.path.join("..", "data", "pos", "test"), labeled=False)
    # dl_test.load_pos_map(dl_train.pos_map)
    #
    # data_loader = DataLoader(
    #     dl_test,
    #     batch_size=64, shuffle=True
    # )
    # for i, (word, vec, (is_start, is_end)) in enumerate(data_loader.dataset):
    #     print(i, word, vec, is_start, is_end)

    # d = [dl_test.__getitem__(i) for i in range(len(dl_test))]
    # e = 1
