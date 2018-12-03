import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
from utils.data_loader import TrainDataLoader


class NeuralNet(nn.Module):
    def __init__(self, layers_size, vocab_size, embedding_dim=50, lr=0.001, pre_trained=None):
        super(NeuralNet, self).__init__()

        self._dim = layers_size
        self._in_dim = layers_size[0]
        if pre_trained:
            self.embeddings, embedding_dim = self._load_pre_trained(pre_trained)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._input = nn.Linear(self._dim[0] * embedding_dim, self._dim[1])
        self._layer1 = nn.Linear(self._dim[1], self._dim[2])

        # set optimizer
        self.optimizer = self.set_optimizer(lr)

    @staticmethod
    def _load_pre_trained(weights_matrix, non_trainable=False):
        weights_matrix = torch.Tensor(np.loadtxt(weights_matrix))
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer, embedding_dim

    # init optimizer with RMS_prop
    def set_optimizer(self, lr):
        return torch.optim.RMSprop(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.embeddings(x).view(x.shape[0], -1)
        x = torch.tanh(x)
        x = self._input(x)
        x = torch.tanh(x)
        x = self._layer1(x)
        x = F.log_softmax(x, 1)
        return x


class PrefSufNet(nn.Module):
    def __init__(self, layers_size, vocab_size, pref_size, suf_size, embedding_dim=50, lr=0.01, pre_trained=None):
        super(PrefSufNet, self).__init__()
        self._dim = layers_size
        # useful info in forward function
        self._in_dim = layers_size[0]
        if pre_trained:
            self.embeddings, embedding_dim = self._load_pre_trained(pre_trained)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.pref_embeddings = nn.Embedding(pref_size, embedding_dim)
        self.pref_embeddings.load_state_dict({'weight': torch.zeros(pref_size, embedding_dim)})
        self.suf_embeddings = nn.Embedding(suf_size, embedding_dim)
        self.suf_embeddings.load_state_dict({'weight': torch.zeros(suf_size, embedding_dim)})

        self._input = nn.Linear(self._dim[0] * embedding_dim , self._dim[1])
        self._layer1 = nn.Linear(self._dim[1], self._dim[2])

        # set optimizer
        self.optimizer = self.set_optimizer(lr)

    @staticmethod
    def _load_pre_trained(weights_matrix, non_trainable=False):
        weights_matrix = torch.Tensor(np.loadtxt(weights_matrix))
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer, embedding_dim

    # init optimizer with RMS_prop
    def set_optimizer(self, lr):
        return torch.optim.RMSprop(self.parameters(), lr=lr)

    def forward(self, x, x_pref, x_suf):
        dim = 1 if len(x.shape) == 1 else x.shape[0]
        x = (self.embeddings(x) + self.pref_embeddings(x_pref) + self.suf_embeddings(x_suf)).view((dim, -1))
        # F.dropout(x, 0.3)
        # x = torch.cat([self.embeddings(x).view((dim, -1)), self.pref_embeddings(x_pref).view((dim, -1)),
        #                self.suf_embeddings(x_suf).view((dim, -1))], 1)
        x = torch.tanh(x)
        x = self._input(x)
        x = torch.tanh(x)
        x = self._layer1(x)
        x = F.log_softmax(x, 1)
        return x


if __name__ == '__main__':
    dl = TrainDataLoader(os.path.join("..", "data", "pos", "train"), suf_pref=True)
    voc_size = dl.vocab_size  # 100232
    pre_size = dl.vocabulary.len_pref()
    suf_size = dl.vocabulary.len_suf()
    embed_dim = 50
    out1 = int(dl.win_size * embed_dim * 0.66)
    out2 = int(dl.win_size * embed_dim * 0.33)
    out3 = int(dl.pos_dim)
    layers_dimensions = (dl.win_size, out1, out2, out3)
    NN = PrefSufNet(layers_dimensions, voc_size, pre_size, suf_size, embedding_dim=embed_dim)
    x, p, s, l = dl.__getitem__(0)
    NN(x, p, s)
    e = 0
