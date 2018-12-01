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
        # useful info in forward function
        self._in_dim = layers_size[0]
        if pre_trained:
            self.embeddings, embedding_dim = self._load_pre_trained(pre_trained)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._input = nn.Linear(self._dim[0] * embedding_dim, self._dim[1])
        self._layer1 = nn.Linear(self._dim[1], self._dim[2])
        self._layer2 = nn.Linear(self._dim[2], self._dim[3])

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
        return torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.embeddings(x).view((x.shape[0], -1))
        x = F.relu(x)
        x = self._input(x)
        x = F.relu(x)
        x = self._layer1(x)
        x = F.relu(x)
        x = self._layer2(x)
        x = F.log_softmax(x, 1)
        return x


if __name__ == '__main__':
    dl = TrainDataLoader(os.path.join("data", "train"), os.path.join("data", "embed_map"))
    voc_size = dl.vocab_size  # 100232
    embed_dim = 50
    out1 = int(dl.win_size * embed_dim * 0.66)
    out2 = int(dl.win_size * embed_dim * 0.33)
    out3 = int(dl.pos_dim)
    layers_dimensions = (dl.win_size, out1, out2, out3)
    NN = NeuralNet(layers_dimensions, embedding_dim=embed_dim, vocab_size=voc_size)
    e = 0
