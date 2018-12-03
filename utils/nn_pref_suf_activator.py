import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch import nn
from utils.data_loader import TrainDataLoader, TestDataLoader
from utils.loggers import PrintLogger
from utils.nn_activator import ModelActivator
from utils.nn_models import NeuralNet, PrefSufNet
from utils.params import END, DEBUG
from sys import stdout


class PrefSufModelActivator:
    def __init__(self, model: nn.Module, train_loader, dev_loader, lr=0.005, gpu=False):
        self._len_data = 0
        self._gpu = gpu
        # init models with current models
        self._model = model
        self._batch_size = 256
        self._model.set_optimizer(lr=lr)
        if self._gpu:
            self._model.cuda()
        # empty list for every model - y axis for plotting loss by epochs
        self._dev_loader = None
        self._train_loader = None
        self._train = train_loader
        self._dev = dev_loader
        self._load_data()

    # load dataset
    def _load_data(self):
        # set train loader
        self._train_loader = DataLoader(
            self._train,
            batch_size=self._batch_size, shuffle=False
        )

        # set validation loader
        self._dev_loader = DataLoader(
            self._dev,
            batch_size=self._batch_size, shuffle=False
        )

    # train a model, input is the enum of the model type
    def train(self, total_epoch, validation_rate=1):
        logger = PrintLogger("NN_train")
        logger.info("start_train")
        loss_vec_dev = []
        loss_vec_train = []
        accuracy_vec_dev = []
        accuracy_vec_train = []

        for epoch_num in range(total_epoch):
            logger.info("epoch:" + str(epoch_num))
            # set model to train mode
            self._model.train()

            # calc number of iteration in current epoch
            for batch_index, (x, p, s, label) in enumerate(self._train_loader):
                stdout.write("\r\r\r%d" % int(100 * (batch_index + 1) / len(self._train_loader)) + "%")
                stdout.flush()

                self._model.zero_grad()                         # zero gradients
                output = self._model(x, p, s)                      # calc output of current model on the current batch

                # if DEBUG:
                #     logger.info("epoch:" + str(epoch_num) + "\tx:" + str(data) +
                #                 "\ny:" + str(label) + "\npred:" + str(torch.argmax(output)))

                loss = F.cross_entropy(output, label.squeeze()) # define loss node (negative log likelihood)
                loss.backward()                                 # back propagation
                self._model.optimizer.step()                    # update weights

            # validate on Dev
            if validation_rate and epoch_num % validation_rate == 0:
                logger.info("validating dev...    epoch:" + "\t" + str(epoch_num) + "/" + str(total_epoch))
                loss, accuracy = self._validate(self._train_loader, job="Train")
                loss_vec_train.append((epoch_num, loss))
                accuracy_vec_train.append((epoch_num, accuracy))
                loss, accuracy = self._validate(self._dev_loader, job="Dev")
                loss_vec_dev.append((epoch_num, loss))
                accuracy_vec_dev.append((epoch_num, accuracy))
        return loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train

    # validation function only the model and the data are important for input, the others are just for print
    def _validate(self, data_loader, job=""):
        logger = PrintLogger(job + "_NN_validation")
        loss_count = 0
        self._model.eval()

        good_preds = 0
        all_preds = 0
        tag_O = data_loader.dataset.pos_to_idx('O')
        # run though all examples in validation set (from input)
        for batch_index, (x, p, s, label) in enumerate(self._train_loader):
            output = self._model(x, p, s)                                               # calc output of the model
            loss_count += F.cross_entropy(output.float(), label.squeeze()).item()    # sum total loss of all iteration

            # good_preds += sum([1 for _ in torch.argmax(output, dim=1) - label.squeeze() if _ == 0])
            for p, l in zip(torch.argmax(output, dim=1), label.squeeze()):
                if l.item() != tag_O:
                    all_preds += 1
                    if p.item() == l.item():
                        good_preds += 1

        loss = float(loss_count / len(data_loader.dataset))
        accuracy = good_preds / all_preds
        logger.info("loss=" + str(loss) + "  ------  accuracy=" + str(accuracy))
        return loss, accuracy

    def predict(self, data):
        test_data_loader = DataLoader(
            data,
            batch_size=1, shuffle=False
        )

        results = []
        for word, x, p, s, (is_start, is_end) in test_data_loader:
            output = torch.argmax(self._model(x, p, s)).item()       # calc output of the model
            results.append((word[0], data.idx_to_pos(output)))
            if is_end:
                results.append((END, END))

        if DEBUG:
            print(results)
        return results


if __name__ == "__main__":
    import os

    dl_train = TrainDataLoader(
        os.path.join("..", "data", "pos", "train"), suf_pref=True)  # , vocab_file=os.path.join("data", "embed_map"))
    dl_train.vocabulary.learn_distribution(os.path.join("..", "data", "pos", "dev"), labeled=True)
    dl_dev = TrainDataLoader(os.path.join("..", "data", "pos", "dev"), vocab=dl_train.vocabulary, suf_pref=True)

    voc_size = dl_train.vocab_size  # 100232
    pre_size = dl_train.vocabulary.len_pref()
    suf_size = dl_train.vocabulary.len_suf()
    embed_dim = 50
    out1 = int(dl_train.win_size * embed_dim * 0.66)
    out2 = int(dl_train.win_size * embed_dim * 0.33)
    out3 = int(dl_train.pos_dim)
    layers_dimensions = (dl_train.win_size, out1, out2, out3)
    NN = PrefSufNet(layers_dimensions, voc_size, pre_size, suf_size, embedding_dim=embed_dim)
    ma = ModelActivator(NN, dl_train, dl_dev)
    loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train = ma.train(5)

    dl_test = TestDataLoader(os.path.join("..", "data", "pos", "test"), dl_train.vocabulary, labeled=False,
                             suf_pref=True)
    dl_test.vocabulary.learn_distribution(os.path.join("..", "data", "pos", "test"), labeled=False)
    dl_test.load_pos_map(dl_train.pos_map)
    ma.predict(dl_test)

