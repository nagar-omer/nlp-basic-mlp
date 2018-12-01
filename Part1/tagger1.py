import sys

from bokeh.resources import Resources

sys.path.insert(0, "..")
from utils.params import END
from utils.data_loader import TrainDataLoader, TestDataLoader
from utils.nn_activator import ModelActivator
from utils.nn_models import NeuralNet
import pickle
import os
from bokeh.plotting import show, figure, save


def _print_usage():
    print("\nOption1 - Train new model"
          "\n>> tagger1 --train   path_to_train_file   path_to_dev_file   num_epochs   model_name"
          "\n\nOption1 - Predict using existing model"
          "\n>> tagger1 --pred    path_to_test_file    model_name   out_file_name\n\n")


def _train_new_model(args):
        train_path = args[2]
        dev_path = args[3]
        num_epoch = int(args[4])
        model_name = args[5]

        dl_train = TrainDataLoader(train_path)               # , vocab_file=vocab)
        dl_train.vocabulary.learn_distribution(dev_path, labeled=True)
        dl_dev = TrainDataLoader(dev_path, vocab=dl_train.vocabulary)

        voc_size = dl_train.vocab_size
        embed_dim = 50
        out1 = int(dl_train.win_size * embed_dim * 0.66)
        out2 = int(dl_train.win_size * embed_dim * 0.33)
        out3 = dl_train.pos_dim
        layers_dimensions = (dl_train.win_size, out1, out2, out3)
        NN = NeuralNet(layers_dimensions, embedding_dim=embed_dim, vocab_size=voc_size)
        activator = ModelActivator(NN, dl_train, dl_dev)
        loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train = activator.train(num_epoch)
        _plot_line(loss_vec_dev, "Dev - loss", "loss")
        _plot_line(loss_vec_train, "Train - loss", "loss")
        _plot_line(accuracy_vec_dev, "Dev - accuracy", "accuracy")
        _plot_line(accuracy_vec_train, "Train - accuracy", "accuracy")
        pickle.dump((activator, dl_train.vocabulary, dl_train.pos_map), open(model_name, "wb"))


def _predict_existing_model(args):
    test_path = args[2]
    model_name = args[3]
    out_name = args[4]

    model_activator, vocabulary, pos_map = pickle.load(open(model_name, "rb"))

    dl_test = TestDataLoader(test_path, vocabulary, labeled=False)
    dl_test.vocabulary.learn_distribution(test_path, labeled=False)
    dl_test.load_pos_map(pos_map)
    res = model_activator.predict(dl_test)
    create_out_file(res, out_name)


def create_out_file(res, file_name):
    out_file = open(file_name, "wt")
    for word, pos in res:
        if word == END:
            continue
        out_file.write(word + " " + pos + "\n")
    out_file.close()


def get_x_y_axis(curve):
    x_axis = []
    y_axis = []
    for x, y in curve:
        x_axis.append(x)
        y_axis.append(y)
    return x_axis, y_axis


def _plot_line(vec, header, y_axis_label):
    if "fig" not in os.listdir("."):
        os.mkdir("fig")
    p = figure(plot_width=600, plot_height=250, title=header,
               x_axis_label="epochs", y_axis_label=y_axis_label)
    x, y = get_x_y_axis(vec)
    p.line(x, y, line_color='green')
    save(p, os.path.join("fig", header + ".html"), title=header, resources=Resources(mode="inline"))


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        _print_usage()
        exit(1)
    job = args[1]
    if args[1] == "--train":
        if len(args) < 6:
            _print_usage()
            exit(1)
        _train_new_model(args)

    elif args[1] == "--pred":
        if len(args) < 5:
            _print_usage()
            exit(1)
        _predict_existing_model(args)
    else:
        _print_usage()
