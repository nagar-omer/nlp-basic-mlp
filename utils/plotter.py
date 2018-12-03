import os
from bokeh.plotting import show, figure, save
from bokeh.resources import Resources


def _plot_loss_and_acc(header, loss, acc):
    if "fig" not in os.listdir(os.path.join("..")):
        os.mkdir(os.path.join("..", "fig"))
    p = figure(plot_width=600, plot_height=250, title=header,
               x_axis_label="epochs", y_axis_label="loss / accuracy")
    x1, y1 = get_x_y_axis(loss)
    x2, y2 = get_x_y_axis(acc)
    p.line(x1, y1, line_color='red')
    p.line(x2, y2, line_color='green')
    save(p, os.path.join("..", "fig", header + ".html"), title=header, resources=Resources(mode="inline"))


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
    import pickle
    res_path = os.path.join("..", "Part4", "res_pos_tagger3_model_pre_trained")
    header_dev = "POS - Dev"
    header_train = "POS - Train"
    loss_vec_dev, accuracy_vec_dev, loss_vec_train, accuracy_vec_train = pickle.load(open(res_path, "rb"))
    _plot_loss_and_acc(header_dev, loss_vec_dev, accuracy_vec_dev)
    _plot_loss_and_acc(header_train, loss_vec_train, accuracy_vec_train)
