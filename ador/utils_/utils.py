import os
import matplotlib.pylab as plt
import metrics
import numpy as np

import threading
import queue


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def get_input(message, channel):
    response = input(message)
    channel.put(response)


def input_with_timeout(message, timeout, default_answer):
    channel = queue.Queue()
    message = message + " [{} sec timeout] ".format(timeout)
    thread = threading.Thread(target=get_input, args=(message, channel))
    # by setting this as a daemon thread, python won't wait for it to complete
    thread.daemon = True
    thread.start()

    try:
        response = channel.get(True, timeout)
        return response
    except queue.Empty:
        pass
    return default_answer


def ignore_func(dir, file_list, ext=".py"):
    ignored = []
    for file_name in file_list:
        file_path = os.path.join(dir, file_name)
        if not os.path.isdir(file_path):
            if not file_path.endswith(ext):
                ignored.append(file_name)
    return ignored


def imshow_(x, **kwargs):
    if x.ndim == 2:
        plt.imshow(x, interpolation="nearest", **kwargs)
    elif x.ndim == 1:
        plt.imshow(x[:, None].T, interpolation="nearest", **kwargs)
        plt.yticks([])
    plt.axis("tight")


def visualizeHeatMapPredictions(P_test, y_test, expFolder, videoName):
    # np.random.seed(1)
    plt.rcParams["figure.figsize"] = 10, 3

    x = np.arange(0, len(P_test))
    fig, (ax, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

    extent = [x[0] - (x[1] - x[0]) / 2., x[-1] + (x[1] - x[0]) / 2., 0, 1]
    im = ax.imshow(np.array(y_test)[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])

    im = ax2.imshow(np.array(P_test)[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent)
    ax2.set_yticks([])
    ax3.plot(x, P_test)
    ax3.set_yticks([])
    ax3.set_xticks([])

    fig.subplots_adjust(right=0.82)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    cb = plt.colorbar(im, cax=cbar_ax)

    # plt.show()

    saveFolder = os.path.join(expFolder, "heatmap")
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    plt.savefig(os.path.join(saveFolder, videoName + ".pdf"))
    plt.close()


def visualize_temporal_action(P_test, y_test, save_path, videoName):
    # fig, ax = plt.subplots(figsize=(12, 3))

    fig, axes = plt.subplots(3, 1)
    fig.set_figwidth(12)

    axes[2].set_ylim(0, 1)
    axes[2].set_xlim(0, len(y_test))

    axes[2].plot(np.arange(0, len(P_test), 1), np.array(P_test))

    P_test = (np.array(P_test) > 0.5).astype('float').tolist()

    for i in range(2):
        axes[i].set_ylim(0, 1)
        axes[i].set_xlim(0, len(y_test))
        axes[i].broken_barh([(0, len(y_test))], (0, 1), facecolors='papayawhip')

        axes[i].set_xlabel('seconds')
        axes[i].set_yticks([0.5])

    axes[0].set_yticklabels(['GT'])
    axes[1].set_yticklabels(['Pred'])

    # ax.broken_barh([(0, len(y_test))], (1.25, 2.25), facecolors='papayawhip')
    # ax.broken_barh([(0, len(P_test))], (0, 1), facecolors='papayawhip')
    #
    # ax.set_ylim(0, 2.25)
    # ax.set_xlim(0, len(y_test))
    #
    # ax.set_xlabel('seconds')
    # ax.set_yticks([0.5, 1.75])
    # ax.set_yticklabels(['Pred', 'GT'])

    p_label, p_start, p_end = metrics.get_labels_start_end_time(P_test)
    y_label, y_start, y_end = metrics.get_labels_start_end_time(y_test)

    range_list = [(ystart, (yend-ystart)) for ystart, yend in zip(y_start, y_end)]
    # for ystart, yend in zip(y_start, y_end):
    axes[0].broken_barh(range_list, (0, 1), facecolors='darkred')

    range_list = [(pstart, (pend-pstart)) for pstart, pend in zip(p_start, p_end)]
    # for pstart, pend in zip(p_start, p_end):
    axes[1].broken_barh(range_list, (0, 1), facecolors='darkred')

    # plt.show()
    plt.savefig(save_path)
    plt.close()


def visualize_temporal_action_bmn(P_test, y_test, start, end, save_path, videoName):
    # fig, ax = plt.subplots(figsize=(12, 3))

    fig, axes = plt.subplots(4, 1,  gridspec_kw={'height_ratios': [1, 1, 2, 2]})

    fig.set_figwidth(12)
    fig.set_figheight(6)

    axes[2].set_ylim(0, 1)
    axes[2].set_xlim(0, len(y_test))

    axes[2].spines['left'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].axhline(0.5, color='black', ls='--')
    # axes[2].axhline(0.5)

    axes[2].plot(np.arange(0, len(P_test), 1), np.array(P_test))

    axes[3].set_ylim(0, 1)
    axes[3].set_xlim(0, len(y_test))

    axes[3].plot(np.arange(0, len(y_test), 1), np.array(start), color='green')
    axes[3].plot(np.arange(0, len(y_test), 1), np.array(end), color='red', linestyle='dashed')

    P_test = (np.array(P_test) > 0.5).astype('float').tolist()

    for i in range(2):
        axes[i].set_ylim(0, 1)
        axes[i].set_xlim(0, len(y_test))
        # axes[i].broken_barh([(0, len(y_test))], (0, 1), facecolors='papayawhip')

        # axes[i].set_xlabel('seconds')
        axes[i].set_yticks([0.5])

    axes[0].set_yticklabels(['GT'])
    axes[1].set_yticklabels(['Pred'])

    # ax.broken_barh([(0, len(y_test))], (1.25, 2.25), facecolors='papayawhip')
    # ax.broken_barh([(0, len(P_test))], (0, 1), facecolors='papayawhip')
    #
    # ax.set_ylim(0, 2.25)
    # ax.set_xlim(0, len(y_test))
    #
    # ax.set_xlabel('seconds')
    # ax.set_yticks([0.5, 1.75])
    # ax.set_yticklabels(['Pred', 'GT'])

    p_label, p_start, p_end = metrics.get_labels_start_end_time(P_test)
    y_label, y_start, y_end = metrics.get_labels_start_end_time(y_test)

    range_list = [(ystart, (yend-ystart)) for ystart, yend in zip(y_start, y_end)]
    # for ystart, yend in zip(y_start, y_end):
    axes[0].broken_barh(range_list, (0, 1), facecolors='green')

    range_list = [(pstart, (pend-pstart)) for pstart, pend in zip(p_start, p_end)]
    # for pstart, pend in zip(p_start, p_end):
    axes[1].broken_barh(range_list, (0, 1), facecolors='green')

    fig.delaxes(axes[1])
    fig.delaxes(axes[3])

    # plt.show()
    plt.savefig(save_path)
    plt.close()


def visualize_temporal_action_bmn_(P_test, y_test, start, end, save_path, videoName):
    # fig, ax = plt.subplots(figsize=(12, 3))

    # font = {'family': 'normal',
    #         'weight': 'bold',
    #         'size': 22}

    plt.rcParams.update({'font.size': 22})

    fig, axes = plt.subplots(2, 1,  gridspec_kw={'height_ratios': [1, 2]})

    fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
    fig.set_figwidth(12)
    fig.set_figheight(5)

    axes[0].set_yticks([])
    axes[1].set_yticks([])

    axes[1].set_ylim(0, 1)
    axes[1].set_xlim(0, len(y_test))

    axes[0].set_xlim(0, len(y_test))

    axes[1].spines['left'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].axhline(0.5, color='black', ls='--')
    axes[1].set_yticks([0, 0.5, 1.0])

    # axes[2].axhline(0.5)

    axes[1].plot(np.arange(0, len(P_test), 1), np.array(P_test))

    P_test = (np.array(P_test) > 0.5).astype('float').tolist()

    # for i in range(1, 2):
    #     axes[i].set_ylim(0, 1)
    #     axes[i].set_xlim(0, len(y_test))
    #     # axes[i].broken_barh([(0, len(y_test))], (0, 1), facecolors='papayawhip')
    #
    #     # axes[i].set_xlabel('seconds')
    #     axes[i].set_yticks([0, 0.5, 1.0])

    # axes[0].set_yticklabels(['GT'])

    axes[1].set(xlabel='Frame Number', ylabel='Anomaly Scores')
    axes[0].set(ylabel='GT')

    # axes[1].set_yticklabels(['Pred'])

    # ax.broken_barh([(0, len(y_test))], (1.25, 2.25), facecolors='papayawhip')
    # ax.broken_barh([(0, len(P_test))], (0, 1), facecolors='papayawhip')
    #
    # ax.set_ylim(0, 2.25)
    # ax.set_xlim(0, len(y_test))
    #
    # ax.set_xlabel('seconds')
    # ax.set_yticks([0.5, 1.75])
    # ax.set_yticklabels(['Pred', 'GT'])

    p_label, p_start, p_end = metrics.get_labels_start_end_time(P_test)
    y_label, y_start, y_end = metrics.get_labels_start_end_time(y_test)

    range_list = [(ystart, (yend-ystart)) for ystart, yend in zip(y_start, y_end)]
    # for ystart, yend in zip(y_start, y_end):
    axes[0].broken_barh(range_list, (0, 1), facecolors='green')

    range_list = [(pstart, (pend-pstart)) for pstart, pend in zip(p_start, p_end)]
    # for pstart, pend in zip(p_start, p_end):
    # axes[1].broken_barh(range_list, (0, 1), facecolors='green')

    # fig.delaxes(axes[1])
    # fig.delaxes(axes[3])

    # plt.show()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    visualizeHeatMapPredictions(np.cumsum(np.random.randn(50)), np.cumsum(np.random.randn(50)), "", "")