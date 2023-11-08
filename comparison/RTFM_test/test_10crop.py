import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import numpy as np
import tqdm.autonotebook as tqdm
import json


def visualize_temporal_action_bmn_(P_test, y_test):
    # fig, ax = plt.subplots(figsize=(12, 3))

    # font = {'family': 'normal',
    #         'weight': 'bold',
    #         'size': 22}

    plt.rcParams.update({'font.size': 22})

    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})

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

    p_label, p_start, p_end = get_labels_start_end_time(P_test)
    y_label, y_start, y_end = get_labels_start_end_time(y_test)

    range_list = [(ystart, (yend - ystart)) for ystart, yend in zip(y_start, y_end)]
    # for ystart, yend in zip(y_start, y_end):
    axes[0].broken_barh(range_list, (0, 1), facecolors='green')

    range_list = [(pstart, (pend - pstart)) for pstart, pend in zip(p_start, p_end)]
    # for pstart, pend in zip(p_start, p_end):
    # axes[1].broken_barh(range_list, (0, 1), facecolors='green')

    # fig.delaxes(axes[1])
    # fig.delaxes(axes[3])

    # plt.show()
    plt.savefig('figure.png')
    plt.close()


def calc_f1(fn, fp, tp):
    precision = float(tp / float(tp + fp + 1e-10))
    recall = float(tp / float(tp + fn + 1e-10))
    f1 = 2.0 * (precision * recall) / (precision + recall + 1e-10)
    f1 = float(np.nan_to_num(f1) * 100)
    # return Dict(f1=f1, precision=precision, recall=recall)
    return f1, precision, recall


def get_labels_start_end_time(frame_wise_labels, bg_class=0):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] != bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] != bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label != bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label != bg_class:
        ends.append(i)
    return labels, starts, ends


def f_score(recognized, ground_truth, overlap, bg_class=0):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class=bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class=bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):

        if len(y_label) == 0:
            fp += 1
            continue

        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        # print(union)
        IoU = (1.0 * intersection / (union + 1e-5)) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        pred_ = []

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        elif args.dataset == 'xd-violance':
            gt = np.load('list/gt-xdv.npy')
        else:
            gt = np.load('list/gt-ucf.npy')

        tp, fp, fn = 0, 0, 0
        gt_list = []
        pred_list = []
        gt_index = 0
        gt_json = dict()
        for input, video_name in tqdm.tqdm(dataloader):

            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            crop_index = 0
            input = input[:, [crop_index], :, :]
            input = input.half().float()
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))
            sig = sig.cpu()
            video_gt = gt[gt_index:gt_index + len(sig)*10:10]
            gt_index += len(sig) * 10
            # if 'Arrest007' not in video_name[0]:
            #     continue
            pred_list.append(sig.numpy()[:, 0])
            gt_list.append(video_gt)
            gt_json[video_name[0]] = dict(num_frames=len(video_gt), labels=video_gt.astype('float').tolist())

        for gt_element, pred_element in zip(gt_list, pred_list):
            pred_element = np.interp(np.linspace(0, len(pred_element)-1, len(gt_element)),
                                     np.arange(0, len(pred_element)), pred_element)
            pred_ += pred_element.tolist()
            tp_, fp_, fn_ = f_score((pred_element > 0.5).astype('float'), gt_element, 0.25, bg_class=-1)
            tp_1, fp_1, fn_1 = f_score((pred_element > 0.5).astype('float'), gt_element, 0.25, bg_class=0)
            tp_0, fp_0, fn_0 = f_score((pred_element < 0.5).astype('float'), gt_element == 0, 0.25, bg_class=0)
            # visualize_temporal_action_bmn_(pred_element, gt_element)
            tp += tp_0
            fp += fp_0
            fn += fn_0

        with open('xd-violence_ground_truth.testing.json', 'w') as fp:
            json.dump(gt_json, fp)

        f1, _, _ = calc_f1(fn, fp, tp)
        print(f1)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 10)

        fpr, tpr, threshold = roc_curve(list(gt), np.array(pred))
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        ap = average_precision_score(list(gt), pred)
        print('ap : ' + str(ap))

        print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        # viz.plot_lines('pr_auc', pr_auc)
        # viz.plot_lines('auc', rec_auc)
        # viz.lines('scores', pred)
        # viz.lines('roc', tpr, fpr)
        return rec_auc
