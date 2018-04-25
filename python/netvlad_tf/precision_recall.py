import matplotlib.pyplot as plt
import numpy as np

def evaluate(gt_positions, closest_match_idcs, closest_match_dists2,
             closest_gt_dists2, gt_radius):
    ''' Positions are row major. '''
    sort_i = np.argsort(closest_match_dists2)

    matched_pos = gt_positions[closest_match_idcs[sort_i], :]
    retr_dists = ((gt_positions[sort_i, :] - matched_pos) ** 2).sum(axis=1)
    tp_on_select = retr_dists < gt_radius ** 2
    fp_on_select = tp_on_select == False

    tp_cs = np.cumsum(tp_on_select)
    fp_cs = np.cumsum(fp_on_select)

    relevant_size = np.count_nonzero(
            closest_gt_dists2[sort_i] < gt_radius ** 2)
    fn_rcs = relevant_size - tp_cs

    precision = tp_cs.astype(float) / (tp_cs + fp_cs + 1e-12)
    recall = tp_cs.astype(float) / (tp_cs + fn_rcs + 1e-12)

    auc = np.sum((recall[1:] - recall[:-1]) * precision[1:])

    return precision, recall, auc
