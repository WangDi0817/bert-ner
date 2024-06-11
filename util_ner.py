
import numpy as np

def compute_metrics(tag_preds, tag_true):
    """ 多标签的评价
    :param tag_preds:
    :param tag_true:
    :return:
    """

    assert len(tag_preds) == len(tag_true), f"pred text num {len(tag_preds)} is not match sample num {len(tag_true)}"
    true_ = 0
    all = 0
    for i in range(len(tag_true)):
        for j in tag_true[i]:
            if j == -1:
                continue
            else:
                true_ += np.equal(tag_true[i][j], tag_preds[i][j])
                all += 1
    f = true_ / all
    return f
