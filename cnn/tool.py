import sys
from typing import Dict, List, Any;
import numpy as np

from cnn.evaluator import Evaluator


def model_ap(det_boxes: Dict[str, List[Any]], gt_boxes: Dict[str, List[Any]], num_pos: Dict[str, int],
             classes: List[str], iou_threshold=0.5):
    """
    计算多个分类模型的AP值
    :param det_boxes: 预测框
    :param gt_boxes: 标签
    :param classes: 分类
    """
    for c in classes:
        dects = det_boxes[c]
        gt_class = gt_boxes[c]
        npos = num_pos[c]
        dects = sorted(dects, key=lambda conf: conf[4], reverse=True)
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        for d in range(len(dects)):
            iouMax = sys.float_info.min
            if dects[d][-1] in gt_class:
                for j in range(len(gt_class[dects[d][-1]])):
                    iou = Evaluator.iou(dects[d][:4], gt_class[dects[d][-1]][j][:4])
                    if (iou > iouMax):
                        iouMax = iou
                        jmax = j
                if (iouMax >= iou_threshold):
                    if gt_class[dects[d][-1]][jmax][4] == 0:
                        TP[d] = 1
                        gt_class[dects[d][-1]][jmax][4] = 4
                    else:
                        FP[d] = 1
                else:
                    FP[d] = 1
            else:
                FP[d] = 1
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum[TP]
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_FP))
        [ap, mpre, mrec, ii] = Evaluator.calculateAveragePrecision(rec, prec)
        yield ap


if __name__ == '__main__':
    model_ap()