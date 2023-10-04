import numpy as np


def softer_nms(dets, confidence=None, thresh=0.01, sigma=0.5, ax=None):

    N = len(dets)
    x1 = dets[:, 0].copy()
    y1 = dets[:, 1].copy()
    x2 = dets[:, 2].copy()
    y2 = dets[:, 3].copy()
    scores = dets[:, 4].copy()
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ious = np.zeros((N, N))
    kls = np.zeros((N, N))
    for i in range(N):
        xx1 = np.maximum(x1[i], x1)
        yy1 = np.maximum(y1[i], y1)
        xx2 = np.minimum(x2[i], x2)
        yy2 = np.minimum(y2[i], y2)

        w = np.maximum(0.0, xx2 - xx1 + 1.)
        h = np.maximum(0.0, yy2 - yy1 + 1.)
        inter = w * h
        ovr = inter / (areas[i] + areas - inter)
        ious[i, :] = ovr.copy()

    i = 0
    while i < N:
        maxpos = dets[i:N, 4].argmax()
        maxpos += i
        dets[[maxpos, i]] = dets[[i, maxpos]]
        confidence[[maxpos, i]] = confidence[[i, maxpos]]
        ious[[maxpos, i]] = ious[[i, maxpos]]
        ious[:, [maxpos, i]] = ious[:, [i, maxpos]]

        ovr_bbox = np.where((ious[i, i:N] > thresh))[0] + i

        pos = i + 1
        while pos < N:
            if ious[i, pos] > 0:
                ovr = ious[i, pos]
                dets[pos, 4] *= np.exp(-(ovr * ovr) / sigma)
                if dets[pos, 4] < 0.001:
                    dets[[pos, N - 1]] = dets[[N - 1, pos]]
                    confidence[[pos, N - 1]] = confidence[[N - 1, pos]]
                    ious[[pos, N - 1]] = ious[[N - 1, pos]]
                    ious[:, [pos, N - 1]] = ious[:, [N - 1, pos]]
                    N -= 1
                    pos -= 1
            pos += 1
        i += 1
    keep = [i for i in range(N)]
    return dets[keep], keep