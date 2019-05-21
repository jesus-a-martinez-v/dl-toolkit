import numpy as np


def rank5_accuracy(predictions, labels):
    rank1 = 0
    rank5 = 0

    for (prediction, ground_truth) in zip(predictions, labels):
        prediction = np.argsort(prediction)[::-1]

        if ground_truth in prediction[:5]:
            rank5 += 1

        if ground_truth == prediction[0]:
            rank1 += 1

    rank1 /= float(len(predictions))
    rank5 /= float(len(predictions))

    return rank1, rank5
