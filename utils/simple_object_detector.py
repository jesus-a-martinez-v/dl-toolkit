import imutils
from keras.applications import imagenet_utils


def sliding_window(image, step, window):
    for y in range(0, image.shape[0] - window[1], step):
        for x in range(0, image.shape[1] - window[0], step):
            yield (x, y, image[y: y + window[1], x: x + window[0]])


def image_pyramid(image, scale=1.5, min_size=(224, 224)):
    yield image

    while True:
        # Compute the dimensions of the next image in the pyramid.
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # If the resized image does not meet the supplied minimum size, then stop the process.
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        yield image


def classify_batch(model, batch_rois, batch_locations, labels, min_prob=0.5, top=10, dimension=(224, 224)):
    # Pass our batch ROIs through our network and decode the predictions
    predictions = model.predict(batch_rois)
    P = imagenet_utils.decode_predictions(predictions, top=top)

    for i in range(0, len(P)):
        for (_, label, prob) in P[i]:
            if prob > min_prob:
                (p_x, p_y) = batch_locations[i]
                box = (p_x, p_y, p_x + dimension[0], p_y + dimension[1])

                L = labels.get(label, [])
                L.append((box, prob))
                labels[label] = L

    return labels
