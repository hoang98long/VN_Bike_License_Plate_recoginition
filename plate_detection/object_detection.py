import tensorflow as tf
import cv2
import numpy as np
import glob
import os

IMAGE_SIZE = (512, 512, 3)


def load_model(model_directory: str):
    detect_fn = tf.saved_model.load(model_directory)
    return detect_fn


def load_images(image_directory: str, exts=['.jpg', '.jpeg', '.png']) -> []:
    image_names = glob.glob(os.path.join(image_directory, '*.*'))
    images = list(filter(lambda x: os.path.splitext(x)[1] in exts, image_names))
    images = [cv2.imread(im)[..., ::-1] for im in images]
    return image_names, images


def ndarray2tensor(im: np.ndarray):
    return tf.convert_to_tensor(np.expand_dims(im, 0), dtype=tf.uint8)


def predict(model_fn, input_tensor):
    detections = model_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    idx_bests = np.argsort(detections['detection_scores'])[-4:]
    boxes = detections['detection_boxes'][idx_bests]
    boxes = boxes.reshape((-1, 2, 2))
    boxes = boxes[..., ::-1]
    pred_points = np.sum(boxes, axis=1)
    pred_points /= 2
    return pred_points


def cropping(image):
    model_directory = "plate_detection/saved_model"
    model_fn = load_model(model_directory)
    im_tnsr = ndarray2tensor(image)
    pre = predict(model_fn, im_tnsr)
    h, w = image.shape[:2]
    pre[..., 0] *= w
    pre[..., 1] *= h
    result = pre.astype(np.int)
    point = np.array(result, dtype=np.int)

    p = [[point[3][0], point[3][1]], [point[1][0], point[1][1]], [point[2][0], point[2][1]],
         [point[0][0], point[0][1]]]
    p.sort(key=lambda x: x[1])
    p1 = [p[0], p[1]]
    p1.sort(key=lambda x: x[0])
    p2 = [p[2], p[3]]
    p2.sort(key=lambda x: x[0])
    res_p = [p1[0], p1[1], p2[0], p2[1]]
    perspective_transform_img = np.float32([[0, 0], [280, 0], [0, 200], [280, 200]])
    transformed_img = cv2.getPerspectiveTransform(np.float32(res_p), perspective_transform_img)
    cropped_img = cv2.warpPerspective(image, transformed_img, (280, 200))
    return cropped_img

# cropping(cv2.imread("plate2.jpg"))
