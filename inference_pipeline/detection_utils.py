import cv2
import time
import errno
import os
import numpy as np
import tensorflow as tf
import lanms

from model import EAST_model
from data_processor import restore_rectangle

def get_perspective_transform(box,img):
    # Set width and height of output image
    W, H = 64, 32
    # Define points in input image: top-left, top-right, bottom-right, bottom-left
    pts0 = np.array(box,dtype=np.float32)

    # Define corresponding points in output image
    pts1 = np.float32([[0,0],[W,0],[W,H],[0,H]])

    # Get perspective transform and apply it
    M = cv2.getPerspectiveTransform(pts0,pts1)
    print("M is ",M)
    result = cv2.warpPerspective(img,M,(W,H))
    return result

def get_images(folder_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    files = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if f.split(".")[1] in exts]
    print('Found {} images'.format(len(files)))
    return files
def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]
    
class TextDetectionInference():

    def __init__(self) -> None:
        pass
        
    def detect(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
        '''
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param timer:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        '''
        
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        start = time.time()
        text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
        print('{} text boxes before nms'.format(text_box_restored.shape[0]))
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        # nms part
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
        if boxes.shape[0] == 0:
            return None
        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]

        return boxes
    
    def load_model(checkpoint_folder_path):
        model = EAST_model()
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=model)
        latest_ckpt = tf.train.latest_checkpoint(checkpoint_folder_path)
        if latest_ckpt:
            ckpt.restore(latest_ckpt)
            print('global_step : {}, checkpoint is restored!'.format(int(ckpt.step)))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_folder_path)
        return model