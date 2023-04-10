import os
import time
import sys
import copy
import cv2
from redis_helper import RedisClient
from detection_utils import get_images
from constants import *
import numpy as np
from detection_utils import sort_poly
redis_client = RedisClient("127.0.0.1", "6379")
QUEUE = "MANAGER"



images_template = {
"img_path": "",
"detection_boxes": [],
"labels": [],
"img_id": -1,
"crop_paths": []
}

def main_caller(job):
    """Invokes model blocks in the following order.
    1. Text Detection.
    2. Text Recognition.
    """

    session_request = job["query"]
    id = session_request["id"]
    session_response = copy.deepcopy(session_request)

    # 0. Prepare the request based on folder path received
    print("[/] preparing request...")
    files = get_images(session_response["folder_path"])
    images = []
    for idx, file in enumerate(files):
        img = copy.deepcopy(images_template)
        img["img_id"] = idx
        img["img_path"] = file
        images.append(img)
    session_response["images"] = images

    #1. generate tokens for text detection
    text_detection_tokens = []
    for idx, query_image in enumerate(session_response["images"]):

        # push query to respective queues
        text_detection_token = redis_client.create_new_job(
            "TEXT_DETECTION", query_image, id, image_id=query_image["img_id"]
        )
        # save tokens
        text_detection_tokens.append(text_detection_token)

    # 2. get results for text detection
    text_detection_results = redis_client.get_results_for_tokens(
        text_detection_tokens, 500
    )
    all_results = []
    for idx, token in enumerate(text_detection_tokens):
        text_detection_result = text_detection_results[token]
        print(text_detection_result)
        all_results.append(text_detection_result)

    session_response = all_results

    #3. Generate tokens for text recognition
    text_recognition_tokens= []
    for idx, query_image in enumerate(session_response):
        text_recognition_token = redis_client.create_new_job(
            "TEXT_RECOGNITION", query_image, id, image_id=query_image["img_id"]
        )
        # save tokens
        text_recognition_tokens.append(text_recognition_token)
    
    text_recognition_results = redis_client.get_results_for_tokens(
        text_recognition_tokens, 500
    )

    all_results_final = []
    for idx, token in enumerate(text_recognition_tokens):
        text_recognition_result = text_recognition_results[token]
        print(text_recognition_result)
        a = copy.deepcopy(text_recognition_result)
        all_results_final.append(a)

    session_response = {"images":all_results_final}
    print("session_response after recognition",session_response)

    if VIZ_FULL:
        os.makedirs(os.path.join(VIZ_FULL_PATH,id),exist_ok=True)
        for idx,image in enumerate(session_response):
            box_labels = []
            for bidx,boxes in enumerate(image["detection_boxes"]):
                for lidx,labels in enumerate(image["labels"]):
                    if bidx==lidx:
                        box_labels.append((boxes,labels))
                im_name = image["img_path"]
                img = cv2.imread(im_name)
                res_file = os.path.join(
                VIZ_FULL_PATH,
                '{}.txt'.format(
                os.path.basename(im_name).split('.')[0]))
                with open(res_file, 'w') as f:
                    for box,label in box_labels:
                        # to avoid submitting errors
                        box = np.array(box,dtype=np.float32)
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                            continue
                        f.write('{},{},{},{},{},{},{},{}, {}\r\n'.format(
                            box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1], label
                        ))
                        cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                        img = cv2.putText(img=img, text=label, org = (int(box[0, 0]), int(box[0, 1])),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale=1,color=(255,0,0))
            img_path = os.path.join(VIZ_FULL_PATH,id,os.path.basename(im_name))
            cv2.imwrite(img_path, img)

    return session_response

if __name__ == "__main__":
    while True:
        try:
            print("Polling {} for jobs".format(QUEUE))
            if not redis_client.is_connected():
                time.sleep(5)
                continue
            id = None
            redis_client.redis_client.set(QUEUE + "_STATUS", "ACTIVE")
            job = redis_client.poll_queue_for_job(
                queue_name=QUEUE, poll_interval=0.1
            )
            if job is None:
                continue
            print("Received Job:", job)
            response = main_caller(job)
            id = job["token"]

            redis_client.create_new_response(token=id, response=response)
        except Exception as e:
            print("[Error happened]", e)
            redis_client.create_log(
                id, "Error in manager. " + str(e)
            )
            redis_client.set_status(id, redis_client.STATUS_FAILED)