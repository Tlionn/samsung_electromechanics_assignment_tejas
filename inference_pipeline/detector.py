import cv2
import os
import numpy as np
import tensorflow as tf
from detection_utils import *
from constants import *
import copy
from redis_helper import RedisClient

redis_client = RedisClient("127.0.0.1", "6379")

QUEUE = "TEXT_DETECTION"
model = TextDetectionInference.load_model(checkpoint_folder_path=DETECTION_MODEL_PATH)

print("Model Loaded")

def get_predictions(job):

    query = job["query"]
    image = query["img_path"]
    id = job["token"].split(":")[1]
    image_id = query["img_id"]
    #Image operations
    img = cv2.imread(image)[:, :, ::-1]
    img_resized, (ratio_h, ratio_w) = resize_image(img)
    img_resized = (img_resized / 127.5) - 1

    #Forawrd pass
    score_map, geo_map = model.predict(img_resized[np.newaxis, :, :, :])
    boxes = TextDetectionInference.detect(score_map,geo_map)

    if boxes is not None:
      boxes = boxes[:, :8].reshape((-1, 4, 2))
      boxes[:, :, 0] /= ratio_w
      boxes[:, :, 1] /= ratio_h

    #Save crops to local folder
    os.makedirs(os.path.join(CROPS_SAVE_PATH,id,str(image_id)),exist_ok=True)
    crop_paths = []
    for i,box in enumerate(boxes):
        result = get_perspective_transform(box,img)
        save_path = os.path.join(CROPS_SAVE_PATH,id,str(image_id),f"{str(i)}.jpg")
        crop_paths.append(save_path)
        cv2.imwrite(save_path,result)
    
    query["detection_boxes"] = boxes.tolist()
    query["crop_paths"] = crop_paths

    job["response"] = query

    if DETECTION_VIZ:
        os.makedirs(os.path.join(DETECTION_VIZ_PATH,id),exist_ok=True)
        if boxes is not None:
            res_file = os.path.join(
                DETECTION_VIZ_PATH,
                '{}.txt'.format(
                os.path.basename(image).split('.')[0]))

            with open(res_file, 'w') as f:
                for box in boxes:
                # to avoid submitting errors
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                    ))
                    cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)

        img_path = os.path.join(DETECTION_VIZ_PATH, id, os.path.basename(image))
        cv2.imwrite(img_path, img[:, :, ::-1])
    return job["response"]

if __name__ == "__main__":
    while True:
        try:
            print("Polling text detector for jobs".format(QUEUE))
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
            response = get_predictions(job)
            id = job["token"]
            redis_client.create_new_response(token=id, response=response)
        except Exception as e:

            print("[Error happened]", e)
            redis_client.create_log(
                id, "Error in text detection. " + str(e)
            )
            redis_client.set_status(id, redis_client.STATUS_FAILED)

    

    
    


