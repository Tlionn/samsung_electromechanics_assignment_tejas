import pandas as pd
from tqdm import tqdm
from mltu.configs import BaseModelConfigs
from recognition_utils import *
from constants import *
import copy
import os
from redis_helper import RedisClient
import time

redis_client = RedisClient("127.0.0.1", "6379")
QUEUE = "TEXT_RECOGNITION"
configs = BaseModelConfigs.load(CONFIG_YAML_PATH)
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
print("Model Loaded")

def get_predictions(job):
    query = job["query"]
    crop_paths = query["crop_paths"]
    img_id = query["img_id"]
    id = job["token"]
    print("Number of crops for image is",len(crop_paths))
    
    if len(crop_paths)>0:
        labels = []
        for crop_path in crop_paths:
            image = cv2.imread(crop_path)
            prediction_text = model.predict(image)
            labels.append(prediction_text)
            if RECOGNITION_VIZ:
                os.makedirs(os.path.join(RECOGNITION_VIZ_PATH,id),exist_ok=True)
                cv2.imwrite(os.path.join(RECOGNITION_VIZ_PATH,id,f"{prediction_text}.jpg"),image)
    query["labels"] = labels
    print(query)
    return query

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
            response = get_predictions(job)
            id = job["token"]
            redis_client.create_new_response(token=id, response=response)
        except Exception as e:

            print("[Error happened]", e)
            redis_client.create_log(
                id, "Error in text recognition. " + str(e)
            )
            redis_client.set_status(id, redis_client.STATUS_FAILED)




