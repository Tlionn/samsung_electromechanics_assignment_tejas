from flask import Flask, request, abort
from redis_helper import RedisClient
import random
import string
import os
from constants import *

app = Flask(__name__)
redis_client = RedisClient("127.0.0.1", "6379")

@app.route("/ocr", methods=["POST"])
def flask_ocr():
    query = request.json
    id = query.get("id","")
    if not id:
        id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) 
    query["id"] = id
    print(query)
    token = redis_client.create_new_job(
                "MANAGER", query, id
            )
    
    session_response = redis_client.get_results_for_tokens(
            tokens=[token],
            timeout=1000,
            poll_interval=0.1,
        )
    if session_response:
        if DELETE_IMAGES_AFTER_PROCESSING:
            files_to_delete = redis_client.get_files(id)
            for f in files_to_delete:
                if os.path.exists(f):
                    os.remove(f)
        print(session_response)
        return session_response[token]

    else:
        print("Failed getting results for session", id)
        redis_client.set_status(id, redis_client.STATUS_FAILED)
        redis_client.create_log(id, "Failed processing session")
        response = {"status": "failure", "logs": redis_client.get_logs(id)}
        if DELETE_IMAGES_AFTER_PROCESSING:
            files_to_delete = redis_client.get_files(id)
            for f in files_to_delete:
                if os.path.exists(f):
                    os.remove(f)
        return response

if __name__ == "__main__":
    print("Running app on port: {}".format(8080))
    app.run(host="0.0.0.0", port=8080)