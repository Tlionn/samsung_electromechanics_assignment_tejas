import json
import time

try:
    import redis
except ImportError:
    print("Redis could not be imported")


class RedisClient:
    STATUS_PROCESSING="processing"
    STATUS_FAILED="failed"
    STATUS_PROCESSED="processed"

    def __init__(self, host, port):
        self.redis_client = redis.Redis(host=host, port=port)
        self.connected = self.is_connected()
        

    def is_connected(self):
        try:
            self.redis_client.ping()
            return True
        except:
            print("Could not connect to redis client")
            return False

    
    def create_new_job(self,queue_name, query, session_id, image_id=None, crop_id=None):
        if not self.is_connected():
            return False

        queue_token = "{}:{}".format(queue_name,session_id)
        if image_id is not None:
            queue_token+=":{}".format(image_id)
        if crop_id is not None:
            queue_token+=":{}".format(crop_id)
        print("queueing",queue_token)

        job = {"token": queue_token, "query": query,"session_id":session_id,"image_id":image_id,"crop_id":crop_id}
        job_string = json.dumps(job)
        self.redis_client.lpush(queue_name, job_string)
        self.redis_client.lpush("TOKENS:"+session_id,queue_token) # Just to keep track of all the tokens present in the session
        self.create_meta(queue_token,"start_timestamp",str(time.time()))
        return queue_token

    def create_new_response(self,token,response):
        if not self.is_connected():
            return False
        self.create_meta(token,"end_timestamp",str(time.time()))
        self.redis_client.set(token, json.dumps(response))

    

    def set_status(self,key,status):
        if not self.is_connected():
            return False
        self.redis_client.set("STATUS:"+key,status)
    
    def get_status(self,key):
        if not self.is_connected():
            return False
        status = self.redis_client.get("STATUS:"+key)
        if status is None:
            return "None"
        return str(status.decode("utf-8"))


    def set_session_info(self,session_id,key,info):
        if not self.is_connected():
            return False
        self.redis_client.hset("SESSION:{}".format(session_id),key,info)
    
    def get_session_info(self,session_id,key):
        if not self.is_connected():
            return False
        return self.redis_client.hget("SESSION:{}".format(session_id),key)
        
    

    def get_results_for_tokens(self,tokens,timeout=None,poll_interval=0.001):
        if not self.is_connected():
            return False
        token2results = {}
        start_time=time.time()
        print("Polling for",tokens)
        while len(token2results) < len(tokens):
            for token in tokens:
                if token in token2results:
                    continue
                session_id = token.split(":")[1]
                if self.get_status(session_id) == self.STATUS_FAILED or self.get_status(token) == self.STATUS_FAILED:
                    print("Session failed",token)
                    return False
                result = self.redis_client.get(token)
                if result is not None:
                    if self.get_status(session_id) == self.STATUS_FAILED  or self.get_status(token) == self.STATUS_FAILED:
                        print("Session failed",token)
                        return False
                    token2results[token] = json.loads(result)

                
            time.sleep(poll_interval)
            polling_duration = time.time()-start_time
            if timeout is not None and polling_duration > timeout:
                print("Timed out in",polling_duration)
                return False
        return token2results

    def poll_queue_for_job(self,queue_name,poll_interval=0.5):
        job=None
        while job is None:
            if not self.is_connected():
                return None
            job_string = self.redis_client.rpop(queue_name)
            if job_string is None:
                time.sleep(poll_interval)
                continue
            try:
                job = json.loads(job_string.decode("utf-8"))
            except Exception as e:
                print("Error {}: Could not decode json\n{}".format(str(e),job_string))
        return job

    def create_meta(self,queue_token,meta_key,meta_value): 
        if not self.is_connected():
            return False
        self.redis_client.set("META:"+queue_token,str(meta_value))  

    def register_file(self,session_id,file):
        if not self.is_connected():
            return False
        self.redis_client.lpush("FILES:"+session_id,file)  
    
    def get_files(self,session_id):
        if not self.is_connected():
            return False
        return self.redis_client.lrange("FILES:"+session_id,0,-1)
    
    def create_log(self,session_id,message):
        if not self.is_connected():
            return False
        session_message = self.redis_client.get("MESSAGE:"+session_id)
        if session_message is None:
            session_message=[]
        else:
            try:
                session_message=json.loads(session_message)
            except TypeError:
                print("Trying with decode")
                session_message=json.loads(session_message.decode("utf-8"))
            except Exception as e:
                print("Error occured in loading messages")
                return

        
        session_message = session_message+[{"timestamp":round(time.time(),3),"message":message}]
        self.redis_client.set("MESSAGE:"+session_id,json.dumps(session_message))

    def get_logs(self,session_id):
        if not self.is_connected():
            return False
        logs = self.redis_client.get("MESSAGE:"+session_id)
        try:
            logs=json.loads(logs)
        except TypeError:
            print("Trying with decode")
            logs=json.loads(logs.decode("utf-8"))
        except Exception as e:
            print("Error occured in loading messages")
            return
        return logs
