# Table of contents
1. [Repo Structure](#folder_structure)
2. [Training Repos](#training_repos)
    1. [EAST-tf2](#easttf2)
    2. [keras_detection](#kerasdetection)
3. [Inference Pipeline](#inf_pip)
4. [Deployment Details](#dep_det)

## Repo Structure <a name="folder_structure"></a>
A brief on the repo structure is as follows. A detailed account for each main folder will be provided later.  
    1. inference_pipeline- the actual deployable pipeline \
    2. training_repos- the repos/scripts used for training text detection and recognition models. \
    3. live_coding_r2_dayananda_nc.ipynb- notebook used in live coding portion of Round 2 interview. \
    4. README.md- this file.

## Training Repos <a name="training_repos"></a>
Text detection and recognition models were trained using two different repositories. A brief account and how to use is presented below. 
### EAST-tf2 <a name="easttf2"></a>
EAST is one of the SOTA models currently used for text detection purposes. A tensorflow2 implementation of the same has been used to train a model. 
**Dataset used**  
ICDAR-2015 (dataset is very small and is included as part of the repository). \
1k training images, 500 test images.
Annotation type: text file (whose name is same as the image file) containing box co-ordinates and corresponding labels. One line corresponds to one text instance. \
**Training** \
Command: \
python3 train.py --training_data_path=./data/ICDAR2015/train_data/ --checkpoint_path=./east_resnet_50_rbox

The author suggests training the model for at least 50k steps in order to see some meaningful detections. Due to time constraints, the model was trained for 59300 steps. Training for 100,000 steps would have produced much better accuracy. 

Checkpoints are created every 50 steps (using checkpoint callback).

Results can be seen in training_repos/EAST-tf2/data/ICDAR2015/test_data_output. \

### keras_detection <a name="kerasdetection"></a>
This was a small experiment to see how keras detector performs and will not be explained in detail. \
**Dataset used**   
Synthetic data was generated. Backgrounds and random text was used to create the same.  
**Training**
Keras takes care of initializing the model and creating the data pipeline as well. h5 models are saved after every few iterations. The notebook in this directory performs synthetic data generation and trains a detector. 

### 01_image_to_word
This repo is used to train a custom model that makes use of residual and bidirectional LSTM layers.  
**Dataset Used**
The dataset used for training the text recognition model is the 90kDICT32px dataset, which is a huge dataset (10GB). The image crops are not based on scene text, so accuracy on such images will be poor. 
**Training**
Command: python train.py  
All parameters are configured inside the train script. Paths would need to be changed to point to the data and working directories.  
Whenever the validation character error rate decreases, the correponding model is saved in h5 format along with the corresponding logs and configs.  
When the training ends, an onnx model is also created. This is used for inference.   
**Evaluation**  
Command: python inferenceModel.py  
The model was evaluated on 5000 crops and gave an average character error rate of 6.7%.  

## Inference Pipeline <a name="inf_pip"></a>
A flask service was created using the models trained. The deployment was done using redis and PM2. The main components of the pipeline are as follows:    
    1. **flask_service**: the script where the flask service is actually hosted. Accepts requests and sends the same to the manager worker.  
    2. **manager**: this script calls the main components of the project- the text detection and recognition models. It also initializes appropriate responses.  
    3. **detector**: this script loads the text detection model and waits for queries and serves the same as they come. It is responsible for detecting text in a given image and saving the crops.  
    4. **recognizer**: this script loads the text recognition model and waits for queries and serves them as they come. For a given image, it loads all the crops detected by the detector and provides labels for the same.  
    5. ***_utils**: these scripts contain all the helper scripts or classes that would be needed to load models, perform inference, etc. This is done to keep the scripts legible and easy to read.  

**Input**  
The service, once started, can be reached using the following URL: "0.0.0.0:8080/ocr" using the same machine. To use from a remote machine, 0.0.0.0 would have to be replaced with the IP of the machine the service is hosted on.  
The query is made very simple:  
{'folder_path': '/opt/test_folder'}  
It just expects the path to a folder containing images. This was chosen to keep deployment simple.  
**Output**  
The service outputs a dictionary with just one key: "images". This in turn is a list of dictionaries. For each image in the given folder path, the path to the image, boxes output by the detector, labels output by the recognizer and an ID is stored.  
A helper script can be used to visualize the final output. However, visualization is enabled at every stage. You can just change the corresponding flags in constants.py. Detection outputs are stored as id.jpg (where id is just the value of an iterator) and has boxes drawn on it. Recognition outputs are stored as label.jpg. The final visualization (detection+recognition) is also stored as id.jpg, with boxes and captions on top.  
Each request is associated with a session ID that is internally generated, and outputs for respective sessions are therefore isolated. This architecture allows for additional post processing rules to be added. One can check raw text detection recognition outputs as well as the final results once all post processing is applied. 

An example output for a folder containing 2 images is shown below:

{"images" : [{'img_path': '/opt/test_folder/smallest-font-size-images.jpg', 'detection_boxes': [[[503.84881591796875, 472.1914978027344], [714.6348876953125, 462.4824523925781], [716.6830444335938, 505.09295654296875], [505.89691162109375, 514.8031005859375]], [[125.60707092285156, 108.12259674072266], [327.3220520019531, 95.64998626708984], [331.1110534667969, 154.76116943359375], [129.39588928222656, 167.23336791992188]], [[485.7228698730469, 216.3488006591797], [666.1574096679688, 226.7913055419922], [663.4014282226562, 269.8768310546875], [482.9669494628906, 259.4333801269531]], [[665.5155029296875, 220.09397888183594], [830.8433837890625, 203.64878845214844], [836.4010620117188, 257.4009704589844], [671.0734252929688, 273.84613037109375]], [[509.5566101074219, 101.62238311767578], [701.0176391601562, 96.05329132080078], [702.2914428710938, 138.12100219726562], [510.8302917480469, 143.69000244140625]], [[486.3559265136719, 348.93853759765625], [716.4520263671875, 339.349853515625], [718.2731323242188, 382.4053039550781], [488.17755126953125, 391.9941711425781]], [[673.9465942382812, 342.1642150878906], [874.309326171875, 333.25836181640625], [876.9695434570312, 385.1495056152344], [676.60693359375, 394.0552673339844]], [[398.51666259765625, 563.1774291992188], [493.7222595214844, 559.5775756835938], [494.59326171875, 581.91064453125], [399.38763427734375, 585.5104370117188]], [[469.1988525390625, 557.4662475585938], [568.490478515625, 560.9746704101562], [567.6572265625, 583.9569702148438], [468.365478515625, 580.4486083984375]], [[445.6435241699219, 459.04266357421875], [564.5621948242188, 464.0471496582031], [562.416015625, 514.4073486328125], [443.49749755859375, 509.40289306640625]]], 'labels': ['otRocKw', 'Icllo', 'ptfutur', 'abold', 'otbodon', 'Ipthelveti', 'eficalich', 'Zsptline', 'neweight', 'Jopt'], 'img_id': 0, 'crop_paths': ['/opt/output_data/crops/3OYSB/0/0.jpg', '/opt/output_data/crops/3OYSB/0/1.jpg', '/opt/output_data/crops/3OYSB/0/2.jpg', '/opt/output_data/crops/3OYSB/0/3.jpg', '/opt/output_data/crops/3OYSB/0/4.jpg', '/opt/output_data/crops/3OYSB/0/5.jpg', '/opt/output_data/crops/3OYSB/0/6.jpg', '/opt/output_data/crops/3OYSB/0/7.jpg', '/opt/output_data/crops/3OYSB/0/8.jpg', '/opt/output_data/crops/3OYSB/0/9.jpg']}]}

## Deployment Details <a name="dep_det"></a>
Redis is used to queue jobs from flask to manager, manager to text detector and recognizer.  
PM2 is used to deploy the scripts. Following commands can be used to deploy the pipeline:  
pm2 start flask_service.py --name flask -- interpreter <path_to_env>  
pm2 start manager.py --name manager -- interpreter <path_to_env>   
pm2 start detector.py --name text_detection -- interpreter <path_to_env>   
pm2 recognizer.py --name text_recognition -- interpreter <path_to_env>    
I used two seperate envs for text detection and recognition. The env used for text detection was also used for flask and manager.  

Please note: lanms requires python3.8. Otherwise, compilation does not complete and the pipeline will not run. When compiling make sure that Python.h is visible in PATH (you can simply use python3.8-dev without any virtual env). 