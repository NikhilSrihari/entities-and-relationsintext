# Identifying Entities And Their Relations In Text

## Overview
Identifying entities and their relations in text is useful for multiple NLP tasks, for example creating Knowledge Graphs, Text Summarization, Question Answering, etc. 
Please refer to this doc  

## Inference:
Here are the steps to run inference:
1. Clone this model repo:  
``git clone -b mediumblog1 ``
2. Starting the Docker container:  
``docker run --gpus all -it --rm -v /mnt/dldata/nsrihari/GauGan/mediumblog1/gaugan:/gaugan --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd nvcr.io/nvidia/nemo:1.0.0rc1``
3. Installing dependencies:  
``pip install jsonlines``
4. Running Inference:  
``cd /gaugan && python inferencepipeline_interactive.py ``

## Training:
There are 2 components, both of which are trained independently:
###1 Entities Extraction:
Here are the steps to run training for the Entities Extraction component:
1. Clone this model repo:  
``git clone -b mediumblog1 ``

###2 Relation Extraction:
Here are the steps to run training for the Relation Extraction component:
1. Clone this model repo:  
``git clone -b mediumblog1 ``
2. Starting the Docker container: 
``docker run --gpus all -it --rm -v /mnt/dldata/nsrihari/GauGan/mediumblog1/gaugan:/gaugan --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd nvcr.io/nvidia/nemo:1.0.0rc1``
3. Installing dependencies:  
``pip install jsonlines``
4. Preprocess data and create data that NeMo needs. 
Make sure that the ``captions_sng.jsonl`` file exists in ``\NeMo\data`` folders. Then run: <br>  
``cd /NeMo/data && python preprocessing.py``  <br>
This will create 4 files: text_train.txt, text_dev.txt, labels_train.txt and labels_dev.txt 
5. Run training
``cd /gaugan && python relation_extraction/token_classification.py ``