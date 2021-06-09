# Identifying Entities And Their Relations In Text

## Overview
Identifying entities and their relations in text is useful for multiple NLP tasks, for example creating Knowledge Graphs, Text Summarization, Question Answering, etc.  
This blog explains 

## Training:
There are 2 components, both of which are trained independently: 
 
### 1 Entities Extraction:
Here are the steps to run training for the Entities Extraction component:
1. Clone the entities-and-relationsintext repo:  
``git clone -b mediumblog1 https://github.com/NikhilSrihari/entities-and-relationsintext.git``  
2. Clone the NeMo repo:  
``git clone -b v1.0.0 https://github.com/NVIDIA/NeMo.git``  
3. Starting Docker container:   
``docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo -v <entities-and-relationsintext_github_folder>:/entities-and-relationsintext --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:21.03-py3``  
3. Installing NeMo and other dependencies:  
``cd /NeMo``  
``apt-get update && apt-get install -y libsndfile1 ffmpeg``  
``pip install Cython``  
``python -m pip install git+https://github.com/NVIDIA/NeMo.git@v1.0.0#egg=nemo_toolkit[all]``  
``pip install jsonlines``  
4. Training Data:  
Place the training data (spread across 4 files ``text_train.txt``, ``text_dev.txt``, ``labels_train.txt`` and ``labels_dev.txt``), following the format as described in the blog, at ``/entities-and-relationsintext/entities_extraction/data/``.    
5. Run Training:  
``cd /entities-and-relationsintext/entities_extraction/``  
``python token_classification.py``

### 2 Relation Extraction:
Here are the steps to run training for the Entities Extraction component:
1. Clone the entities-and-relationsintext repo:  
``git clone -b mediumblog1 https://github.com/NikhilSrihari/entities-and-relationsintext.git``  
2. Clone the NeMo repo:  
``git clone -b v1.0.0 https://github.com/NVIDIA/NeMo.git``  
3. Starting Docker container:   
``docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo -v <entities-and-relationsintext_github_folder>:/entities-and-relationsintext --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:21.03-py3``  
3. Installing NeMo and other dependencies:  
``cd /NeMo``  
``apt-get update && apt-get install -y libsndfile1 ffmpeg``  
``pip install Cython``  
``python -m pip install git+https://github.com/NVIDIA/NeMo.git@v1.0.0#egg=nemo_toolkit[all]``  
``pip install jsonlines``  
4. Training Data:  
Place the training data (spread across 4 files ``text_train.txt``, ``text_dev.txt``, ``labels_train.txt`` and ``labels_dev.txt``), following the format as described in the blog, at ``/entities-and-relationsintext/relation_extraction/data/``.    
5. Run Training:  
``cd /entities-and-relationsintext/relation_extraction/``  
``python token_classification.py``

## Inference:
Here are the steps to run inference:
1. Clone the entities-and-relationsintext repo:  
``git clone -b mediumblog1 https://github.com/NikhilSrihari/entities-and-relationsintext.git``  
2. Clone the NeMo repo:  
``git clone -b v1.0.0 https://github.com/NVIDIA/NeMo.git``  
3. Starting Docker container:   
``docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo -v <entities-and-relationsintext_github_folder>:/entities-and-relationsintext --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:21.03-py3``  
3. Installing NeMo and other dependencies:  
``cd /NeMo``  
``apt-get update && apt-get install -y libsndfile1 ffmpeg``  
``pip install Cython``  
``python -m pip install git+https://github.com/NVIDIA/NeMo.git@v1.0.0#egg=nemo_toolkit[all]``  
``pip install jsonlines``  
4. Downloading trained checkpoints:
The 2 trained checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/11G0p1uiSS2vqBnSd37KyACPuEiM3WyxA?usp=sharing). Merge the downloaded folder with the entities-and-relationsintext repo, such that the models are present at ``entities-and-relationsintext/entities_extraction/model/token_classification_model.nemo`` and ``entities-and-relationsintext/relation_extraction/model/token_classification_model.nemo``  
5. Run Inference:  
``cd /entities-and-relationsintext``  
``python inferencepipeline_interactive.py``