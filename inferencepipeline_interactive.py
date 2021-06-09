# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys
from entities_extraction import token_classification_inference as entities_extraction_inference
from relation_extraction import token_classification_inference as relation_extraction_inference

ENTITIES_EXTRACTION_NEMOMODEL_LOCATION = "/entities-and-relationsintext/entities_extraction/model/token_classification_model.nemo"
RELATION_EXTRACTION_NEMOMODEL_LOCATION = "/entities-and-relationsintext/relation_extraction/model/token_classification_model.nemo"

entities_extraction_model, relation_extraction_model = None, None


def load_model():
    global entities_extraction_model, relation_extraction_model, entities_extraction_model, relation_extraction_nemomodel_location
    entities_extraction_model = entities_extraction_inference.load_model(ENTITIES_EXTRACTION_NEMOMODEL_LOCATION)
    relation_extraction_model = relation_extraction_inference.load_model(RELATION_EXTRACTION_NEMOMODEL_LOCATION)
    return entities_extraction_model, relation_extraction_model


def inference_pipeline(input_text):
    global entities_extraction_model, relation_extraction_model
    entities_extraction_inference_queries = [input_text]
    entities_extraction_inference_results = entities_extraction_inference.run_inference(entities_extraction_model, entities_extraction_inference_queries)
    entities_pairs, relation_extraction_inference_results = relation_extraction_inference.run_inference(relation_extraction_model, input_text, entities_extraction_inference_results)
    return entities_extraction_inference_results, entities_pairs, relation_extraction_inference_results
    
    
def print_results(input_text, entities_extraction_inference_results, entities_pairs, relation_extraction_inference_results):
    print("INPUT TEXT: ", input_text)
    if entities_extraction_inference_results and len(entities_extraction_inference_results)!=0:
        print("ENTITIES: ")
        for entity in entities_extraction_inference_results[0]:
            print("    ", entity)
        print("RELATIONS: ")
        for entities_pair, relation_extraction_inference_result in zip(entities_pairs, relation_extraction_inference_results):
            print("    ", entities_pair[0], " AND ", entities_pair[1], " -> ", relation_extraction_inference_result)


def inference_interactive_loop():
    while True:
        os.system('clear') 
        input_text = input("Enter Input Text: ")
        entities_extraction_inference_results, entities_pairs, relation_extraction_inference_results = inference_pipeline(input_text)
        os.system('clear') 
        print_results(input_text, entities_extraction_inference_results, entities_pairs, relation_extraction_inference_results)
        input("\nPress Enter to continue...")
    

def main():
    load_model()
    inference_interactive_loop()
    

if __name__ == '__main__':
    main()