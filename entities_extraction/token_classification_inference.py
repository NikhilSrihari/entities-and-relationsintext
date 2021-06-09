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

from nemo.collections.nlp.models import TokenClassificationModel

labels = ["[B-ENT]", "[I-ENT]"]

def load_model(nemomodel_location):
    model = TokenClassificationModel.restore_from(nemomodel_location)
    return model

def model_predictions(model, queries):
    results = model.add_predictions(queries)
    return results

def postprocessing(results):
    processed_results = []
    for result in results:
        processed_result = []
        result_split = result.split()
        i = 0
        while i<len(result_split):
            if result_split[i].endswith(labels[0]):
                processed_result.append(result_split[i].replace(labels[0], ""))
            elif result_split[i].endswith(labels[1]):
                processed_result[-1] = processed_result[-1] + " " + result_split[i].replace(labels[1], "")
            i+=1
        processed_results.append(processed_result)
    return processed_results
    
def run_inference(model, queries):
    results = model_predictions(model, queries)
    processed_results = postprocessing(results)
    return processed_results