# Sample FEVER2.0 builder docker image

The FEVER2.0 shared task requires builders to submit Docker images (via dockerhub) as part of the competition to allow 
for adversarial evaluation. Images must contain a single script to make predictions on a given input file using their model and host a web server (by installing the [`fever-api`](https://github.com/j6mes/fever-api) pip package) to allow for interactive evaluation as part of the _breaker_ phase of the competition.
 
This repository contains an example submission based on an AllenNLP implementation of the system (see [`fever-allennlp`](https://github.com/j6mes/fever-allennlp)). We go into depth for the following key information:

* [Prediction Script](#prediction-script)
* [Entrypoint](#entrypoint)
* [Web Server](#web-server)
* [Common Data](#common-data)

It can be run with the following commands. The first command creates a dummy container with the shared FEVER data that is used by the submission.

```bash
#Set up the data container (run once on first time)
docker create --name fever-common feverai/common

#Start a server for interactive querying of the FEVER system via the web API on port 5000
docker run --rm --volumes-from fever-common:ro -e CUDA_DEVICE=-1 -p 5000:5000 feverai/sample

#Alternatively, make predictions on a batch file and output it to `/out/predictions.jsonl` (set CUDA_DEVICE as appropriate)
docker run --rm --volumes-from fever-common:ro -e CUDA_DEVICE=-1 -v $(pwd):/out feverai/sample ./predict.sh /local/fever-common/data/fever-data/paper_dev.jsonl /out/predictions.jsonl
```

### Shared Resources and Fair Use
The FEVER2.0 submissions will be run in a shared environment where resources will be moderated. We urge participants to ensure that these shared resources are respected.

Tensorflow users are asked to implement per-process GPU memory limits: [see this post](https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory). We will set an environment variable `$TF_GPU_MEMORY_FRACTION` that will be tweaked for all systems in phase 2 of the shared task. 


## Prediction Script
The prediction script should take 2 parameters as input: the path to input file to be predicted and the path the output file to be scored:

An optional `CUDA_DEVICE` environment variable should be set  

```bash
#!/usr/bin/env bash

default_cuda_device=0
root_dir=/local/fever-common


python -m fever.evidence.retrieve \
    --index $root_dir/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
    --database $root_dir/data/fever/fever.db \
    --in-file $1 \
    --out-file /tmp/ir.$(basename $1) \
    --max-page 5 \
    --max-sent 5

python -m allennlp.run predict \
    https://jamesthorne.co.uk/fever/fever-da.tar.gz \
    /tmp/ir.$(basename $1) \
    --output-file /tmp/labels.$(basename $1) \
    --predictor fever \
    --include-package fever.reader \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device} \
    --silent

python -m fever.submission.prepare \
    --predicted_labels /tmp/labels.$(basename $1) \
    --predicted_evidence /tmp/ir.$(basename $1) \
    --out_file $2

``` 

## Entrypoint
The submission must run a flask web server to allow for interactive evaluation. In our application, the entrypoint is a function called `my_sample_fever` in the module `sample_application` (see `sample_application.py`).
The `my_sample_fever` function is a factory that returns a `fever_web_api` object. 

``` python
from fever.api.web_server import fever_web_api

def my_sample_fever(*args):
    # Set up and initialize model
    ...
    
    # A prediction function that is called by the API
    def baseline_predict(instances):
        predictions = []
        for instance in instances:
            predictions.append(...prediction for instance...)
        return predictions

    return fever_web_api(baseline_predict)
```

Your dockerfile can then use the `waitress-serve` method as the entrypoint. This will start a wsgi server calling your factory method

```dockerfile
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--call", "sample_application:my_sample_fever"]
``` 


## Web Server
The web server is managed by the `fever-api` package. No setup or modification is required by participants. We use the default flask port of `5000` and host a single endpoint on `/predict`. We recommend using a client such as [Postman](https://www.getpostman.com/) to test your application.


```
POST /predict HTTP/1.1
Host: localhost:5000
Content-Type: application/json

{
	"instances":[
	    {"id":0,"claim":"this is a test claim"}, 
	    {"id":1,"claim":"this is another test claim"}, 
	]
}
```

## API
In our sample submission, we present a simple method `baseline_predict` method. 

```python 
   def baseline_predict(instances):
        predictions = []
        for instance in instances:
            ...prediction for instance...
            predictions.append({"predicted_label":"SUPPORTS", 
                                "predicted_evidence": [(Paris,0),(Paris,5)]})
            
        return predictions
```

Inputs: 

 * `instances` - a list of dictionaries containing a `claim` 

Outputs:

 * A list of dictionaries containing `predicted_label` (string in SUPPORTS/REFUTES/NOT ENOUGH INFO) and `predicted_evidence` (list of `(page_name,line_number)` pairs as defined in [`fever-scorer`](https://github.com/sheffieldnlp/fever-scorer).


## Common Data
We provide common data (the Wikipedia parse and the preprocessed data associated with the first FEVER challenge), that will be mounted in in `/local/fever-common` 

It contains the following files (see [fever.ai/resources.html](https://fever.ai/resources.html) for more info):

```
# Dataset
/local/fever-common/data/fever-data/train.jsonl
/local/fever-common/data/fever-data/paper_dev.jsonl
/local/fever-common/data/fever-data/paper_test.jsonl
/local/fever-common/data/fever-data/shared_task_dev.jsonl
/local/fever-common/data/fever-data/shared_task_test.jsonl

# Preprocessed Wikipedia Dump 
/local/fever-common/data/fever/fever.db

# Wikipedia TF-IDF Index
/local/fever-common/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz

# Preprocessed Wikipedia Pages (Alternative Format)
/local/fever-common/data/wiki-pages/wiki-000.jsonl
...
/local/fever-common/data/wiki-pages/wiki-109.jsonl
```

  
