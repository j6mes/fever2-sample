# Sample FEVER2.0 builder docker image

The FEVER2.0 shared task requires builders to submit Docker images (via dockerhub) as part of the competition to allow 
for adversarial evaluation. Images will host a web server (by installing the [`fever-api`](https://github.com/j6mes/fever-api) pip package).
 
This repository contains an example submission based on an AllenNLP implementation of the system (see [`fever-allennlp`](https://github.com/j6mes/fever-allennlp)). We go into depth for the following key information:

* [Entrypoint](#entrypoint)
* [Web Server](#web-server)
* [Common Data](#common-data)

It can be run with the following commands. The first command creates a dummy container with the shared FEVER data that is used by the submission.

```bash
#Set up the data container (run once on first time)
docker create --name fever-common feverai/common

#Start the server
docker run --rm --volumes-from fever-common -p 5000:5000 feverai/sample
```

## Entrypoint
The submission must run a flask web server. In our application, the entrypoint is a function called `my_sample_fever` in the module `sample_application` (see `sample_application.py`).
The `my_sample_fever` function is a factory that returns a `fever_web_api` object. 

``` python
from fever.api.web_server import fever_web_api

def my_sample_fever():
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

Your dockerfile can then use the `flask run` method as the entrypoint, setting any valid factory as the `FLASK_APP`  

```dockerfile
ENV FLASK_APP sample_application:my_sample_fever
ENTRYPOINT ["flask","run"]
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
We provide common data (the Wikipedia parse and the preprocessed data associated with the first FEVER challenge), that will be mounted in in `/local/common` 

It contains the following files (see [fever.ai/resources.html](https://fever.ai/resources.html) for more info):

```
# Dataset
/local/common/data/fever-data/train.jsonl
/local/common/data/fever-data/paper_dev.jsonl
/local/common/data/fever-data/paper_test.jsonl
/local/common/data/fever-data/shared_task_dev.jsonl
/local/common/data/fever-data/shared_task_test.jsonl

# Preprocessed Wikipedia Dump 
/local/common/data/fever/fever.db

# Wikipedia TF-IDF Index
/local/common/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz

# Preprocessed Wikipedia Pages (Alternative Format)
/local/common/data/wiki-pages/wiki-000.jsonl
...
/local/common/data/wiki-pages/wiki-109.jsonl
```

  