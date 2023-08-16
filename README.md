# bert-triton-truss
Truss for BERT model via Triton backend

### Invoking this model

```
import json
import requests

def invoke_bert_triton_model(text):
   headers = {"Content-Type": "application/json"}

   request = {
      "inputs": [
            {
               "name": "text",
               "datatype": "BYTES",
               "shape": [1, 1],
               "data": [text],
            },
      ],
      "outputs": [{"name": "embedding"}, {"name": "text"}],
   }

   request = json.dumps(request)
   response = requests.post(
      "http://localhost:8000/v2/models/model/infer", data=request, headers=headers
   )

   print(response.text)

   return len(text)

# Test the function
invoke_bert_triton_model("hello world")
```
