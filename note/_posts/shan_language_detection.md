---
tag: ["code"]
date: "May 28, 2024"
title: "Shan language text predictor"
link: "note/shan_language_detection"
description: "Predict if text is in Shan or not"
---

### Using [Meta's fastext](https://fasttext.cc/) model

```python
import fasttext
from huggingface_hub import hf_hub_download

# download model and get the model path
# cache_dir is the path to the folder where the downloaded model will be stored/cached.
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir="./tmp/fasttext")
print("model path:", model_path)

# load the model
model = fasttext.load_model(model_path)

"""Language Identification"""

def is_shan_lang(title):
    predict = model.predict(title)
    lable = predict[0][0]
    result = lable.replace('__label__', '')

    if result != 'shn_Mymr':
        return False
    
    return True

```
