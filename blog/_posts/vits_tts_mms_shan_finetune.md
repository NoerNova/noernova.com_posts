---
tags: ["code", "note"]
date: "June 6, 2024"
title: "Finetune Vits TTS MMS for Shan language"
subtitle: ""
image: https://raw.githubusercontent.com/NoerNova/noernova.com_posts/main/blog/assets/vits_tts_mms_shan_finetune/finetune_tts_mms.webp
link: "blog/vits_tts_mms_shan_finetune"
description: "Note for fintunning VITS base TTSMMS for Shan language"
---
## Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Prepare Dataset, dataset formatting](#prepare-dataset-dataset-formatting)
- [Tokenizer](#tokenizer)
- [Finetune](#finetune)
- [Inference](#inference)

## Introduction

Meta's Facebook research fairseq produce [Massively Multilingual Speech (MMS) project](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) support to 1,000+ of speech-language including Shan.

I was playing around with the TTS (Text-to-speech) model and it's quite impressive, the size of the model is small, and it gives good-quality results.

However, the pronunciation of some words is still unnatural and missing.

I found some consonants are missing in the tokenizer (vocab.json file), somehow the dataset used in pre-train is in Shan's old script before additional consonants are included.

> The number of consonants in a textbook may vary: there are 19 universally accepted Shan consonants (ၵ ၶ င ၸ သ ၺ တ ထ ၼ ပ ၽ ၾ မ ယ ရ လ ဝ ႁ ဢ) and five more which represent sounds not found in Shan, g, z, b, d and th [θ]. These five (ၷ ၹ ၿ ၻ ႀ) are quite rare. In addition, most editors include a dummy consonant (ဢ) used in words with a vowel onset. A textbook may therefore present 18-24 consonants. [wikipedia.org](https://en.wikipedia.org/wiki/Shan_alphabet)

"ၷ", "ၹ", "ၻ", "ၾ", "ႊ" consonants are missing from the tokenizer, especially "ၾ" which is now widely used in the new Shan script.

## Requirements

- Recomment to setup python environment such as [venv](https://docs.python.org/3/library/venv.html), [Anaconda](https://www.anaconda.com/), [Miniconda](https://docs.anaconda.com/free/miniconda/)

- CUDA and NVIDIA graphic card is highly recomment, also up-to-date Nvidia driver.

## Prepare Dataset, dataset formatting

From [coqui.ai](https://docs.coqui.ai/en/latest/what_makes_a_good_dataset.html) document recommantation for "What makes a good TTS dataset".

We needed

- 3-10 Sec per audio clip, Naturalness recordings.
- 16000-22050 Sample-rate (MMS pretrained model used 16kHz)

As a Low-Resource language like Shan, there are no ready-used datasets yet, we have to create a good one on our own.

There are some public audio sources for Shan language online but I choose to grep text from a website [tainovel.com](https://tainovel.com/) and split for sentences, then record my own voice reading those sentences, it would be easier and faster than splitting audio and transcript them.

### Dataset format

Dataset sould place along with audio and it's metadata

```bash
/dataset
 - metadata.csv
 - /audio-data
    - /train
        - /audio1.wav
```

Metadata.csv

```bash
file_name,transcription
audio-data/train/audio1.wav,ဢမ်ႇတႄႇႁဵတ်းမိူဝ်ႈၼႆႉ မိူဝ်ႈၽုၵ်ႈၵေႃႈ...
audio-data/train/audio2.wav,တွၼ်ႈသိပ်းသၢမ် ယႃႈမဝ်းၵမ်...
audio-data/train/audio3.wav,တွၼ်ႈတႃႇၸဝ်ႈၵဝ်ႇလႆႈယူႇလီၵိၼ်လီသေ...
...
```

### Save dataset

We can both upload our dataset to huggingface or use it local, but **for using local we have to modify some finetune code.**

```bash
pip install datasets huggingface_hub
```

```python
from datasets import load_dataset, Audio

dataset = load_dataset("audiofolder", data_dir="<dataset-path>")
dataset = dataset.cast_column("audio", Audio(sampling_rate=22050))
```

To push to huggingface

```python
from huggingface_hub import login
login()
```

```python
model_id = <your_model_id>
dataset.push_to_hub(model_id, private=True)
```

To save local

```python
model_id = <your_model_id>
dataset.save_to_disk(model_id)
```

## Finetune Environment

### Following instruction from [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits)

clone finetune project

```bash
git clone git@github.com:ylacombe/finetune-hf-vits.git
cd finetune-hf-vits
pip install -r requirements.txt
```

Link hugging face account for pull/push model

```bash
git config --global credential.helper store
huggingface-cli login
```

Build the monotonic alignment search function

```bash
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..
```

Download checkpoint for ```shn``` (ISO 693-3 language code)

```bash
cd <path-to-finetune-hf-vits-repo>

python convert_original_discriminator_checkpoint.py --language_code shn --pytorch_dump_folder_path <local-folder> --push_to_hub <repo-id-you-want>
```

 The model will also be pushed to your hub repository ```<your HF handle>/<repo-id-you-want>```. Simply remove --push_to_hub ```<repo-id-you-want>``` if you don't want to push to the hub.

## Tokenizer

As refer before we have to modify model's tokenizer for additional character.

Load tokenizer from our previous checkpoint.

```python
from transformers import VitsTokenizer

save_tokenizer_path = <your_save_tokenizer_path>

tokenizer = VitsTokenizer.from_pretrained(model_id)

tokenizer.save_pretrained()
```

Then edit vocal.json file by adding missing characters.

```json
{
  " ": 43,
  "'": 40,
  "-": 34,
  "|": 0,
  "င": 11,
  "တ": 9,
  "ထ": 36,
  "ပ": 20,
  ...
  // added new_token
  "ၷ": 44,
  "ၹ": 45,
  "ၻ": 46,
  "ၾ": 47,
  "ႊ": 48
}
```

### Modify Embedding tokens and weight

Load Model from checkpoint and tokenizer from modified_tokenizer.

```python
from transformers import VitsTokenizer, VitsModel

checkpoint_model = <your_saved_checkpoint_path>
modified_tokenizer = <your_saved_modified_tokenizer_path>

# Load the VITS MMS TTS tokenizer
model = VitsModel.from_pretrained(checkpoint_model)
tokenizer = VitsTokenizer.from_pretrained(modified_tokenizer)

# Extend the tokenizer's vocabulary with the additional characters
new_tokens = ["ၷ", "ၹ", "ၻ", "ၾ", "ႊ"]
```

```python
import torch.nn as nn

# print(model.text_encoder.embed_tokens)

class VitsModel(nn.Module):
    def __init__(self, model, tokenizer):
        super(VitsModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        
        # Assume `embeddings` is the original embedding layer in the VITS model
        old_embeddings = model.text_encoder.embed_tokens
        old_embedding_weight = old_embeddings.weight.data
        
        # Define new embedding layer with updated size
        new_embedding_layer = nn.Embedding(len(tokenizer) - 1, old_embedding_weight.shape[1])
        
        # Copy old weights into the new embedding layer
        new_embedding_layer.weight.data[:old_embedding_weight.size(0), :] = old_embedding_weight
        
        # Initialize new token embeddings (e.g., with the mean of existing ones)
        new_token_embeddings = old_embedding_weight.mean(dim=0, keepdim=True).repeat(len(new_tokens), 1)
        new_embedding_layer.weight.data[-len(new_tokens):, :] = new_token_embeddings
        
        # Replace the embedding layer in the model
        self.model.text_encoder.embed_tokens = new_embedding_layer
    
    def forward(self, input_ids, *args, **kwargs):
        # Use the modified embedding layer and pass through the model
        embeddings = self.model.text_encoder.embed_tokens(input_ids)
        outputs = self.model(input_ids_embeds=embeddings, *args, **kwargs)
        return outputs

# Create a new model instance with modified embeddings
VitsModel(model, tokenizer)
```

```python
VitsModel(
  (model): VitsModel(
    (text_encoder): VitsTextEncoder(
      (embed_tokens): Embedding(49, 192) <-- now we should have 49 Embedding weight instead of 44
      (encoder): VitsEncoder(
        (layers): ModuleList(
            ...
```

Save new Model and Tokenizer

```python
repo_name = "shn-embeddings-token-model" # or any name prefer.

model.save_pretrained(repo_name)
tokenizer.save_pretrained(repo_name)
```

## Finetune

Now we need just a couple of process to finetune our model

### Config file

```json
{
  "project_name": <your_project_name>,
  "push_to_hub": false, // or true to push to huggingface_hub, login credential require
  "hub_model_id": <your_hub_model_id>,
  "report_to": ["wandb"], // remove if you don't want to virtualize train process
  "overwrite_output_dir": true,
  "output_dir": <your_output_dir>,

  "dataset_name": <your_dataset_id>, // huggingface id or "./mms-tts-datasets/train" for local
  "audio_column_name": "audio",
  "text_column_name": "transcription",
  "train_split_name": "train",
  "eval_split_name": "train",

  "full_generation_sample_text": "ႁႃႇလႄႈၾူၼ်လူင်ဢူၺ် လမ်းလႅင်ႉလူင်ထူဝ်းပဝ်ႇသႂ်ႇ ၾႃႉၾူၼ်ၵမ်ႇလမ်မႃး ၸွမ်းၾင်ႇၼမ်ႉၾင်ႇၼွင်",

  "max_duration_in_seconds": 20,
  "min_duration_in_seconds": 1.0,
  "max_tokens_length": 500,

  "model_name_or_path": <your_model_id>, // huggingface id or "./<your_saved_modified_embeddings-token-model>",

  "preprocessing_num_workers": 4,

  "do_train": true,
  "num_train_epochs": 200,
  "gradient_accumulation_steps": 1,
  "gradient_checkpointing": false,
  "per_device_train_batch_size": 32, // <-- decrease this parameter if you have less VRAM
  "learning_rate": 2e-5,
  "adam_beta1": 0.8,
  "adam_beta2": 0.99,
  "warmup_ratio": 0.01,
  "group_by_length": false,

  "do_eval": true,
  "eval_steps": 50,
  "per_device_eval_batch_size": 32, // <-- decrease this parameter if you have less VRAM
  "max_eval_samples": 25,
  "do_step_schedule_per_epoch": true,

  "weight_disc": 3,
  "weight_fmaps": 1,
  "weight_gen": 1,
  "weight_kl": 1.5,
  "weight_duration": 1,
  "weight_mel": 35,

  "fp16": true, // <-- remove this line if you don't have CUDA or NVIDIA graphic card
  "seed": 456
}
```

Save config file to <any_name>.json

### Run finetune

```bash
accelerate launch run_vits_finetuning.py ./<your_saved_config>.json
```

if you got ```AttributeError: 'NoneType' object has no attribute 'to'```, try

```bash
pip uninstall transformers datasets accelerate # remove the ones installed when you run pip install -r requirements.txt

pip install transformers==4.35.1 datasets[audio]==2.14.7 accelerate==0.24.1
```

## Inference

To read digit in Shan word install ShanNLP

```bash
pip install git+https://github.com/NoerNova/ShanNLP
```

```python
from transformers import VitsModel, VitsTokenizer, set_seed
import torch
from shannlp import util, word_tokenize

def preprocess_string(input_string: str):
    string_token = word_tokenize(input_string)
    num_to_shanword = util.num_to_shanword

    result = []
    for token in string_token:
        if token.strip().isdigit():
            result.append(num_to_shanword(int(token)))
        else:
            result.append(token)

    full_token = ''.join(result)
    return full_token

model_name = "./Finetune/vits_mms_finetune/models/mms-tts-nova-train"
model = VitsModel.from_pretrained(model_name)
tokenizer = VitsTokenizer.from_pretrained(model_name)

text = """မိူဝ်ႈပီ 1958 လိူၼ်မေႊ 21 ဝၼ်းၼၼ်ႉ ၸဝ်ႈၼွႆႉသေႃးယၼ်ႇတ ဢမ်ႇၼၼ် ၸဝ်ႈၼွႆႉ ဢွၼ်ႁူဝ် ၽူႈႁၵ်ႉၸိူဝ်ႉၸၢတ်ႈ 31 ၵေႃႉသေ တိူင်ႇၵၢဝ်ႇယၼ်ႇၸႂ် ၵိၼ်ၼမ်ႉသတ်ႉၸႃႇ တႃႇၵေႃႇတင်ႈပူၵ်းပွင် ၵၢၼ်လုၵ်ႉၽိုၼ်ႉ တီႈႁူၺ်ႈပူႉ ႁိမ်းသူပ်းၼမ်ႉၵျွတ်ႈ ၼႂ်းဢိူင်ႇမိူင်းႁၢင် ၸႄႈဝဵင်းမိူင်းတူၼ် ၸိုင်ႈတႆးပွတ်းဢွၵ်ႇၶူင်း လႅၼ်လိၼ်ၸိုင်ႈထႆး။"""

processed_string = preprocess_string(text)
inputs = tokenizer(processed_string, return_tensors="pt")
set_seed(456)

model.speaking_rate = 1.2
model.noise_scale = 0.8

with torch.no_grad():
    output = model(**inputs)

waveform = output.waveform[0]
```

[Sample Audio](https://github.com/NoerNova/noernova.com_note/raw/main/assets/finetuned_output.wav)

## Conclusion

The key challenge in perfecting Shan Text-to-Speech lies in developing a robust dataset. This process demands considerable time, meticulous attention, sufficient resources, and the resolution of language conflicts.
