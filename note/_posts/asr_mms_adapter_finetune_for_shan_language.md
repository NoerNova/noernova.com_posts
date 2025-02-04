---
tags: ["code"]
date: "June 16, 2024"
title: "Finetune Asr MMS adapter for Shan language"
link: "note/asr_mms_adapter_finetune_for_shan_language"
description: "Note for fintunning ASRMMS for Shan language"
---

## Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Prepare Dataset, dataset formatting](#prepare-dataset-dataset-formatting)
- [Notebook environment](#notebook-environment)
- [Dataset](#datasets)
- [Tokenizer and Vocab extract](#tokenizer-and-vocab-extract)
- [Feature Extractor](#feature-extractor)
- [Training](#training)
- [Save model or push to hub](#save-model-or-push-to-hub)
- [Inference](#inference)
- [Conclusion](#conclusion)

## Introduction

Meta's Facebook research fairseq produce [Massively Multilingual Speech (MMS) project](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) support to 1,000+ of speech-language including Shan.

ASR (Automatic Speech Recognition) is one of the tasks in MMS.

Similarly to MMS TTS (Text to Speech) for Shan language, some consonants are missing by the original datasets. [Finetune Vits TTS MMS for Shan language](note/vits_tts_mms_shan_finetune)

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

## Notebook environment

```python
%%capture
!pip install pandas
!pip install --upgrade pip 
!pip install datasets[audio]
!pip install evaluate
!pip install git+https://github.com/huggingface/transformers.git
!pip install jiwer
!pip install accelerate
```

## Datasets

### Prepair datasets

From [previous TTS finetune datasets](note/vits_tts_mms_shan_finetune) we should split train and validate dataset.

```python
import os
import pandas as pd
import shutil
import random

datasets_path = <dataset-path>
metadata_path = f'{datasets_path}/metadata.csv'
metadata = pd.read_csv(metadata_path)

test_dir = f'{datasets_path}/audio-data/test'
train_dir = f'{datasets_path}/audio-data/train'

if not os.path.exists(test_dir):
    os.makedirs(test_dir)


all_files = metadata['file_name'].tolist()

num_files_to_move = int(0.2 * len(all_files))
files_to_move = random.sample(all_files, num_files_to_move)

for file in files_to_move:
    file_name = os.path.basename(file)
    shutil.move(os.path.join(train_dir, file_name), os.path.join(test_dir, file_name))

    metadata.loc[metadata['file_name'] == file, 'file_name'] = file.replace('train', 'test')

update_metadata_path = f'{datasets_path}/metadata.csv'

metadata.to_csv(update_metadata_path, index=False)
```

Incase you want to push to hugging face hub or save datasets format to disk.

```python
from huggingface_hub import login

login()
```

```python
from datasets import load_dataset, Audio

dataset = load_dataset("audiofolder", data_dir="<saved_datasets_path")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

dataset.save_to_disk("./mms-asr-nova-datasets") # save to disk
dataset.push_to_hub("NorHsangPha/Shan-ASR-Nova", private=True) # push to hub
```

### Load datasets

Load from hub

```python
from datasets import load_dataset

datasets_id = <your_datasets_id>

shn_dataset_train = load_dataset(datasets_id, split="train", token=True)
shn_dataset_test = load_dataset(datasets_id, split="test", token=True)
```

Load from disk

```python
from datasets import load_from_disk

datasets_local_path = <your_datasets_path>

shn_dataset_train = load_from_disk(datasets_local_path)
shn_dataset_test = load_from_disk(datasets_local_path)

print(shn_dataset_train)
print(shn_dataset_test)

# Dataset({
#     features: ['audio', 'transcription'],
#     num_rows: 422
# })
# Dataset({
#     features: ['audio', 'transcription'],
#     num_rows: 105
# })
```

### Clean datasets

Random display some dataset example

```python
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))

show_random_elements(shn_dataset_train.remove_columns(["audio"]), num_examples=10)

# transcription
# 0 ၶဝ်ၵေႃႈ ဢွၼ်ၵၼ်ဢွၵ်ႇသုမ်ႉမႃးတူၺ်းသေ လႆႈႁၼ်ၽူႈမၢႆၼၼ်ႉ ၼွၼ်းလူမ်ႉတၢႆၵႂႃႇယဝ်ႉ
# 1 ၶႃႈတေသိုပ်ႇဢဝ်ပုၼ်ႈၽွၼ်းၵႂႃႇၵူၺ်း လူဝ်ႇႁဵတ်းသင်ၵေႃႈဢုပ်ႇၵၼ်တင်းဝၼ်းၸိုၼ်ႈၵႂႃႇလႄႈၼႃႈ
# 2 သမ်ႉဝႃႈလွင်ႈမၼ်းပူၼ်ႉမႃးႁိုင်တၢၼ်ႇႁိုဝ်ၵေႃႈ လွင်ႈဝၼ်းလင်ၼၼ်ႉ ယင်းတိုၵ်ႉဢဝ်မႃးလၢတ်ႈထိုင်ၵၼ်လႆႈယူႇ
# 3 ဝၼ်းၸိုၼ်ႈယၵ်ႉတူၺ်းၼႃႈၸၢႆးလိူၼ်ယဝ်ႉ ထူၺ်ႈၸႂ်ဢွၵ်ႇမႃးၼင်ႇလူမ်ၸႂ်
# 4 ၵူၼ်းၼႆႉ မၼ်းတေမီးၽႂ်ဢၼ်ႁၼ်ငိုၼ်းယဝ်ႉ တေထဵင်လႆႈ ၼႂ်းၼၼ်ႉၵေႃႈယင်းပႃးၵူၼ်းၸိူင်ႉၼင်ႇၸဝ်ႈမိူင်းၶမ်းယူႇလူး
```

To remove some special character/symbols.

```python
import re
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\။\၊\…]'

def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()
    return batch

shn_dataset_train = shn_dataset_train.map(remove_special_characters)
shn_dataset_test = shn_dataset_test.map(remove_special_characters)
```

## Tokenizer and Vocab extract

```python
def extract_all_chars(batch):
    all_text = " ".join(batch["transcription"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = shn_dataset_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=shn_dataset_train.column_names)
vocab_test = shn_dataset_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=shn_dataset_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

# replace | with empty space
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

# Add UNK and PAD token
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
```

### Wav2Vec2CTCTokenizer

```python
target_lang = "shn"

from transformers import Wav2Vec2CTCTokenizer

mms_adapter_repo = "facebook/mms-1b-all"

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(mms_adapter_repo)
new_vocab = tokenizer.vocab

new_vocab[target_lang] = vocab_dict
```

Save vocab.json

```python
import json
with open('vocab.json', 'w', encoding='utf-8') as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False)
```

Use the json file to load the vocabulary into an instance of the Wav2Vec2CTCTokenizer class.

```python
from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
```

To save tokenizer or push to hub

```python
repo_name = "wav2vec2-mms-shn

tokenizer.push_to_hub(repo_name)
tokenizer.save_pretrained(repo_name)
```

## Feature Extractor

A Wav2Vec2FeatureExtractor object requires the following parameters to be instantiated:

```python
from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
```

Wrapped the feature extractor and tokenizer into Wav2Vec2Processor

```python
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

### Casting dataset to 16kHz format

Check dataset

```python
shn_dataset_train[0]["audio"]

# {'path': 'audio10.wav',
#  'array': array([-1.06073006e-09,  1.29704758e-09, -1.43051804e-09, ...,
#          7.50327745e-05,  6.82075042e-05,  0.00000000e+00]),
#  'sampling_rate': 22050}
```

```python
from datasets import Audio

shn_dataset_train = shn_dataset_train.cast_column("audio", Audio(sampling_rate=16_000))
shn_dataset_test = shn_dataset_test.cast_column("audio", Audio(sampling_rate=16_000))

shn_dataset_train[0]["audio"]

# {'path': 'audio10.wav',
#  'array': array([ 1.44173100e-08, -1.53559085e-08,  1.59961928e-08, ...,
#          3.89381676e-05,  7.58816605e-05,  0.00000000e+00]),
#  'sampling_rate': 16000}

rand_int = random.randint(0, len(shn_dataset_train)-1)

print("Target text:", shn_dataset_train[rand_int]["transcription"])
print("Input array shape:", shn_dataset_train[rand_int]["audio"]["array"].shape)
print("Sampling rate:", shn_dataset_train[rand_int]["audio"]["sampling_rate"])

# Target text: ၵူၺ်းၵႃႈ သူဝူၼ်ႉဝႃႈ ပိူၼ်ႈတေပွႆႇသူလွတ်ႈလႆႈယူႇႁႃႉ
# Input array shape: (73392,)
# Sampling rate: 16000
```

### Preprocess datasets for training

```python
def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    batch["labels"] = processor(text=batch["transcription"]).input_ids
    return batch

shn_dataset_train = shn_dataset_train.map(prepare_dataset, remove_columns=shn_dataset_train.column_names)
shn_dataset_test = shn_dataset_test.map(prepare_dataset, remove_columns=shn_dataset_test.column_names)
```

## Training

### Setup trainer

```python
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
```

### Evaluate function

```python
from evaluate import load
import numpy as np

wer_metric = load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
```

### Load model checkpoint

```python
from transformers import Wav2Vec2ForCTC

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/mms-1b-all",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,
)
```

### Re-initialize all the adapter weights to make sure that only the adapter weights will be trained and that the rest of the model stays frozen

```python
model.init_adapter_layers()
model.freeze_base_model()

adapter_weights = model._get_adapters()
for param in adapter_weights.values():
    param.requires_grad = True
```

### Define all parameters related to training

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir=repo_name,
  group_by_length=True,
  per_device_train_batch_size=32, # decrease this number for less GPU VRAM
  evaluation_strategy="steps",
  num_train_epochs=10,
  gradient_checkpointing=True,
  fp16=True,
  save_steps=200,
  eval_steps=100,
  logging_steps=5,
  learning_rate=1e-3,
  warmup_steps=100,
  save_total_limit=2,
  push_to_hub=False,
)
```

Passed instances to Trainer and Train!

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)

trainer.train()

# {'loss': 10.5366, 'grad_norm': 22.734519958496094, 'learning_rate': 3e-05, 'epoch': 0.36}
# {'loss': 10.3854, 'grad_norm': 25.92425537109375, 'learning_rate': 8e-05, 'epoch': 0.71}
# {'loss': 9.2912, 'grad_norm': 26.86384391784668, 'learning_rate': 0.00013000000000000002, 'epoch': 1.07}
# ...
# {'loss': 0.2802, 'grad_norm': 0.41764652729034424, 'learning_rate': 0.00093, 'epoch': 6.79}
# {'loss': 0.2918, 'grad_norm': 0.35124266147613525, 'learning_rate': 0.00098, 'epoch': 7.14}
```

## Save model or push to hub

```python
from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
import os

adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
adapter_file = os.path.join(training_args.output_dir, adapter_file)

safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})

trainer.save_model(<model_name>)
trainer.push_to_hub()
```

## Inference

Load both finetune model and origin model to compare

```python
from transformers import Wav2Vec2ForCTC, AutoProcessor

finetune_model_id = "<saved_finetuned_model>"
original_model_id = "facebook/mms-1b-all"

finetune_model = Wav2Vec2ForCTC.from_pretrained(finetune_model_id, target_lang="shn", ignore_mismatched_sizes=True).to("cuda")
finetune_processor = AutoProcessor.from_pretrained(finetune_model_id)

original_model = Wav2Vec2ForCTC.from_pretrained(original_model_id).to("cuda")
original_processor = AutoProcessor.from_pretrained(original_model_id)
original_processor.tokenizer.set_target_lang("shn")
original_model.load_adapter("shn")
```

### Load test dataset or use audio file

```python
from datasets import Audio, load_dataset

test_data = load_dataset("<dataset_id>", split="test", token=True)
# or test_data = load_from_disk("dataset_path/train")
test_data = test_data.cast_column("audio", Audio(sampling_rate=16_000))

print(test_data[20]["audio"])
print(test_data[20]["transcription"])

# {'path': 'audio35.wav', 'array': array([-0.0038032 , -0.00612239, -0.00519261, ..., -0.00181367,
#        -0.00168416, -0.00077092]), 'sampling_rate': 16000}
# သိူဝ်ႈႁတ်ႉၶႅၼ်ပွတ်းသီၶဵဝ်လိူင် ထႅင်ႈပၼ်တၢင်းႁၢင်ႈလီၸွမ်းၽိဝ်ၼိူဝ်ႉၼင်ၶၢဝ်လိူင်ဢွၼ်ႇမၼ်းၼၢင်းၼၼ်ႉထူၼ်ႈ

selected_data_sample = test_data[20]
audio_samples = selected_data_sample["audio"]["array"]
```

Or use audio file

```python
import librosa
import pandas as pd

ASR_SAMPLING_RATE = 16_000
sample_num = 301 # sample number
audio_fp = f"{dataset_path}/audio-data/train/audio{sample_num}.wav"
df = pd.read_csv(f"{dataset_path}/metadata.csv")

audio_samples = librosa.load(audio_fp, sr=ASR_SAMPLING_RATE, mono=True)[0]
```

### Transcribe

```python
import torch


finetune_input_dict = finetune_processor(audio_samples, sampling_rate=16_000, return_tensors="pt")
finetune_logits = finetune_model(finetune_input_dict.input_values.to("cuda")).logits
finetune_pred_ids = torch.argmax(finetune_logits, dim=-1)[0]

original_input_dict = original_processor(audio_samples, sampling_rate=16_000, return_tensors="pt")
original_logits = original_model(original_input_dict.input_values.to("cuda")).logits
original_pred_ids = torch.argmax(original_logits, dim=-1)[0]
```

Display result

```python
from IPython.display import Markdown, display

def printmd(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))


printmd("\nFinetune model Prediction:", color="orange")
print(finetune_processor.decode(finetune_pred_ids))

printmd("\nOriginal model Prediction:", color="yellow")
print(original_processor.decode(original_pred_ids))

printmd("\nReference:", color="green")
# print(selected_data_sample["transcription"])
print(df["transcription"][sample_num-1])

# Finetune model Prediction:
# ၼႂ်းၶၢ်းတၢင်းသိပ်းပီဢၼ်ဢမ်ႇလႆႈႁူပ်ႉၺႃးၵၼ်ၼၼ်ႉၵေႃႈ မၼ်းပဵၼ်လွင်ႈတၢင်ႈၸႂ်ႁဝ်းမိူၼ်ၵၼ်
# Original model Prediction:
# ၼႂ်ႉ ၶႃး တၢင်း သိပ်ႉ ပီ ဢၼ် ဢမ်ႇ လႆႈ ႁွပ်ႉၺႃး ၵၼ် ၼၼ်ႉ ၵေႃႈ မၼ်း ပိူၼ်ႇ လွင်ႈ တၢင်ႈၸႂ် ႁဝ်း မိူၼ်ႇ ၵၼ် -
# Reference:
# ၼႂ်းၶၢဝ်းတၢင်းသိပ်းပီ ဢၼ်ဢမ်ႇလႆႈႁူပ်ႉၺႃးၵၼ်လႆႈၼၼ်ႉၵေႃႈ မၼ်းပဵၼ်လွင်ႈတင်ႈၸႂ်ႁဝ်းမိူၼ်ၵၼ်
```

## Conclusion

Funetune model Prediction:
ၼႂ်းၶၢ်းတၢင်းသိပ်းပီဢၼ်ဢမ်ႇလႆႈႁူပ်ႉၺႃးၵၼ်ၼၼ်ႉၵေႃႈ မၼ်းပဵၼ်လွင်ႈတၢင်ႈၸႂ်ႁဝ်းမိူၼ်ၵၼ်

Original model Prediction:
ၼႂ်ႉ ၶႃး တၢင်း သိပ်ႉ ပီ ဢၼ် ဢမ်ႇ လႆႈ ႁွပ်ႉၺႃး ၵၼ် ၼၼ်ႉ ၵေႃႈ မၼ်း ပိူၼ်ႇ လွင်ႈ တၢင်ႈၸႂ် ႁဝ်း မိူၼ်ႇ ၵၼ် -

Reference:
ၼႂ်းၶၢဝ်းတၢင်းသိပ်းပီ ဢၼ်ဢမ်ႇလႆႈႁူပ်ႉၺႃးၵၼ်လႆႈၼၼ်ႉၵေႃႈ မၼ်းပဵၼ်လွင်ႈတင်ႈၸႂ်ႁဝ်းမိူၼ်ၵၼ်

Due to my dataset is about 30 min of audio length (400~ sample file), it's not perfect yet, but you will see some little improvement.
