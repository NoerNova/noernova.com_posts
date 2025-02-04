---
tags: ["gpt-2" ,"shan" ,"LLM" ,"generative-ai"]
date: Jan 12, 2024
title: Fine-Tuning GPT-2 Large Language Model တွၼ်ႈတႃႇလိၵ်ႈတႆး
subtitle: LLM generative model for Shan language
image: https://i.pinimg.com/564x/6f/2b/6b/6f2b6b2b733e936abde09cd24a9765d1.jpg
link: blog/fine-tuning-gpt2-for-shan-language
description: တႃႇတေႁဵတ်းႁႂ်ႈ gpt2 generate ဢွၵ်ႇပၼ်လိၵ်ႈတႆးလႆႈၼၼ်ႉ မီးၶၵ်ႉတွၼ်ႈၸိူင်ႉႁိုဝ်လၢႆလၢႆ?။
---

## Contents

- [Introduction](#introduction)
- [ႁဵတ်းသင်လႄႈလႆႈ Fine-tune ဢဝ် model ပိူၼ်ႈ၊ သင်ဢမ်ႇ train တႄႇငဝ်ႈတေႃႇပၢႆဢဝ်ႁင်းၵူၺ်း?](#ႁဵတ်းသင်လႄႈလႆႈ-fine-tune-ဢဝ်-model-ပိူၼ်ႈ-သင်ဢမ်ႇ-train-တႄႇငဝ်ႈတေႃႇပၢႆဢဝ်ႁင်းၵူၺ်း)
- [Prerequisites](#prerequisites)
- [Step-by-Step](#step-by-step)
  - [Step 0. Mount google drive and login to huggingface hub](#step-0-mount-google-drive-and-login-to-huggingface-hub)
  - [Step 1.  ၵဵပ်းႁွမ်ၶေႃႈမုၼ် Corpus လိၵ်ႈတႆး လႄႈ သုၵ်ႈလၢင်ႉၶေႃႈမုၼ်း Data Cleaning](#step-1--ၵဵပ်းႁွမ်ၶေႃႈမုၼ်-corpus-လိၵ်ႈတႆး-လႄႈ-သုၵ်ႈလၢင်ႉၶေႃႈမုၼ်း-data-cleaning)
  - [Step 2. Tokenization](#step-2-tokenization)
  - [Step 3. Data Preprocessing](#step-3-data-preprocessing)
  - [Step 4. Model Setup and Optimizer](#step-4-model-setup-and-optimizer)
  - [Step 5: Fine-tuning](#step-5-fine-tuning)
  - [Model Test (Top-P sampling)](#model-test-top-p-sampling)
- [Conclusion](#conclusion)
  - [Link](#link)
  - [Source Code](#source-code)
  - [Dataset](#dataset)

## Introduction

GPT-2 Large Language Model ပဵၼ် Generative AI ဢၼ်ၶူင်သၢင်ႈလႄႈပိုၼ်ၽႄႈလူၺ်ႈ OpenAI။ GPT-2 ႁဵတ်းႁႂ်ႈမီးလွင်ႈလႅၵ်ႈလၢႆႈယႂ်ႇလူင်ၼႂ်းၶၵ်ႉၵၢၼ် NLP (Natural Language Language) မၼ်းၸၢင်ႈ generate ၶူင်ဢွၵ်ႇၶေႃႈၵႂၢမ်းဢၼ်မိူၼ်ၵူၼ်းၸႂ်ႉတိုဝ်း၊ ပိၼ်ႇၽႃႇသႃႇ၊ ႁုပ်ႈထွႆႈၵႂၢမ်း လႄႈထႅင်ႈတင်းၼမ်။

> GPT-2 (Generative Pre-trained Transformer 2) ဢၼ်ပိုၼ်ၽႄႈမႃးၼၼ်ႉပဵၼ် pre-trained model ဢၼ်ၸႂ်ႉၶေႃႈမုၼ်း train 40GB (a very large corpus) 8 million web pages.

ၵူၺ်းၵႃႈ ၼႂ်းၶေႃႈမုၼ်းဢၼ်ၸႂ်ႉ train ၸိူဝ်းၼၼ်ႉပဵၼ်ၶေႃႈမုၼ်းၽႃႇသႃႇ english လၢႆလၢႆ ၼႂ်းပွင်ႈၵႂၢမ်းႁူဝ်ၼႆႉၸိုင်တေမႃးဢွၼ်ၸၢမ်းတူၺ်းလွၵ်းလၢႆးၸႂ်ႉၶေႃႈမုၼ်းၽႃႇသႃႇတႆးသေ fine-tune gpt-2 ႁႂ်ႈမၼ်း generate ၽႃႇသႃႇတႆးလႆႈ။

***ယိူင်းမၢႆထီႉ 1 ၶွင် GPT-2 တႃႇႁႂ်ႈမၼ်း generate text ဢွၵ်ႇမႃး ဢိင်ၼိူဝ်ၶေႃႈၵႂၢမ်း prompt ဢၼ်ပၼ် input ၶဝ်ႈ။***

တႃႇတေႁဵတ်းႁႂ်ႈ Computer ဢမ်ႇၼၼ် AI ပွင်ႇၸႂ် context လိၵ်ႈသေ ၶိုၼ်းၶူင်တႅမ်လိၵ်ႈသိုပ်ႈတေႃႇၵၼ်ၵႂႃႇၼၼ်ႉ မိူဝ်ႈၵွၼ်ႇယၢမ်းၸႂ်ႉတိုဝ်းလွၵ်းလၢႆးတၢင်းၼမ် **သိုပ်ႇလူ - [AI/Computer ႁဵတ်းၸိူင်ႉႁိုဝ်သေၸင်ႇပွင်ႇၵႂၢမ်းၵူၼ်းလႆႈ](https://www.noernova.com/blog/markov-chain-language-model)**

probabilistic/statistic model လွၵ်းလၢႆးၵဝ်ႇၸိူဝ်းၼၼ်ႉမီးပၼ်ႁႃမႃး ပေႃးဝႃႈၶေႃႈၵႂၢမ်းထႅဝ်လိၵ်ႈမၼ်းယၢဝ်းမႃးတိၵ်းတိၵ်း မၼ်းၸၢင်ႈလိုမ်းၼႃႈလိုမ်းလင် လႄႈၸႂ်ႉ resource တွၼ်ႈတႃႇတေတွင်း context ၶေႃႈၵႂၢမ်းတင်းၼမ်။

Transformer technique ၸင်ႇၵိူတ်ႇပဵၼ်မႃး တႃႇၵႄႈပၼ်ႁႃဢၼ်ထူပ်းယူႇၸိူဝ်းၼၼ်ႉ **သိုပ်ႇလူ - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)**

GPT-2 ၵေႃႈပဵၼ် transformer-base။

> GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text. The diversity of the dataset causes this simple goal to contain naturally occurring demonstrations of many tasks across diverse domains. GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than 10X the amount of data.

## ႁဵတ်းသင်လႄႈလႆႈ Fine-tune ဢဝ် model ပိူၼ်ႈ၊ သင်ဢမ်ႇ train တႄႇငဝ်ႈတေႃႇပၢႆဢဝ်ႁင်းၵူၺ်း?

GPT-2, GPT-3 Large Language model ၸိူဝ်းၼႆႉ တေလႆႈၸႂ်ႉႁႅင်း Computer/GPU/TPU ၼမ်ႉတႄႉတႄႉဝႃႈဝႃႈသေၸင်ႇၵွႆႈ train ပဵၼ် AI model ဢွၵ်ႇမႃးလႆႈ။

![gpt-2 training time {caption: gpt-2 training time}](/assets/fine-tuning-gpt2-for-shan-language/Screenshot-2567-01-12-at-03.24.18.png)

GPT-2 ၸႂ်ႉၶၢဝ်းယၢမ်း GPU 100,000 ၸူဝ်ႈမွင်း ပွင်ႇဝႃႈပေႃးၸႂ်ႉ GPU 1 ႁူၺ်ႇသေ train တေလႆႈၸႂ်ႉၶၢဝ်းယၢမ်း 11 ပီ။

![gpt training cost {caption: gpt training cost}](/assets/fine-tuning-gpt2-for-shan-language/Screenshot-2567-01-12-at-03.25.08.png)

ၵႃႈၶၼ်ငိုၼ်းတႃႇၸႂ်ႉၼႂ်းၵၢၼ် train gpt-2 **40,000US dollar** လႄႈ GPT-3 ၶိုၼ်ႈၵႂႃႇပဵၼ် **4.6 လၢၼ်ႉ US dollar**။

ၵွပ်ႈၼၼ်လႄႈ ပေႃးဝႃႈဢမ်ႇၸႂ်ႈၶွမ်ႊပၼီႊယႂ်ႇ"လူင်" တႄႉတိုၼ်းဝႃႈဢမ်ႇပဵၼ်လႆႈၼၼ်ႉယဝ်ႉ။
ၵူၺ်းၵႃႈၼႂ်းဝၼ်းမိူဝ်ႈလဵဝ်ၼႆႉ LLM pre-trained model ဢွၵ်ႇမႃးဝႆႉတင်းၼမ် ဢၼ်ႁဝ်းၶိုၼ်းမႃး fine-tune လူၺ်ႈၵၢၼ်သႂ်ႇၶေႃႈမုၼ်းလိၵ်ႈလၢႆးႁဝ်း ႁႂ်ႈၸႂ်ႉၸွမ်း model မႂ်ႇ"ၸိူဝ်းၼႆႉလႆႈလႄႈ ဢၼ်ၵႃႈႁဝ်းႁဵတ်းလႆႈၵေႃႈ ပဵၼ်ၵၢၼ်ႁႃပၼ်ၶေႃႈမုၼ်းလိၵ်ႈၼမ်ၼမ်သေ fine-tune ၵႂႃႇၵေႃႈၸႂ်ႉလႆႈလီငၢမ်းယူႇၶႃႈ။

## Prerequisites

- Python 3
- [Hugging Face (Account & Token)](https://huggingface.co/join)
- Transformers library ('transformers')
- PyTorch
- GPU (CUDA Core)/CPU at leased 20GB of RAM

တႃႇတေ fine-tuning LLM model ၸိူဝ်းၼႆႉၸႂ်ႉႁႅင်း processing တင်းၼမ် ပေႃးဝႃႈၸႂ်ႉ CPU လၢႆလၢႆတေၸႂ်ႉၶၢမ်းယၢမ်းတင်းႁိုင် ပေႃးဝႃႈၸႂ်ႉ CPU 10HR ၼႆ ၼိူဝ် GPU ၸၢင်ႈလူတ်းလိူဝ် 30min - 1HR။

တီႈၶႃႈတႄႉဢမ်ႇမီးဝႆႉ GPU လႄႈ တေၸႂ်ႉ [Google Colab](https://colab.research.google.com/) တူဝ် free မီး VRAM (GPU ram) ဢမ်ႇပဵင်းပေႃးလႄႈ တေလႆႈၸႂ်ႉ Colab Pro (A100 Nvidia)

![google colab pro {caption: google colab pro pricing}](/assets/fine-tuning-gpt2-for-shan-language/Screenshot-2567-01-10-at-11.08.27.png)

***သိုဝ်ႉ Pro သေတႃႉၵေႃႈ လႆႈၸႂ်ႉ A100 ဢမ်ႇပူၼ်ႉသေ သၢမ်ပွၵ်ႈ lol***

## Step-by-Step

### Step 0. Mount google drive and login to huggingface hub

ႁဝ်းတေ mount google drive တွၼ်ႈတႃႇၵဵပ်းသိမ်း data ဝႆႉ ၵွပ်ႈဝႃႈ google colab ၼႆႉပေႃး kill process ၵႂႃႇၼႆ ၶေႃႈမုၼ်းမၼ်းတေႁၢႆၵႂႃႇတင်းမူတ်း။

လႄႈ login hugging face ဝႆႉတွၼ်ႈတႃႇၵဵပ်းသိမ်း model ဝႆႉၼိူဝ် hugging face။

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
from huggingface_hub import notebook_login
notebook_login()
```

Install dependency

```python
# requirements GPU

!pip install accelerate -U
!pip install transformers[torch]
!pip install datasets
```

Optional for TPU ပေႃးဝႃႈၸႂ်ႉ TPU

```python
# requirement TPU

!pip install google-api-python-client>=1.12.5
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.
!pip install --upgrade torch torch-xla
!pip install --upgrade transformers
```

Checking available CUDA

```python
!nvidia-smi
```

### Step 1.  ၵဵပ်းႁွမ်ၶေႃႈမုၼ် Corpus လိၵ်ႈတႆး လႄႈ သုၵ်ႈလၢင်ႉၶေႃႈမုၼ်း Data Cleaning

GPT-2 model တေၵတ်ႉၶႅၼ်ႇ ၶိုၵ်ႉၶႅမ်ႉၵႃႈႁိုဝ်ၼၼ်ႉ ဢိင်ၼိူဝ်ၶေႃႈမုၼ်းမီးၼမ်ၵႃႈႁိုဝ်ၼၼ်ႉယဝ်ႉၶႃႈ တူဝ် pre-trained ၶဝ်ၸႂ်ႉဝႆႉ text data 40GB (ၶေႃႈမုၼ်း website 8 လၢၼ်ႉ ၼႃႈ) ၼႆလႄႈ တွၼ်ႈတႃႇတေႁႂ်ႇမၼ်းႁူႉၸၵ်း လႄႈ generate လိၵ်ႈတႆးလႆႈၼၼ်ႉ ႁဝ်းၵေႃႈလူဝ်ႇပွၼ်ႈၶေႃႈမုၼ်းပၼ်မၼ်းႁႂ်ႈၼမ်တီႈသုတ်းတီႈႁႃလႆႈ။

ႁဝ်းတေ focus ၼိူဝ် generative field ၼႆလႄႈ Text data ဢၼ်လူဝ်ႇၼၼ်ႉ ပဵၼ် text မဵဝ်းႁိုဝ်ၵေႃႈလႆႈ တႅမ်ႈလွင်ႈသင်ၵေႃႈလႆႈ ၵူၺ်းၵႃႈ တီႈဢေႇသုတ်း တူဝ်ၽိတ်းတူဝ်ထုၵ်ႇၸိူဝ်းၼၼ်ႉ လူဝ်ႇလႆႈၶျဵၵ်ႉသေမႄးပၼ်ႁႂ်ႈထုၵ်ႇမႅၼ်ႈၸွမ်းပိူင်လိၵ်ႈလၢႆး။

ၵွပ်ႈၼၼ်လႄႈၶႃႈလိူၵ်ႈၸႂ်ႉၶေႃႈမုၼ်းၼႂ်းဝဵပ်ႉသၢႆႉဢၼ်ယုမ်ႇလႆႈဝႃႈ တေမီးတူဝ်ၽိတ်းဢေႇတီႈသုတ်းမိူၼ်ၼင်ႇဝဵပ်ႉသၢႆႉၸိူဝ်းၼႆႉ

- [shannews.org 66MB](https://shannews.org/) (ၸုမ်းၶၢဝ်ႇၽူႈတွႆႇႁွၵ်ႉ)
- [taifreedom.com 46MB](https://taifreedom.com/) (ၸုမ်းၶၢဝ်ႇတႆးလွတ်ႈလႅဝ်း)
- [shn.wikipedia.com 11MB](https://shn.wikipedia.com/) (ဝီႇၵီႇၽီတီယႃးတႆး)
- [ssppssa.org 2.8MB](https://ssppssa.org/) (ပႃႇတီႇမႂ်ႇသုင်ၸိုင်ႈတႆး)
- [shan.shanhumanrights.org 2.6MB](https://shan.shanhumanrights.org/) (ၵဝ်ႉငဝ်ႈသုၼ်ႇလႆႈၵူၼ်း)

***ပေႃးဝႃႈမီးဝဵပ်ႉသၢႆႉဢၼ်ၸႂ်ႉတိုဝ်းလိၵ်ႈတႆး ၼွၵ်ႈလိူဝ်ၼႆႉထႅင်ႈၸိုင် ၸွႆႈသူင်ႇပၼ်ၽွင်ႈၶႃႈ***

ၶေႃႈမုၼ်းဢၼ်လႆႈမႃးၸိူဝ်းၼႆႉပဵၼ်ၶေႃႈမုၼ်းဢၼ်ပိုၼ်ၽႄႈ api ဝႆႉတင်းမူတ်း ဝၢႆးသေ ၸၼ်ၶေႃႈမုၼ်း (.csv) ဝႆႉယဝ်ႉ လူဝ်ႇႁဵတ်း data cleaning ဢဝ်ဢွၵ်ႇပႅတ်ႈတူဝ်ဢၼ်ဢမ်ႇၸႂ်ႈတူဝ်လိၵ်ႈ မိူၼ်ၼင်ႇ တူဝ် "\n" "\t" လႄႈ HTML tags ၸိူဝ်းၼႆႉ။

![data cleaning {caption: ၶၵ်ႉတွၼ်ႈၵၢၼ်သုၵ်ႈလၢင်ႉၶေႃႈမုၼ်း}](/assets/fine-tuning-gpt2-for-shan-language/Screenshot-2567-01-10-at-14.09.23.png)

***clean .csv data***

```python
import pandas as pd
from bs4 import BeautifulSoup
import html

# read data file
file = "./shan_data.csv"
df = pd.read_csv(file)

content = df.content

# fill NaN content field with title
df.loc[df['content'].isna(), 'content'] = df.loc[df['content'].isna(), 'title']

# remove all html tags
def remove_html_tags(contents):
    html_clean = html.unescape(contents)
    soup = BeautifulSoup(html_clean, "html.parser")
    soup = soup.get_text().replace("\n", " ")
    return soup

content = content.apply(remove_html_tags)
df['content'] = content

# remove empty space
df['content'] = df['content'].str.strip()

df['content'] = df.apply(lambda row: row['title'] if len(row['content']) == 0 else row['content'], axis=1)

# save to csv
df.to_csv(f"cleaned_data/{file}_cleaned.csv", index=False)

print("Finished.")
```

ဝၢႆးသေ cleaned ၶေႃႈမုၼ်းယဝ်ႉ တေလႆႈႁဵတ်းၶေႃႈမုၼ်းပဵၼ်သွင်တွၼ်ႈ 1. တွၼ်ႈတႃႇ train လႄႈ 2. တွၼ်ႈတႃႇ validate လွင်ႈမၢၼ်ႇမႅၼ်ႈမၼ်း။

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./raw_data/shannews.org.csv", encoding='utf-8')
contents = df['content']

def save_combined_text_to_file(data, output_file):
    combined_text = ""

    for content in data:
        combined_text += content + "<|endoftext|>" + "\n"

    with open(output_file, "w", encoding="utf-8") as output:
        output.write(combined_text)
    print("Combined text saved to: ", output_file)

train_text, valid_text = train_test_split(contents, test_size=0.2, random_state=42)

train_file = "./txt_data/shannews/train_data.txt"
valid_file = "./txt_data/shannews/valid_data.txt"

save_combined_text_to_file(train_text, train_file)
save_combined_text_to_file(valid_text, valid_file)

print("Training data saved to: ", train_file)
print("Validation data saved to: ", valid_file)
```

### Step 2. Tokenization

Tokenization ပဵၼ်ၶၵ်ႉတွၼ်ႈလမ်ႇလွင်ႈတွၼ်ႈတႃႇၵၢၼ် NLP ၵူႈဢၼ်ဢၼ် tokenization ပဵၼ်ၵၢၼ်တတ်းထႅဝ်လိၵ်ႈဢွၵ်ႇပဵၼ်ၶေႃႈ” ဢၼ်မီးတီႈပွင်ႇၵႂၢမ်းမၢႆ tokenizer မီးဝႆႉလၢႆမဵဝ်း မိူၼ်ၼင်ႇ Dictionary base tokenization (တတ်းၶေႃႈၵႂၢမ်းဢဝ်ၼႂ်း dictionary [တူၺ်း -> ShanNLP](https://github.com/NoerNova/ShanNLP))၊ Byte-Pair Encoding (BPE tokenization)၊ WordPiece tokenization၊ Unigram tokenization၊ Sentencepiece tokenizer ၸဵမ်ၸိူဝ်းၼႆႉ။
ၼႂ်း GPT-2 တေၸႂ်ႉလွၵ်းလၢႆး BPE tokenization။

တႃႇႁႂ်ႈၸႂ်ႉ tokenizer ႁဝ်းလႆႈၸွမ်း GPT-2 ႁဝ်းတေ train tokenizer လူၺ်ႈ GPT2Tokenizer လႄႈ extended gpt-2 tokenizer ၸွမ်းၼင်ႇ Code ၼႆႉ။

```python
from tokenizers import (decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer)
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

new_tokenizer = Tokenizer(models.BPE())

new_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])

train_file = '/content/drive/MyDrive/gpt2labs/shannews/shannews_datasets.txt'

new_tokenizer.train([train_file], trainer=trainer)
new_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
new_tokenizer.decoder = decoders.ByteLevel()
new_tokenizer = GPT2TokenizerFast(tokenizer_object=new_tokenizer)
new_tokenizer.save_pretrained("shannews_bpe_tokenizer")

new_tokenizer
# save tokenizer to huggingface hub
new_tokenizer.push_to_hub("shannews_bpe_tokenizer")

  
# gpt2 tokenizer
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

print(len(gpt2_tokenizer.get_vocab()))

gpt2_tokenizer

# extended tokenizer
vocab_tokens = list(new_tokenizer.get_vocab())
decoded_tokens = [new_tokenizer.decoder.decode([token]) for token in vocab_tokens]
print(len(vocab_tokens), len(decoded_tokens))

gpt2_tokenizer.add_tokens(decoded_tokens)
gpt2_tokenizer.save_pretrained("shannews_extened_tokenizer_gpt2")
print(len(gpt2_tokenizer.get_vocab()))

gpt2_tokenizer
# save extended tokenizer to huggingface hub
gpt2_tokenizer.push_to_hub("shannews_bpe_extened_tokenizer")
```

![tokenization {caption: BPE tokenization training}](/assets/fine-tuning-gpt2-for-shan-language/Screenshot-2567-01-12-at-02.53.04.png)

### Step 3. Data Preprocessing

ဝၢႆးသေမီး Tokenizer ယဝ်ႉ ၵမ်းၼႆႉတေၸတ်းၵၢၼ် test, train data ႁဝ်း၊ တတ်းၶေႃႈၵႂၢမ်းလူၺ်ႈ tokenizer လႄႈၸတ်းၶေႃႈၵႂၢမ်းပဵၼ်ၵွၼ်ႈ” ဢၼ်ႁွင်ႉဝႃႈ chunks တွၼ်ႈတႃႇပွၼ်ႈသွၼ်ပၼ် machine။

```python
import os
import torch
import time

from datasets import Dataset, DatasetDict
from tokenizers import (decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer)
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

  
# load the tokenizer from huggingface
tokenizer = GPT2TokenizerFast.from_pretrained("NorHsangPha/shannews_bpe_extened_tokenizer")
tokenizer.pad_token = tokenizer.eos_token

print(tokenizer.vocab_size)
print(len(tokenizer))

tokenizer.save_pretrained("/tokenizer/shannews_bpe_extened_tokenizer")

# fine-tuning
train_file = "/content/drive/MyDrive/gpt2labs/shannews/train_data.txt"
valid_file = "/content/drive/MyDrive/gpt2labs/shannews/valid_data.txt"

with open(train_file, 'r', encoding='utf-8') as f:
    train_data = f.readlines()

with open(valid_file, 'r', encoding='utf-8') as f:
    valid_data = f.readlines()

print(len(train_data), len(valid_data))


# dataset object
train_dataset = Dataset.from_dict({"text": train_data})
valid_dataset = Dataset.from_dict({"text": valid_data})

def preprocess_function(examples):
    out = tokenizer(examples["text"])
    return out

# apply tokenization to dataset
train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4, remove_columns=train_dataset.column_names)

valid_dataset = valid_dataset.map(preprocess_function, batched=True, num_proc=4, remove_columns=valid_dataset.column_names)

tokenized_datasets = DatasetDict({"train": train_dataset, "valid": valid_dataset})

  
# group tokenized datasets to blocks
block_size = 128

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in
    examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

  if total_length >= block_size:
      total_length = (total_length // block_size) * block_size
      result = {
        k: [t[i : i + block_size] for i in range(0, total_length,
          block_size)]
          for k, t in concatenated_examples.items()
        }
    result["labels"] = result["input_ids"].copy()
    return result

# group tokenized train dataset
lm_train_dataset = tokenized_datasets['train'].map(group_texts, batched=True, num_proc=4)

# group tokenized valid dataset
lm_valid_dataset = tokenized_datasets['valid'].map(group_texts, batched=True, num_proc=4)

lm_dataset = DatasetDict({"train": lm_train_dataset, "valid": lm_valid_dataset})
lm_dataset

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

out = data_collator([lm_dataset['train'][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")
```

### Step 4. Model Setup and Optimizer

တွၼ်ႈၼႆႉႁဝ်းတေ load gpt-2 model တွၼ်ႈတႃႇ fine-tune လႄႈ optimize ဢိတ်းဢွတ်း တွၼ်ႈတႃႇတေၸွႆႈႁႂ်ႈမၼ်း train ဝႆးၶိုၼ်ႈၵမ်ႈၽွင်ႈ (ပေႃးဝႃႈၽႂ်မီ GPU ၼမ်/ႁႅင်း ၼႆၶၢမ်ႈတွၼ်ႈၼႆႉၵေႃႈလႆႈ)

```python
from transformers import GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the GPT2 model
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Resize the model's embeddings
def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

new_embeddings_size = find_multiple(len(tokenizer), 64)
model.resize_token_embeddings(new_embeddings_size)

freeze_layers = False  

if freeze_layers:
    for name, param in model.named_parameters():
        if 'transformer.wte' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
```

### Step 5: Fine-tuning

ၵမ်းလိုၼ်းသုတ်းၵေႃႈတေႁဵတ်း Fine-tuning gpt-2 model လူၺ်ႈ transformer။
parameters ၸိူဝ်းၼႆႉ setup တႃႇ train ၼိူဝ် google colab pro A100 GPU 40GB of RAMS သင်ဝႃႈမီးႁႅင်း GPU လႄႈၶၢဝ်းယၢမ်းဢၼ်ၸႂ်ႉ GPU လႆႈႁိုင်ၼႆ ၸၢင်ႈလႅၵ်ႈ parameters မိူၼ်ၼင်ႇ batch_size ၶိုၼ်ႈထႅင်ႈလႆႈ 32, 64, 128 တေႁဵတ်းႁႂ်ႈ model ႁဵၼ်းႁူႉလႆႈလီၶိုၼ်ႈထႅင်ႈ ၵူၺ်းၵႃႈ ၵေႃႈတေၸႂ်ႉႁႅင်းလႄႈၶၢဝ်းယၢမ်း GPU ႁိုင်မႃးထႅင်ႈ။

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="/content/shannews_gpt2/",
    overwrite_output_dir=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=500,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=500,
    push_to_hub=True,
    save_total_limit=2
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_valid_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("shannews_gpt2")
```

### Model Test (Top-P sampling)

ဝၢႆးသေ train model ယဝ်ႉ ၸၢမ်းၸႂ်ႉ generate တူၺ်း။

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("NorHsangPha/shannews_bpe_extened_tokenizer")
model = AutoModelForCausalLM.from_pretrained("NorHsangPha/shannews_gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)

text = "တပ်ႉသိုၵ်းၸိုင်ႈတႆး"

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3

sample_outputs = model.generate(
    **model_inputs,
    max_new_tokens=40,
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3,
)

  
print("Output:\n" + 100 * '-')

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output,
    skip_special_tokens=True)))

# Output: ---------------------------------------------------------------------------------------------------- 
# 0: တပ်ႉသိုၵ်းၸိုင်ႈတႆး (ဝၼ်းပိူင်မၢႆမီႈမီး တူၵ်ႇတႅၵ်ႈ (မႄႈၸႄႈဝဵင်းၵျွၵ်ႉမႄးၼၼ်ႉ တေ 
# 1: တပ်ႉသိုၵ်းၸိုင်ႈတႆး တပ်ႉၵွင် ၸိုဝ်ႈ ၶၢဝ်းယၢမ်း TNLA ဢွၵ်ႇဝႆႉၼင်ႇၼႆ။ ၵူၼ်းဝၢၼ်ႈၵူတ 
# 2: တပ်ႉသိုၵ်းၸိုင်ႈတႆး RCSS/SSA လုၵ်ႈၸၢႆး ၵေးသီးလူင် ၸဵင်ႇတူၼ်ႈတၼ်းသႂ်ႇၶွင်ႊသီႊသႅၼ်ႊပ
```

## Conclusion

ဝၢႆးသေလဵပ်ႈႁဵၼ်းလႄႈၸၢမ်း trained, test model ယဝ်ႉၼၼ်ႉ တေႁၼ်ဝႃႈ မၼ်း generate ဢွၵ်ႇပၼ်လိၵ်ႈတႆးလႆႈပဵၼ်ၶေႃႈယူႇသေတႃႉ လွင်ႈတီႈပွင်ႇၵႂၢမ်းမၢႆထႅဝ်လိၵ်ႈ ယင်းပႆႇၵဵဝ်ႇၶွင်ႈၵၼ်လီ ဢၼ်ၼႆႉပဵၼ်လႆႈလၢႆတီႈ

1. ၶေႃႈမုၼ်းၸႂ်ႉ train ၼၼ်ႉၵမ်ႈၼမ်ပဵၼ်ၶေႃႈမုၼ်းၵဵဝ်ႇၵပ်းလွင်ႈၵၢၼ်သိုၵ်းသူဝ်၊ ၵၢၼ်လုၵ်ႉၽိုၼ်ႉ၊ ၶၢဝ်ႇငၢဝ်းငဝ်းလၢႆးဢမ်ႇလီၼႂ်းမိူင်းတႆး ၸိူဝ်းၼႆႉလၢႆလၢႆလႄႈ text ဢၼ် generate ဢွၵ်ႇမႃးၵေႃႈ ပဵၼ်ထွႆႈၵႂၢမ်းၸိူဝ်းၼၼ်ႉလၢႆလၢႆယဝ်ႉ။
2. တၢင်းၼမ်ၶေႃႈမုၼ်းဢၼ်ၸႂ်ႉ train တင်းမူၼ်းၼၼ်ႉ ဝၢႆးသေ cleaned ယဝ်ႉမီ 136.4MB ၵူၺ်း လႆႈဝႃႈတိုၵ်းမီးဢေႇတႄႉတႄႉ ပေႃးႁႂ်ႈလီလူဝ်ႇမီးပဵၼ် GB ၶိုၼ်ႈၼိူဝ်။
3. ႁႅင်း GPU ဢၼ်ၸႂ်ႉလႆႈၼၼ်ႉ မီးၶၢဝ်းယၢမ်းပၼ်ၸႂ်ႉဢေႇၼႃႇလႄႈ parameters ဢၼ်ၸႂ်ႉသွၼ် model လႆႈလူတ်းယွမ်းဝႆႉတင်းၼမ်ႉ ႁဵတ်းႁႂ်ႈ model ဢမ်ႇလႆႈႁဵၼ်းႁူဝ်ယွႆႈၶႅမ်ႉလီ

မိူဝ်းၼႃႈၶၢဝ်းယၢဝ်း တႃႇတေသိုပ်ႈႁဵတ်းႁႂ်ႈၸႂ်ႉ AI model မိူၼ်ၼင်ႇ LLM, Generative AI တွၼ်ႈတႃႇလိၵ်ႈတႆးလႆႈလီလီၼၼ်ႉယင်းလူဝ်ႇတၢင်းၸွႆႈထႅမ်ႁူမ်ႈမိုဝ်းၵၼ်လၢႆပႃႈလၢႆၾၢႆႇမိူၼ်ၼင်ႇ
***ႁႅင်းၶေႃႈမုၼ်းလိၵ်ႈတႆး*** ***ႁႅင်းငိုၼ်းတွင်းတႃႇၸႂ်ႉ*** train ၸိူဝ်းၼႆႉယူႇၶႃႈ။

### Link

- [shn-gpt2](https://huggingface.co/NorHsangPha/shn_gpt2)
- [Fine-tune a pretrained model - huggingface docs](https://huggingface.co/docs/transformers/v4.18.0/en/training)

#### Source Code

1. [Shan GPT-2 Fine-tuning Google Colab](https://colab.research.google.com/drive/1DPJp0WKY-MVWznJTIOWPgzFaUjxfa_zM?usp=sharing)
2. [BPE Tokenizer training](https://colab.research.google.com/drive/1of3hghxLQ2UtwFfRU6AAZKrGNJlOAu3D?usp=sharing)
3. [BPE extended tokenizer training](https://colab.research.google.com/drive/1yKgJLT0EP1WhjvHqA_9dMLPqg_ERzF6N)
4. [GPT-2 explore](https://colab.research.google.com/drive/1plAFbIh2XfGWMYDjBpfM1DOdeenHiBxV?usp=sharing)
5. [Shan tokenizer](https://colab.research.google.com/drive/1U5OuaF8sM72vZszGyMqOo55GDzGnR1Gp?usp=sharing)
6. [Shan Google Sentencepiece tokenizer](https://colab.research.google.com/drive/1UczcN4KUD0USL9iSLIombIF-eem0Ux61#scrollTo=XiX6zV4-H15X)

#### Dataset

- [Shan Data Collections](https://github.com/NoerNova/shan-data-collection)
