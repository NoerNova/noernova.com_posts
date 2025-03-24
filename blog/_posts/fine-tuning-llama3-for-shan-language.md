---
tags: ["llama" ,"shan" ,"LLM" ,"generative-ai", "finetune"]
date: July 25, 2024
title: Fine-Tuning Llama3 Large Language Model for Shan language - NLLB + Quantized + Ollama
subtitle: Instruct note to fine-tuning Llama3 for Shan language with NLLB translation datasets, Quantization and Ollama
image: https://i.pinimg.com/564x/9e/49/01/9e49015a36c3effbf97b3707ad944d9b.jpg
link: blog/fine-tuning-llama3-for-shan-language
description: ၸၢမ်းႁဵတ်းႁႂ်ႈ llama3 ပွင်ႇၸႂ်လိၵ်ႈတႆး။
---
## Contents

- [Introduction](#introduction)
- [Why Llama](#why-llama)
- [Low Resources Language](#low-resources-language)
- [Why and What is Fine-tuning](#why-and-what-is-fine-tuning)
- [Datasets and Fine-tune process](#datasets-and-fine-tune-processes)
- [Quantized + Ollama + Open WebUI](#quantized--ollama)
- [Conclution](#conclution)
- [Links](#links)

## Introduction

ပွင်ႈၵႂၢမ်းႁူဝ်ၼႆႉ ပဵၼ်မၢႆတႃႇတွင်း လွင်ႈလဵပ်ႈႁဵၼ်းတူၺ်း တႃႇတေႁဵတ်းႁႂ်ႈ LLM (Large Language Model) မိူၼ်ၼင်ႇ Llama3 ႁူႉပွင်ႇၸႂ် လႄႈတွပ်ႇၶေႃႈထၢမ်လိၵ်ႈတႆး ၵႂၢမ်းတႆးလႆႈ။

တွၼ်ႈတႃႇလိၵ်ႈတႆး ၵႂၢမ်းတႆး ယင်းတိုၵ်ႉမီးလွင်ႈၶဵင်ႇတႃႉ လႄႈလွင်ႈလူဝ်ႇလဵပ်ႈႁဵၼ်းထႅင်ႈတင်းၼမ် တႃႇတေၶဵၼ်ႇၽိုတ်ႉဢီး ႁဵတ်းႁႂ်ႈလိၵ်ႈတႆး ၵႂၢမ်းတႆးႁဝ်းၸႂ်ႉလႆႈလီၼႂ်းၵၢပ်ႈပၢၼ် AI ၼႆလႄႈ ၼႆႉပဵၼ်ပွင်ႈၵႂၢမ်းမၢႆတွင်း (Study Note) တႃႇလဵပ်ႈႁဵၼ်းတူၺ်းပွတ်းဢွၼ်ႇတွၼ်ႈၼိုင်ႈၵူၺ်းၶႃႈ။

## Why Llama

Llama (Large Language Model Meta AI) ပဵၼ် generative ai မေႃႇတႄႇလ် ဢၼ်ၼိုင်ႈ ဢၼ် Meta (Facebook) ၶဝ်ၶူင်သၢင်ႈဢွၵ်ႇမႃးဝႆႉ လႄႈပဵၼ် မေႃႇတႄႇလ်ဢၼ်ပိုတ်ႇငဝ်ႈတိုၼ်း (Open-Source) ႁႂ်ႈဢဝ်ၸႂ်ႉတိုဝ်းလႆႈၵႂၢင်ႈၵႂၢင်ႈၶႂၢင်ၶႂၢင် ႁူမ်ႈတင်းႁဵတ်း Fine-tunnine လႆႈ။

Llama ပဵၼ်မေႃႇတႄႇလ်ဢၼ်ၼိုင်ႈဢၼ်လႆႈဝႃႈၵတ်ႉၶႅၼ်ႇ ၼႂ်းၵႄႈၵၢင်မေႃႇတႄႇလ်ဢၼ်ဢမ်ႇပိုတ်ႇငဝ်ႈတိုၼ်းတၢင်ႇဢၼ် မိူၼ်ၼင်ႇ GPT3, GPT4, Gemini, Claud.ai ၸိူဝ်းၼႆႉ။

Llama ပဵၼ် multilingual pre-trained model ဢၼ်ၸႂ်ႉၶေႃႈမုၼ်းယႂ်ႇလူင် လႄႈႁႅင်းတိုၼ်းငိုၼ်းၼမ် တႃႇတေႁဵတ်းဢွၵ်ႇပၼ်မႃး checkpoint ဢၼ်ဢဝ်ၵႂႃႇၸႂ်ႉလႆႈလၢႆလၢႆတီႈ မိူၼ်ၼင်ႇ LangChain, Ollama။

ၵူၺ်းၵႃႈဝႃႈ တွၼ်ႈတႃႇလိၵ်ႈတႆးတႄႉ လႆႈဝႃႈၸဵမ်ႁႅင်းၶေႃႈမုၼ်း လႄႈလွင်ႈၵိုင်ႇတၢၼ်ႇတႃႇတေႁဵတ်းဢွၵ်ႇပႆႇတဵမ်ထူၼ် လႄႈ မေႃႇတႄႇလ်တင်းၼမ်ဢၼ်ဢွၵ်ႇမႃးၼၼ်ႉၵေႃႈ ပႆႇႁူႉပွင်ႇၸႂ်လႆႈလိၵ်ႈတႆးလီလီၼၼ်ႉယဝ်ႉ။

## Low-resources Language

ၽႃႇသႃႇလိၵ်ႈလၢႆးၼႂ်းၵမ်ႇၽႃႇ ဢမ်ႇၵွမ်ႉၵႃႈလိၵ်ႈတႆးၵူၺ်း လၢႆလၢႆၽႃႇသႃႇ လႆႈထုၵ်ႇၼပ်ႉဝႃႈပဵၼ် ၽႃႇသႃႇဢၼ်မီးၶေႃႈမုၼ်းၵမ်ႉဢေႇ (Low-Resources Language) ၼၼ်ႉပွင်ႇဝႃႈ ၶေႃႈမုၼ်းလိၵ်ႈလၢႆ ၼမ်ႉၵႂၢမ်း သဵင်လၢတ်ႈ ဢၼ်ပဵၼ် digital-format လႄႈၽွမ်ႉၸႂ်ႉၼႂ်းၶၵ်ႉၵၢၼ် NLP လႄႈ Machine-Learning, Deep-Learning training ၼၼ်ႉဢမ်ႇပႆႇမီးၼမ်ပဵင်းပေႃး။

ၶၵ်ႉၵၢၼ် NLP လႄႈ Machine-Learning, Deep-Learning training ၸိူဝ်းၼၼ်ႉပဵၼ်ၶၵ်ႉၵၢၼ်ဢၼ် တွင်ႉမႆႈၶေႃႈမုၼ်း (Data hungry) တႃႇပွၼ်ႈသွၼ်ပၼ်မၼ်းတင်းၼမ် ဢၼ်ႉၵႆႉႁွင်ႉဝႃႈ large-scale datasets ၶေႃႈၼႆႉတေပိူင်ႈၵၼ်တင်းၶေႃႈၵႂၢမ်းဢၼ်ၵႆႉၺိၼ်းမိူဝ်ႈပူၼ်ႉမႃးဝႃႈ Big-Data ("3 Vs": Volume, Variety, and Velocity.) ၵူၺ်းၶေႃႈမုၼ်းတွၼ်ႈတႃႇ training datasets ၼႆႉလိူဝ်သေလူဝ်ႇမီးတၢင်းယႂ်ႇၼမ်ယဝ်ႉယင်းလူဝ်ႇမီးထႅင်ႈ quality, diversity, and relevance။

## Why and What is Fine-tuning

ၼင်ႇဝႃႈမႃးၼႂ်းတွၼ်ႈ Low-resources language ၼၼ်ႉယဝ်ႉ ပေႃးဝႃႈဢမ်ႇမီးၶေႃႈမုၼ်းတီႈလီ လႄႈၶိုၵ်ႉယႂ်ႇၼၼ်ႉ မေႃႇတႄႇလ်ဢၼ်ဢွၵ်ႇမႃးၼၼ်ႉၵေႃႈ တိုၼ်းဝႃႈတေဢမ်ႇၶိုၵ်ႉၶႅမ်ႈၸႂ်ႉလႆႈ။

လိူဝ်သေလူဝ်ႇၶေႃႈမုၼ်းၶိုၵ်ႉယႂ်ႇယဝ်ႉ တႃႇတေ train deep-learning model သေဢၼ်ဢၼ်တႄႇတီႈငဝ်ႈမၼ်းၼၼ်ႉ လႆႈၸႂ်ႉၶၢဝ်းယၢမ်း လႄႈငိုၼ်းလူင်းတိုၼ်းၼမ် ၵႃႈၶၼ်ယႂ်ႇ၊ ယွၼ်ႉၼၼ်လႄႈ ၸင်ႇမီးလွၵ်းလၢႆးဢၼ်ႁွင်ႉဝႃႈ Fine-Tune ၼၼ်ႉမႃး။

Fine-Tune ၼၼ်ႉပဵၼ်လွၵ်းလၢႆးၼိုင်ႈ ဢၼ်ႁွင်ႉဝႃႈ transfer-learning ၵၢၼ်သိုပ်ႇသူင်ႇတၢင်းႁူႉ ဢၼ် pre-trained မေႃႇတႄႇလ်ၼၼ်ႉလႆႉထုၵ်ႇၾိုၵ်းသွၼ် ႁဵၼ်းႁူႉမႃး၊ မိူၼ်ၼင်ႇ Llama, GPT3, GPT4 ၸိူဝ်းၼၼ်ႉ လုၵ်ႉတီႈၽူႈၶူင်သၢင်ႈၶဝ်သေ ၸႂ်ႉတင်းတိုၼ်းလၢင်း ၶၢဝ်းယၢမ်း လႄႈၶေႃႈမုၼ်းၶိုၵ်ႉယႂ်ႇ တူဝ်ႈဢိၼ်ႇတႃႇၼႅတ်ႉတင်းလုမ်ႈၾႃႉ ၾိုၵ်းသွၼ်မႃးဝႆႉယဝ်ႉ၊ ၵၢၼ် fine-tune မေႃႇတႄႇလ်ၵေႃႈမိူၼ်ၼင်ႇၵၢၼ် ၶိုၼ်းလုပ်ႈၶျေႃး ပွတ်ႈလပ်ႉထႅင်ႈႁႂ်ႈမၼ်းႁဵၼ်းႁူႉ ၸွမ်းၼင်ႇၶေႃႈမုၼ်းႁဝ်းမီးထႅင်ႈၼၼ်ႉယဝ်ႉ။

ယွၼ်ႉၼၼ်လႄႈ တွၼ်ႈတႃႇၽႃႇသႃႇလိၵ်ႈလၢႆးတႆးႁဝ်း မိူဝ်ႈဢၼ်ပႆႇမီးတိုၼ်းလၢင်း လႄႈၶေႃႈမုၼ်းၶိုၵ်ႉယႂ်ႇ တႃႇပွၼ်ႈသွၼ်ပၼ်မၼ်းၼၼ်ႉ လွၵ်းလၢႆး Fine-tune တေပဵၼ်လၢႆးဝႆး လႄႈၵိုင်ႇငၢမ်ႇၸွမ်းငဝ်းလၢႆးယဝ်ႉ။

## Datasets and Fine-tune processes

**Code လႄႈလွၵ်းလၢႆး fine-tune တွၼ်ႈၼႆႉၸႂ်ႉတိုဝ်း [AI-Commandos/LLaMa2lang Convenience scripts](https://github.com/AI-Commandos/LLaMa2lang) ဢၼ်တူင်ႇဝူင်းၸွႆႈၵၼ်ပိုၼ်ၽႄဝႆႉ ပိူဝ်ႈတႃႇ Optimized လႄႈ လႆႈၼမ်ႉတွၼ်းၼႂ်းမေႃႇတႄႇလ်သုင်သုတ်း။**

**Code ဢၼ်ႁၢင်ႈႁႅၼ်းဝႆႉတွၼ်ႈတႃႇၽႃႇသႃႇတႆး [https://github.com/NoerNova/LLaMa2lang.git](https://github.com/NoerNova/LLaMa2lang.git)**

ၶေႃႈမုၼ်းဢၼ်တႃႇတေၸႂ်ႉၼႂ်းၶၵ်ႉတွၼ်ႈၵၢၼ် fine-tune Llama ဢမ်ႇၼၼ်တီႈၼႆႈပဵၼ် Chat-Llama ၼၼ်ႉ ႁဝ်းလူဝ်ႇၶေႃႈမုၼ်း ထၢမ်-တွပ်ႇ ပိူဝ်ႈတႃႇႁႂ်ႈမၼ်းႁဵၼ်းႁူႉလႆႈဝႃႈ ပေႃးမီးၶေႃႈထၢမ် မၼ်းတေလႆႈတွပ်ႇၸိူင်ႉႁိုဝ်။

(သူၼ်ၸႂ်လဵပ်ႈႁဵၼ်း - [Fine-tune GPT2 for Shan text-generator](blog/fine-tuning-gpt2-for-shan-language))

![Sample datasets {caption: prompter-assistant dataset}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2567-07-25-at-02.19.07.png)

![Sample datasets {caption: prompter-assistant dataset}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2567-07-25-at-02.19.42.png)

ၼင်ႇၶေႃႈမုၼ်းၽၢႆႇၼိူဝ်ၼၼ်ႉ ပေႃးၽူႈထၢမ် (prompter) လၢတ်ႈဝႃႈ "မႂ်ႇသုင်ၶႃႈ" ၼႆ assistant တေလႆႈတွပ်ႇဝႃႈၸိူင်ႉႁိုဝ် ၼႆၼၼ်ႉယဝ်ႉ။

ၶေႃႈမုၼ်းၸိူင်ႉၼႆၼႆႉ လႆႈၸႂ်ႉၶၢဝ်းယၢမ်းလႄႈလွင်ႈသိုပ်ႇႁႃၶေႃႈမုၼ်းတင်းၼမ် တႃႇတေလႆႈၶေႃႈ ထၢမ်-တွပ်ႇ ဢၼ်ပဵၼ်ၶေႃႈမုၼ်းမၢၼ်ႇမႅၼ်ႈ လႄႈလွင်ႈထတ်းတူဝ်ၽိတ်းထုၵ်ႇမၼ်း၊ ၶေႃႈမုၼ်းဢၼ်ပဵၼ်ၽႃႇသႃႇတႆးၵေႃႈ တိုၵ်ႉယူႇၼႂ်းၶၵ်ႉတွၼ်ႈၵၢၼ်ၵဵပ်းႁွမ်း လႄႈဢမ်ႇပႆႇမီးလွင်ႈၶိုၵ်ႉၼမ်။

ၵွပ်ႈၼၼ်ၼႂ်းတွၼ်ႈၼႆႉ ႁဝ်းတေၸႂ်ႉလွၵ်ႉလၢႆးၵၢၼ်ပိၼ်ႇၽႃႇသႃႇၸုမ်ႇၶေႃႈမုၼ်း ဢၼ်ၸိုဝ်ႈဝႃႈ [OASST1 (OpenAssistant)](https://huggingface.co/datasets/OpenAssistant/oasst1) ဢၼ်ဢိင်ၼိူဝ် AI model [NLLB - သိုပ်ႇလူ](blog/meta-NLLB-shan-machine-translations) ၵၢၼ်ပိၼ်ႇၽႃႇသႃႇဢၼ်ဢွၵ်ႇမႃးဢွၼ်တၢင်းၼၼ်ႉ။

### 0. Pre-requirements

1. ၵၢၼ် fine-tune LLM model ၼႆႉၸႂ်ႉတိုဝ်းႁႅင်း computational တင်းၼမ်လႄႈ သင်ဝႃႈၸႂ်ႉတိုဝ်း Graphic card မိူၼ်ၼင်ႇ Nvidia GPU ဢၼ်မီး VRAM လႄႈ CUDA တေတိူဝ်းဝႆး လႄႈလီလိူဝ်။
2. ၸႂ်ႉတိုဝ်း python venv ဢမ်ႇၼၼ် anaconda, miniconda တေႁဵတ်းႁႂ်ႈ setup fine-tune environment လႆႈငၢႆႈ။
3. install [pytorch - https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) use of CUDA preferable.

### 1. Clone project and install requirements

```bash
git clone https://github.com/NoerNova/LLaMa2lang.git
```

```bash
pip install -r requirements.txt
```

### 2. Translate base dataset

တီႈၼႆႈႁဝ်းတေၸႂ်ႉတိုဝ်းမေႃႇတႄႇလ် NLLB တႃႇပိၼ်ႇၽႃႇသႃႇၸုမ်ႇၶေႃႈမုၼ်း OASST1 လႄႈ ၼႂ်း ```translators/nllb.py``` ၼၼ်ႉထႅမ်သႂ်ႇပၼ်ပႃး ```"shn": "shn_Mymr```  ဢၼ်ပဵၼ် language_mapping ၽႃႇသႃႇတႆး။

```python
...
class NLLBTranslator(BaseTranslator):
    language_mapping = {
        "en": "eng_Latn",
        "es": "spa_Latn",
        ...
        "shn": "shn_Mymr",
        "bn": "mni_Beng", # ၼႂ်း orginal ဢမ်ႇမီးပႃးၽႃႇသႃႇၼႆႉလႄႈ ၸၢင်ႈ error မိူဝ်ႈပိၼ်ႇၽႃႇသႃႇၸူးၽႃႇသႃႇဢၼ်ၼႆႉ
    }
    ...
```

Run translate to Translate OASST1 to shn language

```python
python translate.py --max_length 1024 nllb --model_size 3.3B shn_Mymr ./output_shn
```

Script ၽၢႆႇၼိူဝ်ၼႆႉ ၸႂ်ႉမေႃႇတႄႇလ် nllb3.3B သေၶိုၼ်းပိၼ်ႇၽႃႇသႃႇၸုမ်ႇၶေႃႈမုၼ်း OASST1 ႁႂ်ႈပဵၼ်ၽႃႇသႃႇတႆး။
သင်ၸိူဝ်ႉဝႃႈ VRAM ႁဝ်းဢမ်မီးၼမ် ႁိုဝ်မီး Error ဝႃႈ CUDA out of memory ၼႆၸၢမ်းယွမ်းတူၺ်း parameter --max_langth ဢမ်ႇၼၼ်လႅၵ်ႈတူၺ်းမေႃႇတႄႇလ်ပဵၼ် nllb1.3။

![NLLB {caption: translate_with_nllb}](blog/assets/fine-tuning-llama3-for-shan-language/translate_with_nllb.jpg)

ၶၵ်ႉတွၼ်ႈၼႆႉၸႂ်ႉၶၢဝ်းယၢမ်းပိၼ်ႇၽႃႇသႃႇ **20 ၸူဝ်ႈမွင်းလိူဝ်လိူဝ်** မီးထႅဝ်ၶေႃႈမုၼ်းမွၵ်ႈ 84,000 ပၢႆ။

ၶေႃႈမုၼ်း output ပိၼ်ႇၽႃႇသႃႇၼႆႉပိုၼ်ၽႄဝႆႉပၼ်တီႈ [huggingface - NorHsangPha/oasst1_shan_translation](https://huggingface.co/datasets/NorHsangPha/oasst1_shan_translation)

### 2.1. Dataset checking

![Checking Datasets {caption: checking_dataset}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-26-002559.png)

ၶေႃႈမုၼ်းဢၼ်ပိၼ်ၽႃႇသႃႇမႃးၼႆႉ ၵမ်ႉဢေႇၵူၺ်းဢၼ်မီးလွင်ႈမၢၼ်ႇမႅၼ်ႈ လႄႈၼမ်ႉၵႂၢမ်းၵေႃႈဢမ်ႇမိူၼ်ၼမ်ႉၵႂၢမ်းဢၼ်ၽႃႇသႃႇတႆးၸႂ်ႉတိုဝ်းၵၼ်၊ လိူဝ်သေၼၼ်ႉ ယင်းထူပ်းပၼ်ႁႃ repetitive problem ဢၼ်မီးတူဝ်သွၼ်ႉလၢႆလၢႆတူဝ်ဢၼ်ဢမ်ႇပွင်ႇၵႂၢမ်းမၢႆသင်၊ ပေႃးပိူင် LLM တႄႉလႆႈဝႃႈပဵၼ်ၶေႃႈမုၼ်းဢၼ်ဢမ်ႇၸႂ်ႉလႆႈလီလီၼၼ်ႉယဝ်ႉ။

ၵူၺ်းၵႃႈတႃႇၸၢမ်းတူၺ်းၵူၺ်းၼႆလႄႈ ႁဝ်းၶႃႈတေသိုပ်ႇၵႂႃႇထႅင်ႈၶၵ်ႉတွၼ်ႈ Fine-tune ႁဝ်းၶႃႈ။

### 3. Combine dataset checkpoint

![Sample datasets {caption: output_shn}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-26-004433.png)

```python
python combine_checkpoints.py ./output_shn <local_folder or HF repo>
```

တွၼ်ႈၼႆႉတေဢဝ်ၸုမ်ႇၶေႃႈမုၼ်းဢၼ်ပိၼ်ႇၽႃႇသႃႇယဝ်ႉၼၼ်ႉ ၶိုၼ်းႁူမ်ႈၵၼ်ႁႂ်ႈပဵၼ် HF's dataset format သေလႄႈသိမ်းဝႆႉၼႂ်း folder မႂ်ႇ ဢမ်ႇၼၼ် Huggingface repo၊ ပေႃးဝႃႈတေသိမ်းၼႂ်း Huggingface repo ၼႆႉႁႂ်ႈပၼ်ပႃး HF_TOKEN env။

![Datasets Checkpoint {caption: combine_checkpoint}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-26004449.png)

### 4. Finetune

```python
python finetune.py  --base_model --batch_size 2 llama3_shan_finetuned .\combine_checkpoints "You are a generic chatbot that always answers in Shan."
```

ၶၵ်ႉတွၼ်ႈၼႆႉပဵၼ်ၵၢၼ် Train ပွၼ်ႈသွၼ်ပၼ်ၶေႃႈမုၼ်းမႂ်ႇပၼ်မေႃႇတႄႇလ် Llama3 (default: NousResearch/Meta-Llama-3-8B-Instruct)။

ၶၵ်ႉတွၼ်ႈၼႆႉၸႂ်ႉတိုဝ်းႁႅင်း VRAM တင်းၼမ်လႄႈႁဝ်းတေၸၢမ်းၵႂႃႇၸႂ်ႉ CloudGPU တီႈ [vast.ai](https://vast.ai/)။

#### Vast.ai

vast.ai ပဵၼ် service ပၼ်ႁိမ် GPU ၸူဝ်ႈၵမ်း လႄႈမီးၵႃႈၶၼ်ထုၵ်ႇ မိူၼ်ၼင်ႇတီႈၼႆႈ ႁဝ်းလၢမ်းဝႃႈလူဝ်ႇ VRAM မွၵ်ႈ 48GB ၼႆ တေတူၵ်းမွၵ်ႈ **1 ၸူဝ်ႈမွင်း 1 တေႃႇလႃႇ** ပေႃးဝႃႈႁဝ်း train ဢမ်ႇႁိုင်ၼႆတႄႉ တေၸၢင်ႈသၢင်ႇထုၵ်ႇယူႇ ၼင်ႇၵၢၼ် finetune llama3 လူၺ်ႈၶေႃႈမုၼ်းဢမ်ႇထိုင် 100,000 record ၸိူဝ်းၼႆႉတေဢမ်ႇပူၼ်ႉသေ 3 ၸူဝ်ႈမွင်း။

![VastAI {caption: vast.ai}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-26-005957.png)

![Train on VastAI {caption: train on RTX6000Ada}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-08-033145.png)

Finetune ၼိူဝ် Vast.ai လိူၵ်ႈၸႂ်ႉ GPU RTX6000Ada ဢၼ်မီးႁႅင်း VRAM 48GB ၸႂ်ႉမိူဝ်ႈၽွင်း train ၼၼ်ႉ 42GB ၸႂ်ႉၶၢဝ်းယၢမ်း setup လႄႈ train 2:45 ၸူဝ်ႈမွင်း။

### 5. Inference

ဝၢႆးသေ Finetune ယဝ်ႉၼၼ်ႉႁဝ်းတေလႆႈမႃးမေႃႇတႄႇလ် safetensors ဢၼ်ႁဝ်းတေဢဝ်မႃးၸၢမ်းၸႂ်ႉတူၺ်း။

```python
import transformers
from transformers import pipeline
import torch

model_id = "merge_llama3_adapter_Shan"

text_pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

# message = [
#     "သူပဵၼ်ၽူႈၸွႆႈထႅမ် ဢၼ်တေတွပ်ႇပၼ်ၶေႃႈတွပ်ႇၵူႈလွင်ႈလွင်ႈ",
# ]

messages = [
    {"role": "system", "content": "You are a genetic chatbot who always responds in Shan."},
    {"role": "user", "content": "တႅမ်ႈပၼ် Hello World ၼႂ်း 10 ၽႃႇသႃႇ"},
]

prompt = text_pipeline.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

terminators = [
    text_pipeline.tokenizer.eos_token_id,
    text_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = text_pipeline(
    prompt,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

print(outputs[0]["generated_text"][len(prompt):])

# output
# မႂ်ႇသုင်ၶႃႈ မိူဝ်ႈၼႆႉပဵၼ် ပရူဝ်ႇၵရမ်ႇ ဢၼ်တႅမ်ႈဝႆႉၼႂ်းၽႃႇသႃႇၵႂၢမ်းလၢတ်ႈလၢႆလၢႆၽႃႇသႃႇၶႃႈဢေႃႈ မႂ်ႇသုင်ၶႃႈ မိူဝ်ႈၼႆႉပဵၼ် ပရူဝ်ႇၵရမ်ႇ ဢၼ်တႅမ်ႈဝႆႉၼႂ်းၽႃႇသႃႇၵႂၢမ်းလၢတ်ႈလၢႆလၢႆၽႃႇသႃႇၶႃႈဢေႃႈ မႂ်ႇသုင်ၶႃႈ မိူဝ်ႈၼႆႉပဵၼ် ပရူဝ်ႇၵ�
```

တေလႆႈႁၼ်ဝႃး မၼ်းထုတ်ႇဢွၵ်ႇလိၵ်ႈတႆးလႆႈယဝ်ႉ ၵူၺ်းႁဝ်းၶႂ်ႈလႆႈ Chat interface လႄႈၸႂ်ႈမိူၼ်ၼင်ႇႁႂ်ႈမၼ်းပဵၼ်ၵၢၼ်လၢတ်ႈတေႃႇၵၼ်ၼၼ်ႉ ႁဝ်းတေမီးၶၵ်ႉတွၼ်ႈထႅင်ႈဢိတ်းဢွတ်း

## Quantized + Ollama

Chat interfeace framework ဢၼ်ပိူၼ်ႈၵႆႉၸႂ်ႉၵၼ်မီးလၢႆလၢႆဢၼ်မိူၼ်ၼင်ႇ Chainlit, LangChain,GPT4All, llama.cpp လႄႈ Ollama၊ framework ဢၼ်လႂ်ဢၼ်ၼၼ်ႉ တေၸၢင်ႈၸႂ်ႉတိုဝ်းမဵဝ်းၶွင်မေႃႇတႄႇလ်ပႅၵ်ႇပိူင်ႈၵၼ်။

တွၼ်ႈၼႆႉႁဝ်းတေၸႂ်ႉ Ollama ဢၼ်ၸႂ်ႉတိုဝ်း .gguf ၼႆလႄႈႁဝ်းလူဝ် convert safetensors model ႁဝ်ႁႂ်ႈပဵၼ်ၸွမ်းၼင်ႇဢၼ်မၼ်းၸႂ်ႉၼၼ်ႉ။

> Ollama ၼၼ်ႉၸႂ်ႉတိုဝ်း llamacpp သေၵေႃႇသၢင်ႈၶိုၼ်ႈမႃးႁႂ်ႈပဵၼ် user-friendly ဢၼ်ၸႂ်ႉတိုဝ်းငၢႆႈ လႄႈ setup ငၢႆႈ

တႃႇတေ Convert model ႁဝ်းတေၸႂ်ႉ script convert ၼႂ်း llamacpp လႅၵ်ႈမေႃႇတႄႇလ်ႁဝ်းႁႂ်ႈပဵၼ် c++ .gguf လႄႈ Quantized ႁဵတ်းႁႂ်ႈမေႃႇတႄႇလ်ႁဝ်းလဵၵ်ႉမႃးထႅင်ႈ (မေႃႇတႄႇလ်လဵၵ်ႉၵေႃႈတေၵၼ် RAM ဢႄႇ)။

### Install llamacpp

Clone and build local

```bash
git clone https://github.com/ggerganov/llama.cpp.git

cd llama.cpp
```

```bash
# using make:
make

# using Cmake:
cmake -B build
# or
cmake --build build --config Release
```

easy install on MacOS and Linux

```bash
# macos
brew install llama.cpp
```

ဝၢႆးသေ install ယဝ်ႉတေလႆႈ script လၢႆလၢႆဢၼ်မိူၼ်ၼင်ႇ

- llama-cli
- llama-server
- llama-quantize

### Convert and Quantized

convert finetuned model ဢၼ်ႁဝ်းဢဝ်ၵႂႃႇဝႆႉၼိူဝ် Huggingface

```python
pip install -r requirements.txt

python convert_hf_to_gguf.py <models/model_repo>
```

quantize 4-bits model ႁဵတ်းႁႂ်ႈ model ႁဝ်းလဵၵ်းမႃး ၸႂ်ႉ llama-quantize script လႄႈ မေႃႇတႄႇလ်ဢၼ် converted ယဝ်ႉတၢင်းၼိူဝ်ၼၼ်ႉ

```python
./llama-quantize ./models/<converted_model>/ggml-model-f16.gguf ./models/ggml-model-Q4_K_M.gguf Q4_K_M
```

official example for convert and quantize - [llama.cpp/examples/quantize](https://github.com/ggerganov/llama.cpp/tree/master/examples/quantize)

quantized models - [merge_llama3_adapter_Shan](https://huggingface.co/NorHsangPha/merge_llama3_adapter_Shan/blob/main/ggml-model-Q4_K_M-v2.gguf)

### Ollama + Open WebUI

Ollama ပဵၼ်ပရူဝ်ႇၵရႅမ်ႇဢၼ်ၸၼ်လူတ်ႇလူင်းလႄႈ install လႆႈၵူႈၽရႅတ်ႉၾွမ်ႇ [macOS/Linux/Windows](https://ollama.com/download)

ဝၢႆးသေ install ယဝ်ႉၵေႃႉၸႂ်ႉလႆႈၼႂ်း terminal ၵမ်းလဵဝ်ယဝ်ႉ။

![Ollama {caption: ollama_cli}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-26-021057.png)

ၼင်ႇႁိုဝ်တေပဵၼ် Chat interface လႄႈမီးၶိူင်ႈမိုဝ်းတႃႇၸၢမ်း LLM ႁဝ်းတင်းၼမ်ၼၼ်ႉ ႁဝ်းတေသႂ်ႇထႅင်ႈ Open WebUI တႃႇတေ serve ollama

```python
pip install open-webui
```

![Open WebUI {caption: open-webui}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-26-021416.png)

#### Run model with Ollama

ႁဵတ်း file ဢၼ်ၼိုင်ႈသေသႂ်ႇပၼ်ထႅဝ်ၼႆႉ ယဝ်ႉသေ save ပဵၼ်ၸိုဝ်ႈသင်ၵေႃႈလႆႈ။

```bash
# path to model
From ggml-model-Q4_K_M-v2.gguf
```

ၼႂ်း terminal run

```bash
ollama create shandemo -f ./shandemo
# ollama create ၸိုဝ်ႈမေႃႇတႄႇလ်သင်ၵေႃႈလႆႈ -f ./ၸိုဝ်ႈၾၢႆႇလ်
```

check တူၺ်း

![Ollama list {caption: ollama list}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-26-022150.png)

#### Serve with Open WebUI

ၼႂ်း terminal run

```bash
open-webui serve
```

![Open WebUI {caption: open-webui serve}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-26-022542.png)

တီႈၼႂ်း Web Browser ၶဝ်ႈၵႂႃႇတီႈ ```http://localhost:8080```

ပွၵ်ႈဢွၼ်တၢင်းသုတ်းၼၼ်ႉ တေလႆႈၵေႃႇသၢင်ႈဢၵွင်ႇဢွၼ်တၢင်း Signup ၸႂ်ႉသင်ၵေႃႈလႆႈ ဢမ်ႇတၢပ်ႈလူဝ်ႇပဵၼ်ဢီးမေးလ်ဢၼ်တႄႉမၼ်း၊ ၵေႃႉဢွၼ်တၢင်းသုတ်းၼၼ်ႉ တေပဵၼ် admin။

![Open WebUI {caption: open-webui login}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-26-022659.png)

ၼႃႈဝႅပ်ႉသၢႆႉ တေငၢႆးငၢႆးမိူၼ် ChatGPT ယူႇ။

![Open WebUI {caption: webui}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-26-022828.png)

ၵႂႃႇတီႈၼႂ်း Workspace သေတေႁၼ်ဝႃႈ Ollama model ႁဝ်းၼၼ်ႉ run ဝႆႉယူႇယဝ်ႉ။

![Open WebUI {caption: webui}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-26-023439.png)

တဵၵ်းတီႈ model ႁဝ်းၼၼ်ႉသေၸၢမ်းလၢတ်ႇတူၺ်းၼႂ်း Chat

![Open WebUI {caption: webui-chat1}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-11-020547.png)

![Open WebUI {caption: webui-chat2}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-11-023045.png)

သင်ဝႃႈမေႃႇတႄႇဢမ်ႇပၼ်ၶေႃႈတွပ်ႇလႆႈလီၵေႃႈ ၸၢမ်း edit တူၺ်း system prompt လႄႈ parameter ၸိူဝ်းၼႆႉတူၺ်း

![Open WebUI {caption: webui-chat2}](blog/assets/fine-tuning-llama3-for-shan-language/Screenshot-2024-07-26-024329.png)

## Conclution

တေႁၼ်ဝႃႈ မေႃႇတႄႇလ်ဢၼ်ႁဝ်းႁဵတ်း finetuned ၼၼ်ႉ မၼ်းၶတ်းၸႂ်ပွင်ႇၸႂ်လိၵ်ႈတႆးယူႇၵမ်ႈၽွင်ႈသေတႃႉၵေႃႈ ၸဵမ်ၶေႃႈမုၼ်းဢႄႇ လႄႈၼမ်ႉတိုၼ်းၶေႃႈမုၼ်း quality ဢမ်ၶိုၵ်ႉၶႅမ်ႉပဵင်းပေႃးလႄႈ မေႃႇတႄႇလ်ႁဝ်းၵေႃႈ ဢမ်ႇၶိုၵ်ႉၶမ်ႇသင်ၵႃႈႁိုဝ် ၸႂ်ႉတိုဝ်တႄႉတႄႉဢမ်ႇပႆႇလႆႈ။

ၵူၺ်းၵႃႈၵေႃႈ ပဵၼ်လွင်ႈလႆႈလဵပ်ႈႁဵၼ်းၶၵ်ႉတွၼ်ႈလႄႈလွင်ႈပွင်ႇၸႂ်ဝႃႈ LLM ဢၼ်ၵူႈမိူင်းမိူင်း ၵူႈၽႃႇသႃႇၶတ်ႈၸႂ်ၶဵင်ႇတႃႉၵၼ်ယူႇၼၼ်ႉ မၼ်းပဵၼ်သင် မီးလွၵ်းလၢႆးၸိူင်ႉႁိုဝ်တႃတေႁဵတ်းႁႂ်ႈပဵၼ်မႃး တႃႇတေႁႂ်ႈၶိုတ်းၸၼ်ႉပဵင်းပိူၼ်။

ယူႇတီႈႁဝ်းၶႃႈၶဝ်သေ တိုၵ်ႉသိုပ်ႇလမ်း သိုပ်ႇလဵပ်ႈႁဵၼ်းလႄႈ ၶတ်းၸႂ်ႁဵတ်းဢွၵ်ႇၶေႃႈမုၼ်း ႁႂ်ႈမီးဝႆႉၶေႃႈမုၼ်းၽွမ်ႉၸႂ်ႉဢၼ်ၶွမ်ႊပၼီႊယႂ်ႇလူင်ၶဝ်ၸၢင်ႈဢဝ်ၵႂႃႇႁွမ်းၸွမ်းသေ ႁဵတ်းပၼ်မႃးပႃးလိၵ်ႈတႆးႁႂ်ႈၸႂ်ႉတိုဝ်းလႆႈလီၼႂ်းၵၢပ်ႈပၢၼ် AI ဢမ်ႇတူၵ်းလႃႈတူၵ်းလိုၼ်း ပႆႉပၼ်ႁႅင်းသေၵမ်းၶႃႈ :D

## Links

- Finetuned models: [https://huggingface.co/NorHsangPha/merge_llama3_adapter_Shan](https://huggingface.co/NorHsangPha/merge_llama3_adapter_Shan)
- Translated datasets: [https://huggingface.co/datasets/NorHsangPha/oasst1_shan_translation](https://huggingface.co/datasets/NorHsangPha/oasst1_shan_translation)
- LLaMa2lang [custom for shan]: [https://github.com/NoerNova/LLaMa2lang](https://github.com/NoerNova/LLaMa2lang)
