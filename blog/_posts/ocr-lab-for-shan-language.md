---
tags: ["shan" ,"language" ,"ocr" ,"shannlp"]
date: March 4, 2025
title: OCR လွင်ႈႁူႉႁၼ်တူဝ်လိၵ်ႈ လူၺ်ႈသၢႆလႅင်း (Labs)
subtitle: လွင်ႈတႅမ်ႈမၢႆၵၢၼ်လဵပ်ႈႁဵၼ်းၼႂ်းႁူဝ်ၶေႃႈ OCR လႄႈၽႃႇသႃႇတႆး
image: https://i.pinimg.com/736x/fe/90/9d/fe909d46154586ff48efe5af61310e45.jpg
link: blog/ocr-lab-for-shan-language
description: Optical Character Recognition (OCR) လွင်ႈႁူႉႁၼ်တူဝ်လိၵ်ႈ လူၺ်ႈသၢႆလႅင်း။ ပဵၼ်ထႅၵ်ႉၶၼေႃႇလေႃႇၸီႇ ဢၼ်လမ်ႇလွင်ႈတႃႇ digitizing ၽႃႇသႃႇဢၼ်ၼိုင်ႈၼႂ်းပၢႆးၶွမ်ႇပိဝ်ႇတႃႇ၊ ၵူၺ်းၵႃႈ ယင်းမီးၽႃႇသႃႇထႅင်ႈတင်းၼမ်ဢၼ်ပႆႇမီး OCR ဢၼ်ၸႂ်ႉတိုဝ်းလႆႈလီလီ၊ မိူၼ်ၼင်ႇၽႃႇသႃႇတႆးႁဝ်းယဝ်ႉ။
---

## Contents

- [Introduction](#introduction)
- [Tesseract-OCR](#tesseract-ocr)
- [Datasets](#step-1-preparing-the-dataset)
- [Training](#step-2-training-tesseract-for-shan)
- [Challengs](#step-3-addressing-challenges)
- [Results](#results-and-future-plans)
- [Conclusion](#conclusion)

## Introduction

ၶေႃႈမုၼ်းလိၵ်ႈတႆး ဢၼ်ႁဝ်းမီးယူႇယၢမ်းလဵဝ်ၼႆႉ ၸႅၵ်ႇပိူင်ယႂ်ႇဝႆႉပဵၼ် 2 မဵဝ်း မဵဝ်းၼိုင်ႈၵေႃႈ ၶေႃႈမုၼ်းဢၼ်ပဵၼ်ၾေႃႇမႅတ်ႉ **တီႇၵျိတ်ႇတႄႇ** (Digital format) ပွင်ႇဝႃႈၶေႃႈမုၼ်းဢၼ်ယူႇၼႂ်းပိူင်သၢင်ႈၶွမ်ႇပိဝ်ႇတႃႇ (Computer system) ၽွမ်ႉတႃႇတေဢဝ်ၵေႃႇၸႂ်ႉတိုဝ်းလႆႈငၢႆႈလူမ်၊ မိူၼ်ၼင်ႇ ၶေႃႈမုၼ်းလိၵ်ႈၼႂ်းၶွမ်း (word, docx, excel, etc.) ဢမ်ႇၼၼ်ၶေႃႈမုၼ်းဢၼ်ယူႇၼိူဝ်ဢိၼ်ႇထိူဝ်ႇၼႅတ်ႉ၊ ဝႅပ်ႉသၢႆႉ ၸိူဝ်းၼၼ်ႉ။

ၶေႃႈမုၼ်းထႅင်ႈမဵဝ်းၼိုင်ႈသမ်ႉ ပဵၼ်ၶေႃႈမုၼ်းဢၼ်ယူႇၼႂ်းပပ်ႉလိၵ်ႈ ပပ်ႉတႅမ်ႈ ၸိူဝ်းၼႆႉ။ မၢင်ၸိူဝ်းယူႇၼႂ်းၶွမ်းယူႇသေတႃႉ ၵူၺ်းတိုၵ်ႉပဵၼ်ၾေႃႇမႅတ်ႉဢၼ်ဢဝ်ၵႂႃႇၸႂ်ႉတိုဝ်းယၢပ်ႇ မိူၼ်ၼင်ႇ ၶႅပ်းႁၢင်ႈ၊ PDF၊ Book image scan ၸိူဝ်းၼႆႉ။ ၶေႃႈမုၼ်းလိၵ်ႈတႆးဢၼ်ယူႇၼႂ်းၾေႃႇမႅတ်ႉၼႆႉၵေႃႈမီးဝႆႉတင်းၼမ်။

ၵွပ်ႈၼႆလႄႈ ႁူဝ်ၶေႃႈၼႆႉႁဝ်းမႃးၸၢမ်းတူၺ်းလွၵ်းလၢႆးၸၼ်ၶေႃႈမုၼ်းၸိူဝ်းၼၼ်ႉ ႁႂ်ႈပဵၼ်တီႇၵျိတ်ႇတႄႇၾေႃႇမႅတ်ႉလႄႈဢဝ်ၸႂ်ႉတိုဝ်းလႆႈငၢႆႈ လူၺ်ႈလွၵ်းလၢႆး OCR။

Optical Character Recognition (OCR) လွင်ႈႁူႉႁၼ်တူဝ်လိၵ်ႈ လူၺ်ႈသၢႆလႅင်း။ ပဵၼ်ထႅၵ်ႉၶၼေႃႇလေႃႇၸီႇ ဢၼ်လမ်ႇလွင်ႈတႃႇ digitizing ၽႃႇသႃႇဢၼ်ၼိုင်ႈၼႂ်းပၢႆးၶွမ်ႇပိဝ်ႇတႃႇ၊ ၵူၺ်းၵႃႈ ယင်းမီးၽႃႇသႃႇထႅင်ႈတင်းၼမ်ဢၼ်ပႆႇမီး OCR ဢၼ်ၸႂ်ႉတိုဝ်းလႆႈလီလီ၊ မိူၼ်ၼင်ႇၽႃႇသႃႇတႆးႁဝ်းယဝ်ႉ။

ပွင်ႈၵၢမ်းႁူဝ်းၼႆႉ ပဵၼ်လွင်ႈတႅမ်ႈမၢႆၵၢၼ်လဵပ်ႈႁဵၼ်းၼႂ်းႁူဝ်ၶေႃႈ OCR လႄႈပဵၼ်ထႅၵ်ႉၼိၵ်ႉၶူဝ်ႇလ် (technical) တင်းၼမ်။

![Book Example 1 {caption: ပပ်ႉလိၵ်ႈဢၼ်ပဵၼ်ၶႅပ်းႁၢင်ႈ PDF}](blog/assets/ocr-lab-for-shan-language/book_example1.png)

![Book OCR Example {caption: တႅၵ်ႉၼိူင်းၵၼ်တင်းမိူဝ်ႈပႆႇႁဵတ်း OCR သေ copy လႄႈ မိူဝ်းႁဵတ်း OCR}](blog/assets/ocr-lab-for-shan-language/book_example2.png)

## Tesseract-OCR?

Tesseract-OCR ပဵၼ် open-source OCR engines ဢၼ်ပိုတ်ႇငဝ်ႈတိုၼ်း ၸႂ်ႈငၢႆႈၸႂ်ႉလႆႈလွတ်ႈလႅဝ်း ႁွင်းႁပ်ႉ support လၢႆလၢႆ platforms မိူၼ်ၼင်ႇ Windows, Mac, Linux, Python wrapper ၸိူဝ်းၼႆႉ လိူဝ်သေၼၼ်ႉယင်းၸၢင်ႈ training သွၼ်ပၼ်ၽႃႇသႃႇမႂ်ႇမႂ်ႇလႆႈ။

Tesseract-OCR ယင်းၶဝ်ႈၵၼ်လႆႈတင်း AI frameworks မႂ်ႇမႂ်ႇ မိူၼ်ၼင်ႇ ChatGPT ၸိူဝ်းၼႆႉလူးၵွၼ်ႇ၊ ၵွပ်ႈၼၼ်လႄႈလႆႈလိူၵ်ႈၸႂ်ႉ Tesseract-OCR တႃႇလဵပ်းႁဵၼ်းၼႂ်းႁူဝ်ၶေႃႈၼႆႉ။

## Step 1: Preparing the Dataset

ဢွၼ်တၢင်းသုတ်း ဢၼ်လမ်ႇလွင်ႈသုတ်းၼႂ်းၵၢၼ် train AI models ၼၼ်ႉပဵၼ်ၶေႃႈမုၼ်း၊ AI models ဢၼ် train ဢွၵ်ႇမႃးၼၼ်ႉတေၶႅမ်ႉဢမ်ႇၶႅမ်ႉၼၼ်ႉၵေႃႈ ယူႇတီႈၸုမ်ႇၶေႃႈမုၼ်းဢၼ်ႁဝ်းတေသွၼ်ၼၼ်ႉတင်းမူတ်းယဝ်ႉ။

### ၸုမ်ႇၶေႃႈမုၼ်းတွၼ်ႈတႃႇ OCR သမ်ႉလူဝ်ႇၸိူင်ႉႁိုဝ်ၼႆၼၼ်ႉ?

OCR ၼႆႉတေသွၼ်ႇဝႃႈ Input ဢၼ်တေႁပ်ႉမႃးၼၼ်ႉပဵၼ်မိူၼ်ၼင်ႇၶႅပ်းႁၢင်ႈ မိူၼ်ဢဝ်တႃသေတူၺ်းၶႅပ်းႁၢင်ႈၼိုင်ႈဢၼ် (Image) ၼိူဝ်ၶႅပ်းႁၢင်ႈၼၼ်ႉတႅမ်ႈသင်ဝႆႉလၢႆလၢႆၼႆ ပေႃးပဵၼ်လိၵ်ႈဢၼ်မၼ်းဢမ်ႇယၢမ်ႈႁဵၼ်းသွၼ်မႃးသေပွၵ်ႈ မၼ်းၵေႃႈတေဢမ်ႇၸၢင်ႈႁူႉလႆႈ၊ ႁဝ်းတေလႆႈလၢတ်ႈၼႄ တႅမ်ႈၼႄ ႁႂ်ႈမၼ်းႁူႉၸၵ်းသေၵွၼ်ႇ (ground truth text)

ၵွပ်ႈၼၼ်ၶေႃႈမုၼ်းတႃႇသွၼ်ပၼ် OCR တေလႆႈမီးသွင်တွၼ်ႈ

1. ၶႅပ်းႁၢင်ႈ - Image1.tif (tesseract-ocr ၸႂ်ႉၶႅပ်းႁၢင်ႈ tif, png, jpeg) ၶႅပ်းႁၢင်ႈတီႈၼႆႈၼႆႉ မိူၼ်ၼင်ႇ screen-shot, ၶႅပ်းႁၢင်ႈဢၼ်ဢဝ် camera သေပေႃႉဝႆႉ, ၶႅပ်းႁၢင်ႈလၢႆႈမိုဝ်းတႅမ်ႈ ၸိူဝ်းၼႆႉ
2. လိၵ်ႈဢၼ်မီးၼိူဝ်ၶႅပ်းႁၢင်ႈၼၼ်ႉ - Image1.gt.txt ၸိုဝ်ႈၾၢႆႇလ်တေလႆႈမိူၼ်ၵၼ်တင်းၾၢႆႇလ်ၶႅပ်းႁၢင်ႈ လႄႈ prefix တေလႆႈပဵၼ် .gt.txt
(Optional) 3. Box data, OCR မၢင်ၸိူဝ်းဢမ်ႇမီး Automatic box bounding လႄႈတေလႆႈၸၼ်ႁင်းၵူၺ်း၊ တွၼ်ႈၼႆႉႁဝ်းၸႂ်ႉ Tesseract-OCR V5 လႄႈ ဢမ်ႇလူဝ်ႇ။

![Box example {caption: ၵၢၼ်ႁဵတ်းၶေႃႈမုၼ်း Box}](blog/assets/ocr-lab-for-shan-language/box_example.png)

ၶေႃႈမုၼ်း OCR ဢၼ်ၵူႈၵၼ်ဝႆႉၸိူင်ႉၼႆတွၼ်ႈတႃႇလိၵ်ႈတႆးလႆႈဝႃႈဢမ်ႇပႆႇမီးသေပွၵ်ႈ ၼႆလႄႈတီႈၼႆႈၼႆႉႁဝ်းတေၸႂ်ႉ python script သေ generate ဢဝ်

### Collecting and Cleaned Text Data

ၶေႃႈမုၼ်း text တီႈၼႆႈၼႆႉတေၸၼ်ဢဝ်တီႈၶေႃႈမုၼ်းဢၼ်လႆႈၸၼ်ဝႆႉသေတၢင်ႇဝႆႉၼႂ်း huggingface မိူၼ်ၼင်ႇ tainovel.com. shannews.org, taifreedom.com, shn.wikipedia.com ၸိူဝ်းၼႆႉ [huggingface.com/haohaa](huggingface.com/haohaa)
ႁဝ်းၶႃႈ HaoHaa လႆႈၸၼ်ၶေႃႈမုၼ်းလိၵ်ႈတႆးၼိူဝ်ဝႅပ်ႉသၢႆႉသ်ဢၼ်တႅမ်ႈလိၵ်ႈတႆးၵူႈဝႅပ်ႉသၢႆႉသ်ဢၼ်ႁႃလႆႈသေ ၶိုၼ်းတၢင်ႇဝႆႉတီႈ Huggingface ဢၼ်ပဵၼ်တီႈၸူႉတုမ်ၸုမ်ႇၶေႃႈမုၼ်း (datasets) လႄႈမေႃႇတႄႇလ် AI ဢၼ်ၸႂ်ႉၵၼ်ၼမ်တင်းလုမ်ႈၾႃႉယၢမ်းမိူဝ်ႈလဵဝ် ပိူင်ၼိုင်ႈၵေႃႈ ၸုမ်ႇၶေႃႈမုၼ်ၸိူဝ်းၼႆႉတေၸၼ်ဢဝ်ၸႂ်ႉငၢႆႈလိူဝ်သေဝႅပ်ႉသၢႆႉသ်။

![Load Dataset {caption: ၵၢၼ်ၸၼ်ၸႂ်ႉၶေႃႈမုၼ်းၼႂ်း huggingface}](blog/assets/ocr-lab-for-shan-language/load_datasets_1.png)

ၼႂ်းၵၢၼ်ၸၼ်ၶေႃႈမုၼ်းတင်းၼိူဝ်ၼႆႉတေႁၼ်ဝႃႈ မၢင်တီႈမီးတူဝ်လိူဝ်၊ emoji လႄႈတူဝ်ဢၼ်ဢမ်ႇၸႂ်ႈလိၵ်ႈတႆးဝႆႉယူႇ ၸိူဝ်းၼၼ်ႉၼႆႁဝ်းတေလႆႈ clean သုၵ်ႈလၢင်ႉပႅတ်ႈတင်းသဵင်ႈ

မိူၼ်ၼင်ႇ "ထိုင်ႉႉႉႉႉႉႉႉႉႉႉႉႉႉႉ", "================", "📃📃" ၸိူဝ်းၼႆႉ။

လိူဝ်သေၼၼ်ႉ ၼႂ်းၶေႃႈမုၼ်းလိၵ်ႈႁဝ်းယင်းမီးပႃးလိၵ်ႈတၢင်ႇမဵဝ်း မိူၼ်ၼင်ႇ တူဝ်လိၵ်ႈဢိင်းၵရဵတ်ႈ၊ တူဝ်လိၵ်ႈမၢၼ်ႈ၊ တူဝ်ၼပ်ႉ ၸိူဝ်းၼႆႉ ပေႃးႁႂ်ႈလီတႄႉ ႁႂ်ႈသုၵ်ႈလၢင်ႉပႅတ်ႈသဵင်ႈသဵင်ႈ။

```python
# ဢဝ်ဢွၵ်ႇပႅတ်ႈ emoji
def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", re.UNICODE)
    return re.sub(emoj, '', data)
```

```python
# ဢဝ်ဢွၵ်ႇပႅတ်ႈတူဝ်လိၵ်ႈဢၼ်ဢမ်ႇယူႇၼႂ်း Myanmar Unicode Block \U1000 - \U109F
def remove_latin_text(text):
    text = re.sub(r"[^\u1000-\u109f\s]", '', text)
    text = re.sub(r"\s+", " ", text)
    return text
```

ၵမ်းၼႆႉ တႃႇတေၸႅၵ်ႇဢွၵ်ႇတူဝ်လိၵ်ႈမၢၼ်ႈၼၼ်ႉ ၵွပ်ႈဝႃႈတူဝ်လိၵ်ႈတႆးလႄႈတူဝ်လိၵ်ႈမၢၼ်ႈ မၢင်တူဝ်လႆႈၸႂ်ႉတူဝ်ႁူမ်ႈၵၼ်ဝႆႉ

```Markdown
 "င", "တ", "ထ", "ပ", "မ", "ယ", "ရ", "လ", "ဝ", "သ", "ိ", "ီ", "ု", "ူ", "ေ", "ဵ", "း", "်", "ျ", "ြ", "ွ", "၊", "။"
```

ၵွပ်ႈၼႆလႄႈပဵၼ်ဢၼ်ၸႅၵ်ႇတူဝ်လိၵ်ႈတႆးလႄႈတူဝ်လိၵ်ႈမၢၼ်ႈယၢပ်ႇဝႆႉ ၸႂ်ႉ regex သေၸၼ်ဢွၵ်ႇၵေႃႈတေလႆႈတႅမ်ႈပိူင် rules မၼ်းၼမ်ဝႆႉ။

လႄႈၸၢမ်းဝူၼ်ႉတူၺ်းလၢႆးငၢႆႈမၼ်း။

> ၼႂ်းၶေႃႈလိၵ်ႈမၢၼ်ႈၼၼ်ႉ ၵမ်ႈၼမ်ၸႂ်ႉတူဝ်ဢၼ်ႁူမ်ႈၵၼ်တင်းတႆးသေတႃႉ ၵူၺ်းၼႂ်းၼိုင်ႈၶေႃႈၸၢင်ႈမီးပႃးတူဝ်ဢၼ်ၼႂ်းလိၵ်ႈတႆးဢမ်ႇလႆႈၸႂ်ႉ

ၵွပ်ႈၼၼ်လႄႈ ပေႃးႁဝ်းထႅၵ်ႇဢွၵ်ႇပဵၼ်ၶေႃႈသေတူၺ်းတူဝ်မႄႈတင်းမူတ်း ပေႃႈမီးတီႈဢေႇသုတ်းဢမ်ႇၸႂ်ႈတူဝ်လိၵ်ႈတႆးၼႆ ႁဝ်းတေဢဝ်ဢွၵ်ႇပႅတ်ႈတူဝ်ၼၼ်ႉ

```python
# တႃႇထႅၵ်ႇလိၵ်ႈဢွၵ်ႇပဵၼ်ၶေႃႈၼၼ်ႉ ၸႂ်ႉ ShanNLP project
from shannlp import word_tokenize

def remove_myanmar_text(text):
    tokens = word_tokenize(text, engine="newmm")
    cleaned_words = []
    for word in tokens:
        is_shan_word = True
        for char in word:
            if char not in shan_characters and not char.isspace():
                is_shan_word = False
                break
        if is_shan_word:
            cleaned_words.append(word)
    
    cleaned_text = "".join(cleaned_words)
    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

text = "ဝၢၼ်ႈယေႇပူႇၵေႃႉၸႅပ်ႉ (ရေပူကကော့စပ်ရွာ)"

print(word_tokenize(text, engine="newmm"))
# -> ['ဝၢၼ်ႈ', 'ယေႇ', 'ပူႇ', 'ၵေႃႉ', 'ၸႅပ်ႉ', ' ', '(ရေပူကကော့စပ်ရွာ)']

print(remove_myanmar_text(text))
# -> ဝၢၼ်ႈယေႇပူႇၵေႃႉၸႅပ်ႉ
```

ၵူၺ်းလၢႆႈၼႆႉဢမ်ႇပႆႇၸႂ်ႈလွၵ်းလၢႆးဢၼ်လီတီႈသုတ်း မိူၼ်ၼင်ႇ

```python
text = "ဝၢၼ်ႈယေႇပူႇၵေႃႉၸႅပ်ႉ ရေပူကကော့စပ်ရွာ"

print(word_tokenize(text, engine="newmm"))
# ['ဝၢၼ်ႈ', 'ယေႇ', 'ပူႇ', 'ၵေႃႉ', 'ၸႅပ်ႉ', ' ', 'ရေ', 'ပူ', 'ကကော့စပ်ရွာ']

print(remove_myanmar_text(text))
# ဝၢၼ်ႈယေႇပူႇၵေႃႉၸႅပ်ႉ ရေပူ -> ၵိုတ်းမၢင်တူဝ်ဢၼ်ပဵၼ်တူဝ်လိၵ်ႈတႆးတင်းမူတ်း
```

ၸိူင်ႉႁိုဝ်ၵေႃႈ တွၼ်ႈတႃႇ OCR project တႄႉၵေႃႈၸႂ်ႉလႆႈယူႇ

ၼႂ်းၸႃႉၶေႃႈမုၼ်းလိၵ်ႈတႆးၼႂ်းဝႅပ်ႉသၢႆႉသ်ၸိူဝ်းၼႆႉ တေႁၼ်ဝႃႈ **ၼိုင်ႈဝႅပ်ႉသၢႆႉသ်မီးၵၢၼ်တႅမ်ႈ typing ၽႂ်မၼ်းဢမ်ႇမိူၼ်ၵၼ် ပႅၵ်ႇပိူင်ႈၵၼ်ဢိတ်းဢွတ်း ၸဵမ်လွၵ်းမိုဝ်း လၢႆးပေႃႉလိၵ်ႈ လႄႈ encoding ၸိူဝ်းၼႆႉ ႁၼ်ဝႆႉဝႃႈမီးတီႈဢၼ်ဢမ်ႇမိူၼ်ၵၼ်ဢိတ်းဢွတ်း**

```python
# သုၵ်ႈလၢင်ႈလိၵ်ႈတႆးထႅင်ႈ
text = text.replace("၊", "၊ ").replace("။", "။ ").replace(" ၊", "၊ ").replace(" ။", "။ ").strip()
text = re.sub(r"ႉ{2,}", "ႉ", text)
text = text.replace("ႆၢ", "ၢႆ")
text = text.replace("ေတ", "တေ")
text = text.replace("ꩡ", "ၸ")
```

> - ၼႂ်း code တၢင်းၼိူဝ်ၼႆႉလႆႈလႅၵ်ႈလၢႆႈတူဝ်လိၵ်ႈတႆးဝႆႉမၢင်တူဝ် မိူၼ်ၼင်ႇလၢႆးၸႂ်ႉ ၊, ။ ၸိူဝ်းၼႆႉ မၢင်တီႈလႆႈႁၼ်ဝႆႉဝႃႈဢမ်ႇလႆႈလိူဝ်ႁွင်ႈပဝ်ႇဝၢႆးသေၸႂ်ႉ ၊, ။
> - မၢင်တီႈသႂ်ႇၸမ်ႈတႂ်ႈလိူဝ်ဝႆႉ (ဝူၼ်ႉဝႃႈၸႂ်ႉတၢင် full-stop ၼႂ်းၽႃႇသႃႇဢိင်းၵရဵတ်ႈ)
> - မၢင်ဝႅပ်ႉသၢႆႉသ် တႅမ်ႈ ၵႆၢၶိုၼ်း ဢွၼ်တၢင်း ဢႃပွတ်း (ၵၢႆ, ၵႆၢ -> တေလႆႈတႅမ်ႈ ၵ - ၢ - ႆ)
> - မၢင်ဝႅပ်ႉသၢႆႉသ် တႅမ်ႈ ေဢသႆႇ ဢွၼ်တၢင်း (ေဢ, ဢေ -> တေလႆႈတႅမ်ႈ ဢ - ေ)
> - မၢင်ဝႅပ်ႉသၢႆႉသ် ၸႂ်ႉ "ꩡ" တႅၼ်းတီႈ "ၸ" ၼႂ်း unicode ဢမ်ႇၸႂ်ႈဢၼ်လဵဝ်ၵၼ် "ꩡ" (U+AA61) Myanmar Letter Khamti Ca, "ၸ" (U+1078) Myanmar Letter Shan Ca

[dataset generate & cleanning code](https://github.com/NoerNova/tesstrain/blob/main/shan-datasets/generate_shn_datasets.ipynb)

### Generating Synthetic Data

ဝၢႆးသေၸၼ်ၶေႃႈမုၼ်းသေ သုၵ်ႈလၢင်ႉၶေႃႈမုၼ်းယဝ်ႉ ႁဝ်းတေမႃးႁဵတ်းဢွၵ်ႇ **Synthetic** ၶေႃႈမုၼ်း။
တွၼ်ႈၼႆႉလႆႈၸၢမ်းဝႆႉယူႇသွင်လွၵ်းလၢႆး

1. [Python pillow package](https://pypi.org/project/pillow/) ပဵၼ် library ၸတ်းၵၢၼ်ၶႅပ်းႁၢင်ႈလူၺ်ႈ python
2. [Tesseract text2image](https://github.com/tesseract-ocr/tesseract/blob/main/src/training/text2image.cpp) ပဵၼ် library ဢၼ်ၵိုၵ်းမႃးၸွမ်း tesseract-ocr

တီႈၼႆႉလႆႈႁၼ်မႅၼ်ႈဝႃႈၶေႃႈမုၼ်းဢၼ်လႆႈမႃးၼၼ်ႉ ဢဝ်ၵႂႃႇၸႂ်ႉယဝ်ႉလႆႈၽွၼ်းပၢင်ႈဢမ်ႇမိူၼ်ၵၼ် ဝၢႆးသေ Trained ယဝ်ႉၶေႃႈမုၼ်းဢၼ်ၼိုင်ႈ လႆႈမေႃႇတႄႇလ်ဢၼ်ၸႂ်ႉလႆႈလီတီႈၼိုင်ႈ ၵူၺ်းဢမ်ႇလီတီႈၼိုင်ႈ ၼင်ႇႁူဝ်ႁုပ်ႈဢၼ်တေၼႄပႃႈတႂ်ႈ။

**Generate datasets with pillow** -> [https://github.com/NoerNova/tesstrain/blob/main/shan-datasets/generate_shn_datasets.ipynb](https://github.com/NoerNova/tesstrain/blob/main/shan-datasets/generate_shn_datasets.ipynb)

ၼႂ်း code ၼႆႉမီးဝႆႉ data source သွင်ဢၼ် -> From docx ဢၼ်တေၸၼ်ၶေႃႈမုၼ်းတီႈ .docx words files လႄႈ From huggingface တီႈၼႆႈတေၸႂ်ႉ huggingface ပဵၼ်လၵ်း

**Generate datasets with text2image** -> [https://github.com/NoerNova/tesstrain/blob/main/shan-datasets/dataset_labs.ipynb](https://github.com/NoerNova/tesstrain/blob/main/shan-datasets/dataset_labs.ipynb)

**All Datasets generate resource** -> [https://github.com/NoerNova/tesstrain/tree/main/shan-datasets](https://github.com/NoerNova/tesstrain/tree/main/shan-datasets)

ၸုမ်ႇၶေႃႈမုၼ်း datasets ဢၼ် generated ယဝ်ႉၼၼ်ႉႁႂ်ႈမီးဝႆႉၼႂ်းၾူဝ်ႇတိူဝ်ႇ ``data/`` မိူၼ်ၼင်ႇ ``data/shn-ground-truth`` ***ၸိုဝ်ႈတေလႆႈပဵၼ်ဢၼ်လဵဝ်ၵၼ်တင်းၸိုဝ်ႈမေႃႇတႄႇလ် \*-ground-truth***

## Step 2: Training Tesseract for Shan

ဝၢႆးသေလႆႈၶေႃႈမုၼ်းမႃးယဝ်ႉ ၸမ်ထိုင်မႃးၶၵ်ႉတွၼ်ႈၵၢၼ် Train.

### Setting Up Tesseract Training Environment

ဢွၼ်တၢင်းသုတ်းတေလႆႈ config environment တႃႇႁႂ်ႈၸႂ်ႉ tesseract-ocr လႆႈၼႂ်းၶိူင်ႈၶွမ်းႁဝ်း ဢိၵ်ႇပႃးတင်းၶိူင်ႈမိုဝ်းတွၼ်ႈတႃႇ train ၸိူဝ်းၼႆႉတင်းသဵင်ႈ။

ၵွပ်ႈၼၼ်ႁဝ်းတေလႆႈလူင်း Tesseract-OCR ၼိုင်ႈဢၼ် လႄႈ Tesseract-OCR/Tesstrain ထႅင်ႈၼိုင်ႈဢၼ်

#### Tesseract-OCR install

တူၺ်း Tesserat-OCR official document -> [https://github.com/tesseract-ocr/tesseract#installing-tesseract](https://github.com/tesseract-ocr/tesseract#installing-tesseract)

တွၼ်ႈၼႆႉႁႂ်ႈလူင်း install လူၺ်ႈ [build it from source](https://tesseract-ocr.github.io/tessdoc/Compiling.html) ၵွပ်ႈႁဝ်းတေလႆႈမေးၵႄႈၾၢႆႇလ်ဢၼ်ၼိုင် ``src/training/unicharset/validate_myanmar.cpp`` ဢၼ်ၼႂ်း original tesseract-ocr မီးဝႆႉ missing character တႃႇလိၵ်ႈတႆး။

***crd:*** [MyOCR - SengKyaut](https://github.com/sengkyaut/MyOCR/blob/main/prepare_traindata/myunicharset.py)

ဝၢႆးသေၵႄႈၾၢႆႇလ်ၼၼ်ႉယဝ်ႉၵေႃႈ build သေလူင်း install ၸွမ်းၼင်ႇ platform ဢၼ်ႁဝ်းၸႂ်ႉ

- [Linux](https://tesseract-ocr.github.io/tessdoc/Compiling.html#linux)
- [Windows](https://tesseract-ocr.github.io/tessdoc/Compiling.html#windows)
- [macOS](https://tesseract-ocr.github.io/tessdoc/Compiling.html#macos)

***Linux လႄႈ macOS တေ build ငၢႆႈလိူဝ်သေ Windows (ၸႂ်ႉ WSL ၸၢင်ႈငၢႆႈလိူဝ်)***

#### Tesseract-OCR/Tesstrain

ဝၢႆးသေ install Tesseract-OCR ယဝ်ႉ တႃႇတေ train လႆႈၼၼ်ႉ တေလႆႈလူင်း install ထႅင်ႈ [tesseract-ocr/tesstrain](https://github.com/tesseract-ocr/tesstrain) ဢၼ်ပဵၼ် Training workflow for Tesseract 5 ႁႂ်ႈ train လႆႈငၢႆႈလိူဝ်သေၸႂ်ႉၶိူင်ႈမိုဝ်းၼႂ်း Tesseract-OCR မဵဝ်းလဵဝ်။

တူၺ်း Tesseract-OCR/Tesstrain official document -> [https://github.com/tesseract-ocr/tesstrain#installation](https://github.com/tesseract-ocr/tesstrain#installation)

ပေႃးဝႃႈ Install Tesseract-OCR ယဝ်ႉ ၶၢမ်ႈၵႂႃႇတွၼ်ႈ [https://github.com/tesseract-ocr/tesstrain#python](https://github.com/tesseract-ocr/tesstrain#python)

```bash
git clone https://github.com/tesseract-ocr/tesstrain.git
cd tesstrain
pip install -r requirements.txt
make tesseract-langdata
```

ဢမ်ႇၼၼ်ၸၼ်ဢဝ် repo ဢၼ်ၼႆႉ

```bash
git clone https://github.com/NoerNova/tesstrain.git
cd tesstrain
pip install -r requirements.txt # dependency တႃႇ tesstrain
cd shan-datasets
pip install -r requirements.txt # dependency တႃႇ shan-datasets
```

### Training

Run training script

**Official** -> [https://github.com/tesseract-ocr/tesstrain#train](https://github.com/tesseract-ocr/tesstrain#train)

```bash
#!/bin/bash

# Define paths
MODEL_NAME="shn"
LANG_CODE="shn"
TESSTRAIN_REPO="$HOME/Labs/tesstrain"
DATA_DIR="$TESSTRAIN_REPO/data"
TESSDATA_PREFIX="/usr/local/share/tessdata"
WORDLIST_FILE="$TESSTRAIN_REPO/data/shn.wordlist.txt"
NUMBERS_FILE="$TESSTRAIN_REPO/data/shn.numbers.txt"
PUNC_FILE="$TESSTRAIN_REPO/data/shn.punc.txt"
MAX_ITERATIONS=200000
LEARNING_RATE=0.0005

# Create the training directory
rm -rf "$DATA_DIR/$MODEL_NAME" # remove old trained files/dir
rm -r "$DATA_DIR/$MODEL_NAME.traineddata" # remove old traineddata
mkdir -p "$DATA_DIR/$MODEL_NAME"

# Generate training data
cd $TESSTRAIN_REPO
make training \
    MODEL_NAME=$MODEL_NAME \
    LANG_CODE=$LANG_CODE \
    TESSDATA=$TESSDATA_PREFIX \
    WORDLIST_FILE=$WORDLIST_FILE \
    PUNC_FILE=$PUNC_FILE \
    NET_SPEC="[1,36,0,1 Ct3,3,32 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx512 O1c###]" \
    MAX_ITERATIONS=$MAX_ITERATIONS \
    LEARNING_RATE=$LEARNING_RATE
```

ၶၵ်ႉတွၼ်ႈ training ၼႆႉ ပေႃးၸႂ်ႉၸုမ်ႇၶေႃႈမုၼ်းဢၼ်ၸၼ်လူၺ်ႈ pillow ၼၼ်ႉပႆႇလႆႈၸၼ် .box လႄႈ ၶၵ်ႉတွၼ်ႈၼႆႉတေ generate .box လႄႈ .lstmf ထႅင်ႈၼႂ်း *-ground-truth ၸႂ်ႉၶၢဝ်းယၢမ်းႁိုင်တၢၼ်ႇတၢၼ်ႇ ဢိင်ၼိူဝ်ႁူဝ်ၼပ်ႉၶေႃႈမုၼ်းဢၼ်ႁဝ်းၸၼ်ဝႆႉ။
ပေႃးၸႂ်ႉၶေႃႈမုၼ်းဢၼ်ၸၼ်လူၺ်ႈ text2image မီးဝႆႉ .box ယဝ်ႉလႄႈ တေ generate ထႅင်ႈ .lstmf ၵူၺ်း တေၽႆးလိူဝ်။

လိူဝ်သေၼၼ်ႉတေ generate unicharset ၾၢႆႇလ် ဢၼ်မီးၶေႃႈမုၼ်းတူဝ်လိၵ်ႈတင်းမူတ်းဢၼ်မီးၼႂ်းၸုမ်ႇၶေႃႈမုၼ်းႁဝ်းၼၼ်ႉ

ပေႃးၼႂ်း training log ဢွၵ်ႇမႃးၼင်ပႃႈတႂ်ႈၼႆႉၼမ်

```bash
Encoding of string failed! Failure bytes: e1 81 b7 e1 80 ad e1 80 b0 e1 80 9d e1 80 ba e1 82 8a e1 81 8b
Can't encode transcription: 'ၼပ်ႉပဵၼ် ႁူဝ်ႁူဝ်လၢၼ်ႉဢေႊၷိူဝ်ႊ။' in language ''
```

ပွင်ႇဝႃႈၼႂ်း unicharset ၼၼ်ႉဢမ်ႇၸၼ်လႆႈတူဝ်လိၵ်ႈမၢင်တူဝ်၊ တီႈၼႆႈတေသႂ်ႇဢဝ်ႁင်းၵူၺ်းၵေႃႈလႆႈ၊ မူၼ်ၼင်ႇၽၢႆႇၼိူဝ်ၼႆႉ ၼႂ်း unicharset ၾၢႆႇလ် ဢမ်ႇမီးတူဝ် "ၷ" ၼႆ

```bash
48 # လၢႆႈပဵၼ် 49, +1 ၸွမ်ႁူဝ်ၼပ်ႉတူဝ်လိၵ်ႈဢၼ်သႂ်ႇမႃး
NULL 0 NULL 0
Joined 0 0,255,0,255,486,1218,0,30,486,1188 NULL 0 0 0 # Joined [4a 6f 69 6e 65 64 ]
|Broken|0|1 0 0,255,0,255,892,2138,0,80,892,2058 NULL 0 0 0 # Broken
ယ 1 0,255,0,255,336,359,20,21,378,402 Myanmar 3 0 3 ယ # ယ [101a ]x
ွ 0 0,255,0,255,0,0,0,0,0,0 Myanmar 4 17 4 ွ # ွ [103d ]
ၼ 1 0,255,0,255,354,355,20,21,395,402 Myanmar 5 0 5 ၼ # ၼ [107c ]x
် 0 0,255,0,255,0,0,0,0,0,0 Myanmar 6 17 6 ် # ် [103a ]
ႉ 0 0,255,0,255,62,86,20,21,106,128 Myanmar 7 0 7 ႉ # ႉ [1089 ]
...
ၷ 1 0,255,0,255,0,0,0,0,0,0 Myanmar 48 0 48 ၷ # ၷ [1077 ]x --> သႂ်ႇဝႆႉတင်းတႂ်ႈသုတ်း
# Format: character [consonant/1 vowel/0 numbers/8 punc/10] 0,255,0,255,0,0,0,0,0,0 Lang...
```

ပေႃး trained ယဝ်ႉတူဝ်ႈတေလႆႈ result ၸိူင်ႉပႃႈတႂ်ႈၼႆႉ

```bash
...
2 Percent improvement time=23727, best error was 4.629 @ 71700
At iteration 95427/192700/192702, mean rms=0.807%, delta=0.538%, BCER train=2.563%, BWER train=22.000%, skip ratio=0.000%, New best BCER = 2.563 wrote checkpoint.
...
At iteration 96640/199700/199702, mean rms=0.839%, delta=0.545%, BCER train=3.105%, BWER train=29.510%, skip ratio=0.000%, wrote checkpoint.
At iteration 96650/199800/199802, mean rms=0.823%, delta=0.545%, BCER train=3.109%, BWER train=28.702%, skip ratio=0.000%, wrote checkpoint.
At iteration 96661/199900/199902, mean rms=0.806%, delta=0.508%, BCER train=3.073%, BWER train=28.593%, skip ratio=0.000%, wrote checkpoint.
At iteration 96679/200000/200002, mean rms=0.809%, delta=0.514%, BCER train=3.109%, BWER train=28.927%, skip ratio=0.000%, wrote checkpoint.
Finished! Selected model with minimal training error rate (BCER) = 2.563
```

Model ဢၼ် trained ယဝ်ႉၽၢႆႇၼိူဝ်ၼႆႉ ၸႂ်ႉတိုဝ်းၶေႃႈမုၼ်းဢၼ်ၸၼ်လူၺ်ႈ pillow

```bash
base: version 3
finetune: version 1
dataset: tainovel.com, shannews.org medium {50000, 50000} https://huggingface.co/datasets/NorHsangPha/shan-novel-tainovel_com, https://huggingface.co/datasets/NorHsangPha/shan-news-shannews_org
MAX_ITERATIONS=200000
LEARNING_RATE=0.0001
NET_SPEC=[1,36,0,1 Ct3,3,32 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx512 O1c###]
```

လႆႈ best result တီႈ iteration: 192702 (BCER) = 2.563% (BWER) = 22.000%
ႁဝ်းတေ Focus တီႈ Best character error rate (BCER) လႄႈ Best words error rate (BWER) ႁႂ်ႈမီးတူဝ်ၼပ်ႉ % ဢႄႇဝႆႉၵွၼ်ႇ။

## Step 3: Addressing Challenges

- **Font Limitations**: ၶေႃႈမုၼ်းၵၢၼ်ၸႂ်ႉ Fonts ၵေႃႈလမ်ႇလွင်ႈတွၼ်ႈတႃႇ OCR မိူၼ်ၵၼ် ၼင်ႇႁိုဝ်မၼ်းတေႁူႉၸၵ်းလႄႈၸႂ်ႉလႆႈၸွမ်းၾွၼ်ႉတင်းၼမ်ၼၼ်ႉ ၼႂ်းၸုမ်ႇၶေႃႈမုၼ်းလီမီး Fonts ၼမ်ၼမ် မိူဝ်လဵဝ်ၸႂ်ႉဝႆႉၾွၼ်ႉ
  - GreatHorKham_Taunggyi
  - MyanmarText
  - PangLong
  - Pyidaungsu
  - Shan
- **Lack of Hand-Writting Datasets**: ယင်းပႆႇမီးၶေႃႈမုၼ်းလၢႆမိုဝ်းတႅမ်ႈ ဢၼ်ၵုမ်ႇထူၼ်ႈတႃႇၾိုၵ်းသွၼ်

- **Lack of Benchmark Datasets**: တႃႇတေၸၢမ်းတႅၵ်ႈလွင်ႈၶိုၵ်ႉၶႅမ်ႉ OCR ၼၼ်ႉပႆႇမီးၶေႃႈမုၼ်းဢၼ်တေ Benchmark မၼ်း (ၸုမ်ႇၶေႃႈမုၼ်းဢၼ်လႆႈၸႂ်ၵၼ်ဝႆႉ လႄႈၶဝ်ႈပိူင်လုမ်ႈၾႃႉဝႃႈၸႂ်ႉတႃႇတႅၵ်ႈၶေႃႈတွပ်ႇ OCR)

## Results and Future Plans

ဝၢႆးသေ trained ယဝ်ႉ တေလႆႈမေႃႇတႄႇလ် shn.traineddata ၾၢႆႇလ်

Copy to /usr/local/share/tessdata or /usr/share/tessdata

```bash
sudo cp data/shn.traineddata /usr/local/share/tessdata
```

ဢမ်ႇၼၼ်ၵေႃႈ ``set TESSDATA_PREFIX={path_to_tessdata}, TESSDATA={path_to_tessdata}``

Tesseract တေလႆႈႁၼ်မေႃႇတႄႇလ်ဢၼ်ႁဝ်း trained ယဝ်ႉ

### ၸၢမ်းၸႂ်ႉတူၺ်းတင်းၶႅပ်းႁၢင်ႈ

```python
import pytesseract
from PIL import Image

oem_psm_config = r'--oem 1 --psm 13'

image_path = "data/testimage2.tif"
image = Image.open(image_path)

extracted_text = pytesseract.image_to_string(image, lang="shn", config=oem_psm_config)  # Assuming the text is in Burmese

display(image)
print(extracted_text)
```

![Example 1 {caption: ၶႅပ်ႈႁၢင်ႈ generated single line}](blog/assets/ocr-lab-for-shan-language/example_re_1.png)
![Example 2 {caption: ၶႅပ်ႈႁၢင်ႈ screenshot single line}](blog/assets/ocr-lab-for-shan-language/example_re_2.png)
![Example 3 {caption: ၶႅပ်ႈႁၢင်ႈ screenshot single line}](blog/assets/ocr-lab-for-shan-language/example_re_3.png)

ၽၢႆႇၼိူဝ်ၼႆႉၸၢမ်းတူၺ်းတင်းၶႅပ်းႁၢင်ႈဢၼ်ပဵၼ် Single Line text ထႅဝ်လဵဝ် ႁၼ်ဝႃႈယင်းမီး missing character ၵမ်ႈဢွင်ႈ

![Example 4 {caption: ၶႅပ်ႈႁၢင်ႈ screenshot multi line}](blog/assets/ocr-lab-for-shan-language/example_re_4.png)

ၽၢႆႇၼိူဝ်ၼႆႉၸၢမ်းတူၺ်းတင်းၶႅပ်းႁၢင်ႈဢၼ်ပဵၼ် Multi Line text လိၵ်ႈလၢႆထႅဝ် ႁၼ်ဝႃႈမီး missing character တင်းၼမ် ပိူင်ၼိုင်ႈၵေႃႈ
ၸုမ်ႇၶေႃႈမုၼ်းဢၼ်ႁဝ်းၸႂ်ႉၾိုၵ်းသွၼ်ၼၼ်ႉပဵၼ်ဝႆႉ Single Line text တင်းမူတ်းလႄႈ တေလႆႈမီး ၸုမ်ႇၶေႃႈမုၼ်း Multi Line text မႃးပွၼ်ႈသွၼ်ပၼ်ထႅင်ႈ။

### ၸၢမ်းၸႂ်ႉတူၺ်းတင်း PDF

ယိူင်းမၢႆဢၼ်ၼႆႉၵေႃႈ ပိူဝ်ႈတႃႇတေၸၼ်ၶေႃႈမုၼ်းလိၵ်ႈၼႂ်း PDF လႆႈၼႆလႄႈ ၸၢမ်းတူၺ်းတင်း PDF file
တီႈၼႆႉႁဝ်းတေၸႂ်ႉ [ocrmypdf](https://github.com/ocrmypdf/OCRmyPDF) adds on

![Book Example 1 {caption: ပပ်ႉလိၵ်ႈဢၼ်ပဵၼ်ၶႅပ်းႁၢင်ႈ PDF}](blog/assets/ocr-lab-for-shan-language/book_example1.png)

![Book OCR Example {caption: တႅၵ်ႉၼိူင်းၵၼ်တင်းမိူဝ်ႈပႆႇႁဵတ်း OCR သေ copy လႄႈ မိူဝ်းႁဵတ်း OCR}](blog/assets/ocr-lab-for-shan-language/book_example2.png)

![Book Example 2 {caption: ပပ်ႉလိၵ်ႈဢၼ်ပဵၼ်ၶႅပ်းႁၢင်ႈ PDF}](blog/assets/ocr-lab-for-shan-language/book_example3.png)

![Book OCR Example 2 {caption: တႅၵ်ႉၼိူင်းၵၼ်တင်းမိူဝ်ႈပႆႇႁဵတ်း OCR သေ copy လႄႈ မိူဝ်းႁဵတ်း OCR}](blog/assets/ocr-lab-for-shan-language/book_example4.png)

တေႁၼ်ဝႃႈဢွၵ်ႇလိၵ်ႈတႆးမႃးယူႇ ၵူၺ်းယင်းတိုၵ်မီး missing words တင်းၼမ် တီႈၼႆႈၵေႃႈတေလႆႈမီး ၸုမ်ႇၶေႃႈမုၼ်း Page level text မႃးပွၼ်ႈသွၼ်ပၼ်ထႅင်ႈ။

## Conclusion

ၵၢၼ်လဵပ်ႈႁဵၼ်း လိၵ်ႈတႆးလႄႈထႅၵ်ႉၶၼေႃႇလေႃႇၸီႇ လႆႈႁၼ်ၶေႃႈယူပ်ႈယွမ်းလိၵ်ႈတႆးႁဝ်း လႄႈဢၼ်လိၵ်ႈတႆးႁဝ်းလူဝ်ႇမီးထႅင်ႈတင်းၼမ် ပၼ်ႁႃလူင်တႃႇလိၵ်ႈတႆးႁဝ်းဝၼ်းမိူဝ်ႈၼႆႉ ၼႂ်းၵၢပ်ႈပၢၼ် AI ၸိူင်ႉၼႆ
ၶေႃႈမုၼ်းဢၼ်ပဵၼ်တီႇၵျိတ်ႇတႄႇ လႄႈၽွမ်ႉၸႂ်ႉၼၼ်ႉ မီးႁႅင်းဢႄႇတႄႉတႄႉ ဢိၵ်ႇလူၺ်ႈၵၢၼ်ၸႂ်ႉတိုဝ်းလိၵ်ႈတႆးဢမ်ႇမိူၼ်ၵၼ်ၵေႃႈမီးထႅင်ႈတင်းၼမ် ပႆႇမီး standard ဢမ်ႇႁူႉတေလႆႈဢၼ်ဢၼ်လႂ်ဝႃႈ ႁဵတ်းၵႂႃႇ ႁဵၼ်းႁူႉၵႂႃႇႁင်းၽႂ်ႁင်းမၼ်း
ယူႇၸိူင်ႉၼႆ လိူဝ်သေဢဝ်မႃးသိုပ်ႇၶႆႈလဝ်ႈၼႄၵၼ်ၸိူင်ႉၼႆၵေႃႈ ဢမ်ႇၸၢင်ႈႁဵတ်းႁိုဝ်ထႅင်ႈ။

မိူၼ်ၵၢၼ်လဵပ်ႈႁဵတ်း project ၸိူဝ်းၼႆႉ မိူဝ်ႈလဵဝ်ၶၢတ်ႇဝႆႉၶေႃႈမုၼ်းမၼ်း ၼႆယုမ်ႇယမ်ဝႃႈ မိူဝ်းၼႃႈမႃးတေမီးၶေႃႈမုၼ်း လႄႈၵၢၼ်လဵပ်ႈႁဵၼ်းမႃးသိုပ်ႇတေႃႇၵႂႃႇထႅင်ႈယူႇ

[OCR](https://www.noernova.com/blog/ocr-lab-for-shan-language),
[ShanNLP](https://github.com/NoerNova/ShanNLP),
[TTS](https://www.noernova.com/note/vits_tts_mms_shan_finetune),
[ASR](https://www.noernova.com/note/asr_mms_adapter_finetune_for_shan_language),
[Text Generation](https://www.noernova.com/blog/fine-tuning-gpt2-for-shan-language),
[LLM](https://www.noernova.com/blog/fine-tuning-llama3-for-shan-language)

[Project Repo](https://github.com/NoerNova/tesstrain)
