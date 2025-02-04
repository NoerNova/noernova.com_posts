---
tags: ["computer", "shan-language", "shan", "deep-learning", "machine-translations"]
date: "September 8, 2023"
title: "ၶိူင်ႈပိၼ်ႇၽႃႇသႃႇတႆး ၶွင် Facebook ပဵၼ်သင် ၶဝ်ႁဵတ်းႁိုဝ်ႁဵတ်း tech recap"
subtitle: "ၶိူင်ႈပိၼ်ႇၽႃႇသႃႇတႆး ၶွင် Facebook လႄႈၶူင်းၵၢၼ် NLLB ႁူဝ်ယွႆႈလွင်ႈ tech"
image: "https://raw.githubusercontent.com/NoerNova/noernova.com_blog/main/assets/fairseq-NLLB-machine-translations-analize/nllb-translation-demo.jpeg"
link: "blog/fairseq-NLLB-machine-translations-analize"
description: "ၶိူင်ႈပိၼ်ႇၽႃႇသႃႇတႆး ၶွင် Facebook လႄႈၶူင်းၵၢၼ် NLLB ႁူဝ်ယွႆႈလွင်ႈ tech"
---

### Facebook NLLB

Facebook (Meta) NLLB ပဵၼ်ၶူင်းၵၢၼ် research ၶွင် Facebook ၸိုဝ်ႈမႂ်ႇ Meta ၶဝ် သိုပ်ႇၶူင်းၵၢၼ် research ဢၼ်ႁွင်ႉၸိုဝ်ႈဝႃႈ fairseq (Facebook AI Research) ဢၼ်တႄႇၶူင်းၵၢၼ်မႃးၸဵမ်မိူဝ်ႈပီ 2017၊ fairseq ပဵၼ်ၶူင်းၵၢၼ် research လွင်ႈ ai လၢႆလၢႆႁူဝ်ၶေႃႈဢၼ်ၵဵဝ်ႇလူၺ်ႈလွင်ႈၽႃႇသႃႇလိၵ်ႈလၢႆး လႄႈ NLP (Natural Language Processing) မိူၼ်ၼင်ႇ fasttext, Text-to-Speech, Speech-to-Text, ၸိူဝ်းၼႆႉ [ၸုမ်းႁဝ်းႁႃး လႆႈၸၼ်ႁဵတ်း website မႃးပၼ်တႃႇၸၢမ်းတူၺ်း TTS](https://www.facebook.com/official.haohaa/posts/pfbid031aQyiSYWYzcmBTAWbdTAHdZiwBWrZqZNJFSw2Kzdw1KbPRupHfsd7FUYRMAtKR2Ql) လိၵ်ႈတႆးၵႂၢမ်းတႆးၼၼ်ႉၵေႃႈ လုၵ်ႉတီႈၶူင်းၵၢၼ် fairseq ၼႆႉမႃးယဝ်ႉၶႃႈ။

ဝၢႆးသေၶူင်းၵၢၼ် fairseq, fasttext တေႇၸႂ်ႉၵၢၼ်လႆႈမႃးၼၼ်ႉ ပီ 2017 ၼၼ်ႉၼင်ႇၵဝ်ႇ လွၵ်းလၢႆး AI/Deep Learning ဢၼ်မႂ်ႇ ဢၼ်ႁွင်ႉဝႃႈ [Transformer](https://arxiv.org/abs/1706.03762) ၼႆဢွၵ်ႇမႃး လႄႈၸႂ်ႉၵၢၼ်ၼႂ်း ၶၵ်ႉၵၢၼ် Machine Translation လႆႈလီလိူဝ်လွၵ်းလၢႆးၵဝ်ႇ မိူၼ် CNN, RNN ၶဝ်၊

Transformer ၼႆႉမီးၶေႃႈမုၼ်းဢႄႇသေတႃႉၵေႃႉမၼ်းႁဵတ်းၵၢၼ်လႆႈဝႃႈၸႂ်ႉလႆႈၼႆယူႇ၊
facebookresearch ၶဝ်ႁၼ်ဝႃႈ မၼ်းၸၢင်ႈၶႂၢၵ်ႈၵႂႃႇၸႂ်ႉၸွမ်းၸိူဝ်းၽႃႇသႃႇဢွၼ်ႇဢၼ်မီးၶေႃႈမုၼ်းတႃႇသွၼ်ပၼ် ai ၵႄႇ (Low resource language) ၶဝ်ၸင်ႇတေႇထႅင်ႈၶူင်းၵၢၼ်ဢၼ်ႁွင်ႉဝႃႈ NLLP (No Language Left Behind) ၼႂ်းပီ 2022 လႄႈဢမ်ႇႁိုင် July 2022 ၶဝ်ပွႆႇဢွၵ်ႇမႃး ai model NLLB-200 မိူၼ်ၼင်ႇ ၶိူင်ႈပိၼ်ႇၽႃႇသႃႇ Machine Translation ဢၼ်မီးပႃးလိၵ်ႈတႆးၼၼ်ႉယဝ်ႉၶႃႈ။

----

လွၵ်းလၢႆးၸၢမ်းၸႂ်ႉ Machine Translation (ၸၢႆးၸွမ်တႆး) -> [https://saizomtai.hashnode.dev/english-to-shan-translation](https://saizomtai.hashnode.dev/english-to-shan-translation)
ၸၢမ်းၸႂ်ႉ Text to Speech (ၸုမ်းႁဝ်းႁႃး) -> [https://shantts-playground.haohaa.com/](https://shantts-playground.haohaa.com/)

ၶူင်းၵၢၼ် fairseq လႄႈ NLLB မီးၶူင်းၵၢၼ်ယွႆႈလႄႈဢၼ်လီသူၼ်ၸႂ်ထႅင်ႈတၢင်းၼမ် သိုပ်ႇတူၺ်း

fairseq -> [https://github.com/facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)
NLLB -> [https://github.com/facebookresearch/flores](https://github.com/facebookresearch/flores)

----

ပွင်ႈၵႂၢမ်းဢၼ်ၼႆႉတေမႃးလဵပ်းႁဵၼ်းတူၺ်းဝႃႈ ၶဝ်ၸႂ်ႉလွၵ်းလၢႆးသင်၊ ဢဝ်ၶေႃႈမုၼ်းတီႈလႂ်မႃးသွၼ် AI ႁႂ်ႈႁူႉၸၵ်ႉလိၵ်ႈတႆးၵႂၢမ်းတႆးၼႆၶႃႈ။

#### MMS (Massively Multilingual Speech)

- Auto Speech Recognition (ASR)
- Text-to-Speech (TTS)
- Speech-to-Text (STT)
- Language Identification (LID)

ၶေႃႈမုၼ်းသဵင်လႄႈ script ဢၼ်ၸႂ်ႉတွၼ်ႈတႃႇႁဵတ်း Text-to-Speech ၼႂ်းၶူင်းၵၢၼ်ၼႆႉ ၵမ်ႈၼမ်ပဵၼ်းၶေႃႈမုၼ်းၽိုၼ်လိၵ်ႈ bible

> As part of this project, we created a dataset of readings of the New Testament in over 1,100 languages, which provided on average 32 hours of data per language.
    > - [https://ai.meta.com/blog/multilingual-model-speech-recognition/](https://ai.meta.com/blog/multilingual-model-speech-recognition/)

တူဝ်ယၢင်ႇၶေႃႈမုၼ်းၽႃႇသႃႇတႆး -> [https://globalrecordings.net/en/language/shn](https://globalrecordings.net/en/language/shn)

ၶေႃႈမုၼ်း bible ၸိူဝ်းၼႆႉတီႉဢတ်းသဵင်ဝႆႉၼပ်ႉႁူဝ်သိပ်းသၢဝ်းပီပူၼ်ႉမႃးယဝ်ႉ ၸၢမ်းထွမ်ႇတူၺ်းမၢင်ၶေႃႈၼႂ်းၶေႃႈမုၼ်းသဵင်ၸိူဝ်းၼၼ်ႉၵေႃႈ မၢင်ၶေႃႈမိူၼ်ၼင်ႇ "ၵၢင်ႁၢဝ်", "ၵင်ႁဝ်" ၸိူဝ်းၼႆႉၵေႃႈယင်းတိုၵ်ႉပိူင်ႈၵၼ်ဝႆႉလႄႈ တေႁၼ်ဝႃႈပေႃးၸၢမ်းၸႂ်ႉတူၺ်းၼႆသဵင်ဢၼ်ဢွၵ်ႇမႃးၼၼ်ႉတေဢမ်ႇၽဵၼ်ႈမိူၼ်ၵႂၢမ်းတႆးလူင်ဢၼ်ႁဝ်းၸႂ်ႉဝႆႉမိူဝ်ႈလဵဝ်

![IMG_4898 {caption: script လိၵ်ႈတႆးၼႂ်းၽိုၼ်လိၵ်ႈ bible}](/assets/fairseq-NLLB-machine-translations-analize/IMG_4898.png)

ထႅင်ႈပိူင်ၼိုင်ႈ ၵွပ်ႈဝႃႈၶေႃႈမုၼ်းသဵင်ဢၼ်ၸႂ်ႉဢမ်ႇၸႂ်ႈလၢႆးဢၼ်ႁဝ်းၸႂ်ႉတႃႇလၢတ်ႈတေႃႇၵၼ်ၵူႈဝၼ်းၼႆလႄႈ မၢင်ၶေႃႈ လမ်ႇၶိုၼ်ႈလူင်းၵေႃႈမၼ်းတေၶႆႈပိူင်ႈဝႆႉ။

ပေႃးၶႂ်ႈၸၢမ်း Train model တွၼ်ႈတႃႇ TTS ႁင်းၵူၺ်းၼႆ fairseq ၶဝ် open-source code လႄႈလွၵ်းလၢႆးတွၼ်ႈတႃႇ train ဝႆႉတီႈ [github](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)

minimum spec တွၼ်ႈတႃႇ computer ဢၼ်တေၸႂ်ႉ train

- NVIDIA GPU with at least 12GB of memory.
- RAM 32GB (at least) 64GB Recommend
- A multi-core X64-based architecture CPU.
- DISK: Minimum 200GB available
- SWAP Memory: At least 128GB

spec computer ၼႆႉဢိင်ၼိူဝ်ၶေႃႈမုၼ်းဢၼ်ႁဝ်းတေၸႂ်ႉ train ၼၼ်ႉ ပေႃးဝႃႈမီးၼမ်ၼႆၵေႃႈတေလႆႈမီး RAM ၼမ်ၼမ် ၵူၺ်းပေႃးၶေႃႈမုၼ်းဢမ်ႇမီးၼမ်ၼႆသမ်ႉ model ဢၼ်ဢွၵ်ႇမႃးၼၼ်ႉၵေႃႈ တေဢမ်ႇပေႃးၶိုၵ်ႉၶႅမ်ႉ

#### Flores (Facebook Low Resource)

- Flore101 (Large-Scale Multilingual Machine Translation)
- Flore200 (ML: ပႃးလိၵ်ႈတႆး)
- OCR (Optical Recognition)

ၶေႃႈမုၼ်းဢၼ်ၸႂ်ႉ train Machine Translations ၼႆႉသမ်လူဝ်ႇပဵၼ်ၶေႃႈမုၼ်း Language-Pair သွင်ၽႃႇသႃႇ [ပပ်ႉသပ်း Dictionary လႄႈ ၶိူင်ႈပိၼ်ႇၽႃႇသႃႇ Machine Translations](https://www.noernova.com/blog/dictionary-and-machine-translations)
ၼႂ်း paper ၶဝ်ၼၼ်ႉၼႄႉၼမ်းဝႃႈပေႃးႁႂ်ႈလီတီႈသုၼ်းႁႂ်ႈမီးၶေႃႈမုၼ်း translations ၼႂ်း wikipedia ဢၼ်ႁွင်ႉဝႃႈ [Wikipedia:List of articles all languages should have](https://simple.wikipedia.org/wiki/Wikipedia:List_of_articles_all_languages_should_have)
ၽႃႇသႃႇတႆးႁဝ်းသမ်ႉတိုၵ်ႉပဵၼ် Very Low Resource လႄႈ ပွင်ႈၵႂၢမ်းၸိူဝ်းၼႆႉယင်းပႆႇမီးၸေး။

လွၵ်းလၢႆးထႅင်ႈဢၼ်ၼိုင်ႈၶဝ်ၸႂ်ႉတႃႇၽႃႇသႃႇ VLR ၼၼ်ႉၶဝ်ဝႃႈ

> Then, we trace the development process of professionally-translated seed bitext data in 39 low-resource languages, giving us the ability to train any models that require parallel data

![Screenshot_2566-09-08_at_20.21.08 {caption: တူဝ်ယၢင်ႇၶေႃႈမုၼ်းပိၼ်ႇၽႃႇသႃႇ NLLB_200}](/assets/fairseq-NLLB-machine-translations-analize/Screenshot_2566-09-08_at_20.21.08.png)

ၶဝ်ၸႂ်ႉၵူၼ်းသေပိၼ်ႇၽႃႇသႃႇ Language-Pair လိၵ်ႈဢိင်းၵရဵတ်ႈလႄႈၽႃႇသႃႇဢၼ်တေပိၼ်ႇ

![Screenshot_2566-09-08_at_20.25.05 {caption: တူဝ်ယၢင်ႇၶေႃႈမုၼ်းပိၼ်ႇၽႃႇသႃႇ NLLB_dataset}](/assets/fairseq-NLLB-machine-translations-analize/Screenshot_2566-09-08_at_20.25.05.png)

တူဝ်ယၢင်ႇၶေႃႈမုၼ်းပိၼ်ႇၽႃႇသႃႇလူၺ်ႈၵူၼ်း ၼႂ်း [NLLB-200-SEED](https://github.com/facebookresearch/flores/tree/main/nllb_seed)

> If you are doing these things, reveal yourself to the world."
> ပေႃး မႂ်း ႁဵတ်း လွင်ႈတၢင်း ၸိူဝ်း ၼႆႉ ၸိုင်၊ၼႄပျႃး တူဝ်ၸဝ်ႈၵဝ်ႇ ထၢင်ႇထၢင်ႇသႃးသႃး ၵႃႈၼႂ်း လေႃးၵႃႉ ၼႆႉ တႃႉ၊" ဝႃႈ ၼင်ႇ ၼႆ ဢေႃႈ။

တေႁၼ်ပၼ်ႁႃဝႃႈၶေႃႈမုၼ်းၸိူဝ်းၼႆႉၶႆႈပႅၵ်ႇပိူင်ႈၵၼ်တင်းဢၼ်ႁဝ်းယၢမ်ႈႁၼ် ယၢမ်ႈယိၼ်းဝႆႉ ဢမ်ႇၼၼ်ၵေႃႈမၼ်းပိၼ်ႇဝႆႉဢမ်ႇထုၵ်ႇႁႃႉၼႆ။
ၶေႃႈမုၼ်းၸိူဝ်းၼႆႉတေႃႈၼင်ႇႁႂ်ႈႁဝ်းပိၼ်ႇၽႃႇသႃႇၼႆၵေႃႈ ႁဝ်းယင်းတေဢမ်ႇမေႃပိၼ်ႇလီလီလူးၵွၼ်ႇ ၵွပ်ႈမၢင်ၶေႃႈၼႆၵႂၢမ်းတႆးၵေႃႈယင်းပႆႇမီး။

> "He went with her to look up the graves and, returning late, said,"
> If we had not feared you would wait supper we would have stayed and been buried there.,
>
> 2013 ၼၼ်ႉ ၶဝ်လူင်း ၵႂႃႇၼႂ်းဝၢၼ်ႈသေ ႁွင်ႉဢဝ်လုင်းႁဵင်မႂ်ႇ လုင်းမွင်းၺႃး လႄႈ လုင်းႁဵင်ၵဝ်ႇ လုင်းတူႉမၼ ဝၢၼ်ႈၼွင်လိူဝ်ႇသေ လၢတ်ႈဝႃႈ \" သိုၵ်းတႆး မီးတီႈလႂ် တေလႆႈပွင်ႇၶၢဝ်ႇပၼ်ႁဝ်း ပေႃးဢမ်ႇပွင်ႇၶၢဝ်ႇလႄႈသဵင်ၵွင်ႈတႅၵ်ႇၼႆၸိုင် ယႃႇဝႃႈႁဝ်းမိူၵ်ႈ ပေႃးသိုၵ်းတႆး ယိုဝ်းႁဝ်းၶိုၼ်းတႄႉ ဝၢၼ်ႈသူ တေႁၢမ်း\" ၼႆသေ ဢွၵ်ႇၵႂႃႇ ၸွမ်းထိူၼ်ႇ ၽၢႆႇဢွၵ်ႇဝၢၼ်ႈ ဝႃႈၼႆ။"
>
> - NLLB dataset - [https://huggingface.co/datasets/allenai/nllb/viewer/eng_Latn-shn_Mymr/train?row=12](https://huggingface.co/datasets/allenai/nllb/viewer/eng_Latn-shn_Mymr/train?row=12)
> - [belebele dataset](https://github.com/facebookresearch/belebele)

ပေႃးၶႂ်ႈၸၢမ်း Train model တွၼ်ႈတႃႇ Machine Translation ႁင်းၵူၺ်းၼႆ fairseq ၶဝ် open-source code လႄႈလွၵ်းလၢႆးတွၼ်ႈတႃႇ train ဝႆႉတီႈ [documents](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model)

minimum spec တွၼ်ႈတႃႇ computer ဢၼ်တေၸႂ်ႉ train

- NVIDIA GPU with at least 12GB of memory.
- RAM 32GB (at least) 64GB Recommend
- A multi-core X64-based architecture CPU.
- DISK: Minimum 300GB available
- SWAP Memory: At least 128GB

Technical ၸိူဝ်းၼႆႉလူဝ်ႇလွင်ႈလူင်းတိုၼ်းလူင်းႁႅင်းတၢင်းၼမ် ႁဝ်းႁဵတ်းပႆႇလႆႈသေတႃႇၵေႃႈ မဵဝ်းၼိုင်ႈဢၼ်ႁဝ်းၸွႆႈၵၼ်လႆႈၵမ်းလဵဝ်ၼၼ်ႉ ပဵၼ်လွင်ႈပိၼ်ႇၽႃႇသႃႇၼႂ်း wikipedia  ဢမ်ႇဝႃႈပဵၼ်ၽႃႇသႃႇလႂ်သေတႃႉ ႁႂ်ႈပေႃးမီးၶေႃႈမုၼ်းတႃႇၽႃႇသႃႇတႆးၼမ်ၼမ်ၶႃႈ။

English - [Wikipedia:List of articles all languages should have](https://simple.wikipedia.org/wiki/Wikipedia:List_of_articles_all_languages_should_have)
Myanmar - [ဝီကီပီးဒီးယား:မြန်မာဝီကီတွင် ရှိသင့်သော ဆောင်းပါးမျာ](https://my.wikipedia.org/wiki/%E1%80%9D%E1%80%AE%E1%80%80%E1%80%AE%E1%80%95%E1%80%AE%E1%80%B8%E1%80%92%E1%80%AE%E1%80%B8%E1%80%9A%E1%80%AC%E1%80%B8:%E1%80%99%E1%80%BC%E1%80%94%E1%80%BA%E1%80%99%E1%80%AC%E1%80%9D%E1%80%AE%E1%80%80%E1%80%AE%E1%80%90%E1%80%BD%E1%80%84%E1%80%BA_%E1%80%9B%E1%80%BE%E1%80%AD%E1%80%9E%E1%80%84%E1%80%B7%E1%80%BA%E1%80%9E%E1%80%B1%E1%80%AC_%E1%80%86%E1%80%B1%E1%80%AC%E1%80%84%E1%80%BA%E1%80%B8%E1%80%95%E1%80%AB%E1%80%B8%E1%80%99%E1%80%BB%E1%80%AC%E1%80%B8)
Thai - [วิกิพีเดีย:รายการบทความที่วิกิพีเดียทุกภาษาควรมี](https://th.wikipedia.org/wiki/%E0%B8%A7%E0%B8%B4%E0%B8%81%E0%B8%B4%E0%B8%9E%E0%B8%B5%E0%B9%80%E0%B8%94%E0%B8%B5%E0%B8%A2:%E0%B8%A3%E0%B8%B2%E0%B8%A2%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%9A%E0%B8%97%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B8%A7%E0%B8%B4%E0%B8%81%E0%B8%B4%E0%B8%9E%E0%B8%B5%E0%B9%80%E0%B8%94%E0%B8%B5%E0%B8%A2%E0%B8%97%E0%B8%B8%E0%B8%81%E0%B8%A0%E0%B8%B2%E0%B8%A9%E0%B8%B2%E0%B8%84%E0%B8%A7%E0%B8%A3%E0%B8%A1%E0%B8%B5)
Chinese - [维基百科:基礎條目](https://zh.wikipedia.org/wiki/Wikipedia:%E5%9F%BA%E7%A4%8E%E6%A2%9D%E7%9B%AE)
