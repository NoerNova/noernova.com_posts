---
tags: ["qnapNAS", "NAS", "Search"]
date: "September 28, 2022"
title: "QNAP NAS create own search system with File Station API Part I"
subtitle: "ႁဵတ်း search system တႃႇ QNAP NAS Part I"
image: "https://user-images.githubusercontent.com/9565672/192691085-9264b646-5558-43f5-a981-0410a7d181ea.png"
link: "blog/qnap_nas_search_p1"
description: "ႁဵတ်း website တွၼ်ႈတႃႇၶူၼ်ႉႁႃၶေႃႈမုၼ်းၼိူဝ် NAS (Network Attached Storage) လူၺ်ႈ NextJS, Docker, Nginx"
---

## ႁဵတ်း search system တႃႇ QNAP NAS Part I

QNAP NAS ပဵၼ် Storage ၵဵပ်းၶေႃႈမုၼ်းဢၼ်ၸႂ်ႉတိုဝ်းၵၼ်ၼမ်ဝႆႉမဵဝ်းၼိုင်ႈ သိုပ်ႇလူလွင်ႈ [NAS](https://www.seagate.com/sg/en/tech-insights/what-is-nas-master-ti/)

ပၼ်ႁႃဢၼ်ၼိုင်ႈၼႂ်း QNAP NAS ပေႃးၶူၼ်ႉႁႃ file ၼႂ်း File Station ၼႆမႂ်းထိူင်း ဢမ်ႇဝႃႈတေၸႂ်ႉတၢင်း Internet ႁိုဝ် ၸႂ်ႉတေႃႇၶွမ်ႊၵမ်းသိုဝ်ႈ ပေႃးဝႃႈ ၸႂ်ႉ QSirch ၵေႃႈ ၵိၼ် resource ၼမ်ႁႅင်းပူၼ်ႉတီႈ
ပေႃးၼႂ်း office မီးၵူၼ်းၸႂ်ႉမွၵ်ႈ 5 ၵေႃႉၵေႃႈၶႂ်ႈၶမ်ၵိုတ်းၶမ်ၵိုတ်းယဝ်ႉ

ၼႆလႄႈၸင်ႇဝူၼ်ႉဝႃႈပေႃးႁဵတ်း Search System ႁင်းၵူၺ်းမၼ်းဝႆႉတႃႇ search ၼိုင်ႈဢၼ်ၼႆ တေၸွႆႈႁႂ်ႈမဝ်လွင်ႈ resource ယူႇၵမ်ႈၽွင်ႈဢေႉ ။

![Screen Shot 2565-09-28 at 11 48 37](https://user-images.githubusercontent.com/9565672/192691085-9264b646-5558-43f5-a981-0410a7d181ea.png)

## 1. Make NAS private

တွၼ်ႈတႃႇလွင်ႈႁူမ်ႇလူမ်ႈလွတ်ႈၽေးၶေႃႈမုၼ်း expert ၶဝ်ၼႄႈၼမ်းဝႃႈ ယႃႇၸႂ်ႉ NAS လူၺ်ႈတၢင်း Internet ပေႃးဝႃႈဢမ်ႇဝေႈလႆႈၼႆၸိုင် ႁႂ်ႈၸႂ်ႉတၢင်း VPN သေၶဝ်ႈ (မၢႆထိုင် setup VPN server ၼႂ်း NAS သေ ၶိုၼ်း ၸႂ်ႉ VPN client သေၶဝ်ႈၸႂ်ႉ)

လွၵ်းလၢႆးၸႂ်ႉတိုဝ်း NAS တီႈ office ၼႆႉတႄႉၵေႃႈ ဢမ်ႇသူႈလႆႈလုၵ်ႉတၢင်းၼွၵ်ႈသေၶဝ်ႈဢဝ်ၶေႃႈမုၼ်းၼႆလႄႈ ႁဵတ်းႁႂ်ႈၸႂ်ႉ NAS လႆႈတီႈ office ၵူၺ်း internet တၢင်းၼွၵ်ႈတေဢမ်ႇၶဝ်ႈလႆႈ

### DDNS settings

QNAP NAS
ၵႂႃႇတီႈ ``myQNAPcloud --> My DDNS သေပိၵ်ႉဝႆႉ My DDNS``

![Screen Shot 2565-09-28 at 11 10 36](https://user-images.githubusercontent.com/9565672/192690809-fa955fa7-8e76-4a34-9f80-27d709d6a543.png)

ၵႂႃႇတီႈ ``myQNAPcloud --> Access Control တီႈ Device access controls ၼၼ်ႉလိူၵ်ႈ Private``

![Screen Shot 2565-09-28 at 11 10 50](https://user-images.githubusercontent.com/9565672/192690874-4fcb302a-703b-4559-8589-a01ed2a2eba4.png)

ႁဵတ်းၸိူင်ႉၼႆၵေႃႈ NAS ႁဝ်းတေပဵၼ် Private ၶဝ်ႈလႆႈတင်း Network တီႈလုမ်းၵူၺ်းယဝ်ႉ

## 1.1 MAP IP address and NAS Device

ပေႃးဝႃႈ NAS ႁဝ်းပဵၼ် Private ၶဝ်ႈလႆႈတီႈလုမ်းၵူၺ်းၸိုင် DNS ၸူဝ်းၼႆႉတေၸႂ်ႉဢမ်ႇလႆႈ တေလႆႈၶဝ်ႈတင်း IP 192.168.x.x ၸူဝ်းၼႆႉၵူၺ်း
သင်ဝႃႈၶႂ်ႈၶဝ်ႈတၢင်းၸိုဝ်ႈ Domain ၼႆ တေလႆႈ map ip တီႈ router မိူၼ်ၼင်ႇ map search.cloud.local -> 192.168.1.111 ပေႃး connect တင်း router ၼၼ်ႉသေၶဝ်ႈ search.cloud.local ၵေႃႈမၼ်းတေ direct ၵႂႃႇတီႈ NAS ႁဝ်းၵမ်းလဵဝ်

### IP address settings

admin ၵႂႃႇတီႈ ip တွၼ်ႈတႃႇ setting router မိူၼ်ၼင်ႇ 192.168.1.1 (router ဢၼ်ၼိုင်ႈလႄႈဢၼ်ၼိုင်ႈ တေဢမ်ႇမိူၼ်ၵၼ်)
ႁႃတွၼ်ႈ ``Local network --> DNS --> Host Name`` သေ map ၸိုဝ်ႈ domain ဢၼ်ႁဝ်းၶႂ်ႈၸႂ်ႉတင်း IP ၶွင် NAS ႁဝ်း

![Screen Shot 2565-09-28 at 11 22 39](https://user-images.githubusercontent.com/9565672/192690949-07196faa-aa43-494e-adcd-bfc4acdbe78a.png)

ပေႃးႁဵတ်း setting ၸူဝ်းၼႆႉယဝ်ႉ router ႁဝ်းတေပဵၼ်မိူၼ်ၼင်ႇ DNS server ပေႃးဝႃႈၸႂ်ႉဝႆႉ QuFirewall ၵေႃႈယႃႇလိုမ်း add DNS server IP ဝႆႉၸွမ်း

ၸၢမ်းၶဝ်ႈတူၺ်းတီႈၸိုဝ်ႈ Domain ဢၼ်တင်ႈဝႆႉ ပေႃးဝႃႈပႆႇၶဝ်ႈလႆ ႁႂ်ႈ disconnect wifi သေၶိုၼ်းၶဝ်ႈ မၼ်းတေ update DNS ပၼ်ထႅင်ႈၵမ်းၼိုင်ႈ

### 1.2 SSL

ပေႃးဝႃႈၶဝ်ႈလႆႈယဝ်ႉၼႆတေႁၼ်ဝႃႈမၼ်းပဵၼ်ဝႆႉ Not Secure Connection ပေႃးဝႃႈၶႂ်ႈၸႂ်ႉ SSL ပႃးၸွမ်းၼႆ ႁဝ်းတေလႆႈ generate SSL သေ

1. Generate SSL certificate
    - [https://github.com/FiloSottile/mkcert](https://github.com/FiloSottile/mkcert)
2. import ၶဝ်ႈၼႂ်း qnap nas
![Screen Shot 2565-09-28 at 11 38 15](https://user-images.githubusercontent.com/9565672/192690989-9656366e-063c-462f-8842-f1d9bcac9efb.png)
3. Install ၼႂ်း Browser/ ၼႂ်း Com ႁဝ်း
![Screen Shot 2565-09-28 at 11 43 39](https://user-images.githubusercontent.com/9565672/192691033-ae3cc3e3-725d-4494-b040-0347a92a476e.png)

တူဝ်ယၢင်ႇ add SSL certificate ၼႂ်း Firefox

- [Firefox](https://docs.titanhq.com/en/3834-importing-ssl-certificate-in-mozilla-firefox.html)
- [Chrome](https://support.securly.com/hc/en-us/articles/206081828-How-do-I-manually-install-the-Securly-SSL-certificate-in-Chrome-)
- **ပေႃးပဵၼ်ၼႂ်း MacOS တေလႆႈ add ၶဝ်ႈၼႂ်း Keychain**

ၼႆၵေႃႈႁဝ်းတေၸႂ်ႉ NAS လႆႈလူၺ်ႈပဵၼ် Private Network လႄႈမီး SSL ယဝ်ႉၶႃႈ

ပႆႉသိုပ်ႇ Part II.
