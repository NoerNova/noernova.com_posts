---
tags:
  - computer
  - aws
  - cloud
  - cloud-computing
date: February 15, 2024
title: AWS (Amazon Web Service) BILL SHOCK!
subtitle: เจอ bill shock จาก AWS ไป 20,000 ขอคืนเงินได้นะ
image: https://i.pinimg.com/564x/54/b0/06/54b00642c4f6d37b725781d0da05939b.jpg
link: blog/aws-bill-shock
description: ลองขอคืนเงินจาก AWS service ที่เราพลาดเอง
---
เรื่องมีอยู่ว่า สักประมาณตี 4 กว่าๆเรากำลังจะเข้านอนตามปกติ มี email ดึ้งๆเข้ามาว่า

![amazon-web-service {caption: AWS billing}](/assets/aws-bill-shock/IMG_6402.jpg)

ห่ะ!! อะไรนะ **20,852.28** บาท ในใจคือคิดว่าเอาแล้ว account โดนแฮ็กแน่ๆ เพราะไม่ได้เปิด service อะไรเพิ่มมาเดือนกว่าแล้วแน่ๆ ปกติค่า service ที่เปิดไว้ใช้ไม่เกิน 20$
แต่เอ๊ะ ล่าสุดที่เปิด service ใหม่เดือนกว่าๆหรอ? เลย login เข้าไปใน account ก็ดู ปกติดี ไม่มีอะไรโดนแก้ แอคน่าจะยังปลอดภัย เลยเข้าไปดู service ที่เปิดไว้แล้วกินตังค์เยอะสุดซิ ...

![aws-account-service {caption: AWS account service}](/assets/aws-bill-shock/Screenshot-2566-04-05-at-13.07.00.png)

อ่ออ โง่เองแหละ  เปิด RDBMS Aurora DB ทิ้งไว้ 😆 จำไม่ได้ด้วยซ้ำว่าเปิดไว้เพื่อทดสอบอะไร lol, เดือนเดียวซดไปสองหมื่นเลยเราะ ในใจคิดว่าเมื่อเราพลาดเองคงทำอะไรไม่ได้แล้วแหละ ตอนจะ kill service ลองคิดเล่นๆว่า ถ้าเราบอกว่าเราตั้งค่าผิดแล้วบอกเขาว่าเป็น unintent charge จะได้ไหมนะเพราะเราไม่ได้ใช้งานมันเลยนิ ... อ่ะไหนๆก็ไหนๆละ เปิดไว้งั้นก่อน เผื่อเขาจะได้เช็คว่าเราไม่ได้ใช้งานอะไรเลยจริงๆนะ

### ขอตังค์คืนหน่อยค้าบบบ

แรกสุดเลย เราเข้าไปเปิด support ticket ไว้ใน support center หัวข้อ Account & billing เปิด Create case

![aws-support](/assets/aws-bill-shock/Screenshot-2566-04-05-at-13.07.59.png)

![aws-support](/assets/aws-bill-shock/Screenshot-2566-04-05-at-13.08.51.png)

แล้วเขียนไปประมาณว่า เออ เราเผลอเปิด service นึงทิ้งไว้โดนไม่ได้ใช้งาน จริงๆเราตั้งใจใช้แค่ Free VPS service นึง แต่มันดันพ่วง Aurora DB ที่กินตังค์เข้ามาด้วย พอเราปิด VPS service นั้น Aurora มันไม่ได้ปิดไปด้วย แต่เราไม่ได้ใช้งานมันเลยนะ possible to make a refund ให้เราได้ป่าว? ประมาณนี้เป็นภาษาอังกฤษแหละ

![aws-support](/assets/aws-bill-shock/Screenshot-2566-04-05-at-13.09.13.png)

ตอนแรกเราตั้งเป็น web ticket ไว้แต่เหมือนไม่ได้ผลหรืออาจใช้เวลานาน เลยเปิด Chat support ticket อีกรอบ

อันนี้เราจะได้คุยกับ supporter ที่เป็นคนจริงๆเลย ก็อธิบายไปเหมือนข้างบน เขาก็จะเช็คๆว่าเราไม่ได้ใช้งาน service นั้นจริงๆรึเปล่า อ่ะพอเขาเห็นว่าเราไม่ได้ใช้งานมันจริงๆแค่เปิดไว้เฉยๆ เขาจะเปิด ticket ใหม่ให้เราอีกอัน ระหว่างนั้นเขาก็บอกให้เราปิด service นั้นไปได้เลยนะ เดี๋ยวไปคุยกันต่อใน ticket

ใน ticket ใหม่เขาจะมีชุดคำถามทั่วๆไปใหเราตอบ

![aws-support](/assets/aws-bill-shock/Screenshot-2566-04-05-at-13.05.52.png)

หลังตอบเสร็จประมาณ สองชั่วโมง

![aws-support](/assets/aws-bill-shock/Screenshot-2566-04-05-at-13.05.38.png)

อ่าาาห์ ได้ตังค์คืน 3-5 business days ตอนเราได้เงินคืนน่าจะเนื่องมาจากค่าเงินตอนโดนตัด กันตอนได้เงินคืน หรือเรื่องภาษีด้วยมั้ง เลยได้คืน 19,xxx กว่าๆ แต่ก็ดีกว่าไม่ได้เลย

สรุป จากที่เจอเอง แล้วอ่านของคนอื่นๆใน community

- สำหรับมือใหม่ AWS Free tier มันไม่ได้ completely free 100% หรอกนะ ระวังพวก service พ่วงดีๆ มันจะมี shock bill จากพวกนั้นแหละ
- หากเจอ shock bill เช็คที่ service account ก่อนว่าเปิดอะไรใว้บ้าง แล้วเราได้
  - เช่นหากเราเปิด DBMS service ไว้แล้วมีการเขียนอ่านไปแม้แต่ transection เดียวก่อนเช็คบิล ก็หมดสิทธิได้เงินคืน
  - หากแน่ใจว่าไม่ได้ใช้ service นั้นจริงๆ ให้ terminated ไว้ก่อน อย่าเพิ่งไป delete เผื่อให้เข้าเช็คได้ว่าเราไม่ได้ใช้จริงๆนะ
- อันที่จะได้เงินคืนง่ายๆหน่อยจะเป็นพวก bill ที่มากับ service พ่วงมากกว่า เช่นกรณีนี้ เราได้คืนแค่เฉพาะ Aurora DB ที่พ่วงไว้กับ VPS ตัวนึง ส่วน VPS ที่โดนคิดตังค์ไป 14$ ไม่ได้คืนนะ
- ตั้ง billing limited และ billing alert ไว้เสมอ
