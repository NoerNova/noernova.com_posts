---
tags: ["google-sheet-api", "apexcharts"]
date: "July 8, 2020"
title: " [Part I] Google Sheet แสดงยอดเงินบริจาคบนเว็บไซต์ แบบง่ายๆไวๆ"
subtitle: "Google sheet แสดงยอดเงินบริจาคบนเว็บไซต์ the serie"
image: "https://miro.medium.com/v2/resize:fit:720/format:webp/1*5MgFnE99GHp8V7zaWRb8HA.jpeg"
link: "blog/google_sheet-donation_amount-part1"
description: "ใช้ Google Sheet ทำ Databse เก็บข้อมูล แล้วดึง api มาแสดงผลสวยๆกัน!!!"
---

**สวัสดีครับ** จากกระแสการออกรับเงินบริจาคแล้วไม่ยอมแสดงความโปร่งใสกันแบบเนิ่นๆในบ้านเรา …. _ไม่เอาๆ ริวจะไม่ยุ่งเรื่องนี้ lol_

...

เริ่มใหม่ๆ เนื่องจากผมได้รับการไหว้วาน (งานฟรี) ให้ช่วยในโครงการรับบริจาคขององค์กรหนึ่งไม่แสวงหาผลกำไร มี requirement ดังนี้

1. มีกราฟแสดงหลักไมล์การรับบริจาค
2. มีเลขแสดงจำนวนผู้บริจาค
3. รองรับ 6 สกุลเงิน [บาท,จัตพม่า,ปอนด์,หยวน,ดอลล่า,ดอลล่าสิงคโปร์]
4. ซึ่ง total ยอดบริจาคจะแสดงเป็น "จัตพม่า"
5. อัพเดทตามอัตราแลกเปลี่ยนทุกวัน
6. ฝ่ายการเงินจะเป็นคนอัพเดทยอดเงินบริจาค (ซึ่งต้องอัพเดทง่ายและเร็วที่สุด)
7. ทำเสร็จภายในวันนี้ (ครึ่งวัน)
8. ไม่มีงบในส่วนนี้

นะๆๆ ช่วยหน่อยน้าาาา (ทำตาปริบๆ)
คำถามแรก… ฝ่ายการเงินจะจัดการบัญชีด้วยอีท่าไหนถึงจะง่าย และเร็วที่สุด Excel แน่นอนอยู่แล้ว คำถามต่อไปคือจะทำยังไงถึงจะดึงตัวเลขในไฟล์ excel ไปแสดงบนเว็บไซด์ล่ะเนี่ย (ไม่เคยทำวุ้ยยย) เลยกูเกิลดูซะหน่อย
ก็พอจะเจออยู่หลายวิธี แต่!!! ฝ่ายการเงินส์ (เติม s) หลายคน ไฟล์ excel คนล่ะไฟล์ อยู่คนละประเทศ (ดูแลตามสกุลเงิน) เอาว่ะ งานรีบแบบเน้ง่ายสุดที่คิดออกก็ Google Sheet ละกัน น่าจะตอบโจทย์สุด

...

จึงลิสรายการสิ่งที่ต้องการออกมา
**- Google Sheet 1 ไฟล์ แชร์แบบ Publish view**
**- api สำหรับแลกเปลี่ยนสกุลเงิน**
**- api สำหรับแสดงผลกราฟ**
**- เขียนด้วย html + javascript (เหลือเวลาอีกสี่ชั่วโมง)**

### ออกแบบ ไฟล์ Sheet เลย

หน้าตา Sheet ของเราคร่าวๆตามนี้

![Screen shot](https://cdn-images-1.medium.com/max/1600/1*isnIuTuXDjpXP5SZns7sKQ.png)
_google-sheet_

**(เซลล์สีเขียวสำหรับแก้ไขได้)** ประกอบด้วยสกุลเงินทั้ง 6 และเลขแสดงจำนวนผู้บริจาค
(เซลล์สีเหลืองสำหรับแก้ไขไม่ได้ อัพเดทอัตโนมัติ) แสดง total ยอดบริจาค แปลงเป็นสกุลเงิน จัต แล้ว, ยอดบริจาคที่ต้องการ และหลักไมล์ที่มาถึง แสดงเป็นเปอร์เซ็น

**Google sheet** นั้นสามารถกำหนดสิทธิได้ว่าเซลล์ (cell) ไหนใครแก้ไขได้ โดยการเลือกเซลล์ ที่จะกำหนดสิทธิ์ → Data → Protected sheets and ranges

กำหนดสิทธิ์ไปเลยว่าใครแก้ช่องไหน ตรงนี้สำคัญ จะได้ไม่ต้องกังวลว่าฝ่ายการเงินประเทศไหนแก้ไขผิดช่องรึเปล่าหว่าาา ข้อดีอีกอย่างคือเราสามารถดูประวัติการแก้ไขได้ยาวๆเลยนะ

> **โน๊ตตัวใหญ่ๆตรงนี้ว่า** Sheet เราจะแชร์ลิงค์เป็น public ใครก็สามารถเข้ามาดูไฟล์ได้ แต่ไม่ให้สิทธิ์ในการแก้ไขนะครับ (Share with anyone with the link can view) ตรงนี้มีเหตุผล อ่านต่อเล้ยยย :D

#### โอเคทีนี้… How to อัพเดท total อัตโนมัติ

พวกยอดบริจาคตามสกุลเงินต่างๆนั้นฝ่ายการเงินเข้ามาอัพเดทให้เราอยู่แล้วสิ่งที่เราต้องทำคือ

- แปลงไปเป็นสกุลเงินจัต (MMK) ทั้งหมด
- รวมยอด
- อัพเดทเซลล์ total
- อัพเดทเซลล์ landing percentage
- และสิ่งที่เราจะใช้ในการให้มันอัพเดท (ซึ่งต้องอัพเดททุกครั้งที่มีการแก้ไขยอดไม่ว่าในสกุลเงินใดๆก็ตาม) คืออออ Google Apps Script นั่นเอง

> Google Apps Script is a scripting language based on JavaScript that lets you do new and cool things with Google Apps like Docs, Sheets, and Forms. There's nothing to install - we give you a code editor right in your browser, and your scripts run on Google's servers.

เขานิยามไว้แบบเน้ ก็ไม่ลงลึกละนะ ไปศึกษากันได้ประยุกต์ใช้ได้หลายประโยชน์
ไปทำกันเลยยย

### Tools → Script Editor

![เจ้าหน้าตา Script Editor ของ Google Apps Script](https://cdn-images-1.medium.com/max/1600/1*v9_ugcCYduTySl1-0bMm5g.png)

แต่ก่อนอื่น เราต้องหา api สำหรับการแปลงสกุลเงิน **Currency Converter** ก่อน สิ่งสำคัญสองประการคือ ฟรี และ **รองรับสกุลเงินจัต (MMK)** ซึ่งตัวฟรีนั้นก็พอหาได้เยอะอยู่แต่ส่วนใหญ่แล้วไม่รองรับสกุลเงินจัต (MMK) T_T

หาไปซักผ้า เอ้ยย ซักผัก เอ้ยยย สักพัก เอ้ยยย ถูกแล้ววว!!!
กำลังจะถอดใจไปฟิกแบบแม่นวลแล้ว เจอเว็บนี้มา ฟรีแบบมีเงื่อนไข แต่รองรับ MMK โอเคใช้ไปก่อน currencyconverterapi.com ต้องขอ Free API Key

_**Code เล้ยยย**_

![Code ใน Script Editor {caption: Code ใน Script Editor}](https://cdn-images-1.medium.com/max/1600/1*djBUQXA-M1Ganu63YYAswA.png)

**อธิบายโค้ด**
ตัวโค้ดมีอยู่ห้าส่วนสองฟังก์ชัน

1. ฟังก์ชัน `getCurrencyRate()` ทำหน้าที่ดึง exchange rate มาแล้วคายเป็น JavaScript object ออกมา
2. ฟังก์ชัน `converter()` ฟังก์ชันหลักของเรามีส่วนประกอบดังนี้

```javascript
var api = getCurrencyRate(); // เรียกใช้ getCurrencyRate()

var thb = api["THB_MMK"];
var usd = api["USD_MMK"];
var cny = api["CNY_MMK"];
var gbp = api["GBP_MMK"];
var sgd = api["SGD_MMK"];
// ประกาศตัวแปรสำหรับเก็บ exchange rate ของแต่ละสกุลเงิน
// จ๊าดพม่าไม่ต้องเพราะไม่ต้องแลกเปลี่ยน
```

_**ทำงานกับ Sheet**_

```javascript
var openSpreadSheet = SpreadsheetApp.getActiveSpreadsheet();
// ประกาศการเข้าถึง sheet ของเรา

var sheet = openSpreadSheet.getActiveSheet();
// เรียกใช้ แท็บ, แผ่นงาน ที่เราใช้ ในกรณีมีหลายแผ่นงาน อาจใช้
// getSheetByName() แทน

var Bath = sheet.getRange("B2").getValue();
var Dollar = sheet.getRange("C2").getValue();
var Yuan = sheet.getRange("D2").getValue();
var Pound = sheet.getRange("E2").getValue();
var SgDollar = sheet.getRange("F2").getValue();
// ประกาศตัวแปลสำหรับเก็บค่ายอดบริจาคตามสกุลเงินต่างๆจากใน sheet
// getRange ช่วงตำแหน่งใน sheet รูปแบบ
// → getRange(row, column, numRows, numColumns)
// ในที่นี้ผมเรียกตามตำแหน่ง cell เลยง่ายดี
```

_**สมการแลกเปลี่ยน**_

```javascript
var kyat =
  sheet.getRange("A2").getValue() +
  Bath * thb +
  Dollar * usd +
  Yuan * cny +
  Pound * gbp +
  SgDollar * sgd;
// จาก sheet ของเรา 'A2' คือ จ๊าด (MMK) เลยรวบๆเรียกตรงนี้แล้วบวกค่าเงินที่แลกเปลี่ยน
// exchange = money x exchangeRate
```

_**set ค่า**_

```javascript
sheet.getRange("I2").setValue(kyat);
// เซ็ตค่าสุดท้ายให้ช่อง Grand total ของเรา
// เป็นยอดบริจาคทั้งหมดที่ผ่านการแลกเปลี่ยนเป็นเงินจ๊าดแล้ว

// Logger.log(kyat); ใช้ log เพื่อดีบั๊กบ้าง
```

กดเซฟ & Run
ไปที่เมนู Run → Run function
จะเห็นชื่อฟังก์ชันที่เราเขียนไว้ ในกรณีของเราเรียกใช้ฟังก์ชันหลัก `converter()` ก็กด run ไป

หากเป็นการ run ครั้งแรก จะต้องมีการยืนยันตรวจสอบสิทธิ์กันก่อน

![การตรวจสอบสิทธิ์ในการ run ครั้งแรก {caption: การตรวจสอบสิทธิ์ในการ run ครั้งแรก}](https://cdn-images-1.medium.com/max/1600/1*6ix4D0Pe8Ly0wmN5pb543A.png)

กดตรวจสอบสิทธิ์ → Advance → Goto …**ชื่อที่เราตั้งไว้** แล้ว Allow ได้เลย จะมีการขอสิทธิ์ See, edit, create, delete สำหรับ sheet ของเรา และสิทธิ์สำหรับเรียกใช้ external service ในการเรียกใช้ api อัตราแลกเปลี่ยนของเรา

run เสร็จแล้วกลับไปดูหน้า sheet ของเรา

![Grand total ได้รับการอัพเดทแล้ว {caption: Grand total ได้รับการอัพเดทแล้ว}](https://cdn-images-1.medium.com/max/1600/1*sPAF3-86L7oJaU9uBlYY6Q.png)

โอเช **Grand total** มาแล้ว ช่อง **Percent count to Landing** ก็ใช้ท่า excel ธรรมดา
$$ =(I2/J2)*100 $$
จบในส่วนของ Google App Script

---

### [ต่อ Part 2 เนาะ](/blog/google_sheet-donation_amount-part2)
