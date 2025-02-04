---
tags: ["google-sheet-api", "apexcharts"]
date: "July 9, 2020"
title: "[Part II] Google Sheet แสดงยอดเงินบริจาคบนเว็บไซต์ แบบง่ายๆไวๆ"
subtitle: "Google sheet แสดงยอดเงินบริจาคบนเว็บไซต์ the serie"
image: "https://miro.medium.com/v2/resize:fit:720/format:webp/1*uYUssUmqDYtQZb833sm9Kg.png"
link: "blog/google_sheet-donation_amount-part2"
description: "ใช้ Google Sheet ทำ Databse เก็บข้อมูล แล้วดึง api มาแสดงผลสวยๆกัน!!! ภาค 2"
---

[จากตอนที่แล้ว(Part I)](/blog/google_sheet-donation_amount-part1)

สิ่งที่เราจะทำต่อไปคือการดึงข้อมูลยอดบริจาคทั้งหมดจาก Sheet ของเราไปใช้บนเว็บไซต์ ซึ่ง**ร่างไว้แบบนี้**

![Before {caption: Before}](https://cdn-images-1.medium.com/max/800/1*5MgFnE99GHp8V7zaWRb8HA.jpeg)

เอาไปทำแบบนี้

![After {caption: After}](https://cdn-images-1.medium.com/max/800/1*uYUssUmqDYtQZb833sm9Kg.png)

> ตัว UI นั้นขอไม่ลงรายละเอียดในที่นี้ครับเดี๋ยวจะยุบยับไปหน่อย ในที่นี้เราจะสนใจเฉพาะการดึงข้อมูลจาก **Google sheet** มาใช้
> #(**edit**) ตัวกราฟนั้นผมใช้ [apexcharts.com](https://apexcharts.com/) เป็น Open Source JavaScript Charts ใช้กันฟรีๆเลยทีเดียวเชียว

...

การดึง data จาก **Google sheet** จะเป็นลักษณะของการเรียกใช้ **api** นะครับ ซึ่งเราจะเรียกมาเฉพาะ `value`
รูปแบบการเรียกจากเว็บไซด์หลัก **Google sheet API v4** แนะนำไว้แบบนี้

```bash
GET https://sheets.googleapis.com/v4/spreadsheets/{spreadsheetId}/values/{range}
```

ลองเอาไปเรียกใช้บน Browser ดู

> **ตรงนี้ทำได้เพราะเราเปิด Public Sheet ของเรานะครับ**
>
> ย้ำโน๊ตตัวใหญ่ๆว่า Data ใน sheetsเราจะแชร์ลิงค์เป็น public ใครก็สามารถเข้ามาดูไฟล์ได้ แต่ไม่ให้สิทธิ์ในการแก้ไขนะครับ (Share with anyone with the link can view) ดังนั้นหากใครจะใช้กับ Data ที่ต้องการ privacy ก็ต้องไป config ดีๆ ใช้ OAuth 2.0 นะครับ การเรียกก็จะเป็นอีกแบบ

```json
// 20200707185756
// https://sheets.googleapis.com/v4/spreadsheets/{SheetsID}/values/Sheet1!A2:K2
{
  "error": {
    "code": 403,
    "message": "The request is missing a valid API key.",
    "status": "PERMISSION_DENIED"
  }
}
```

ผลการเรียกใน Browser บอกว่าเราต้องใช้ **API key** ด้วย

> _จะเสร็จมั้ยยยย 5555 T_T_

### โอเค ฮึบไว้ ไปล่า API key กัน

ตรงนี้เราต้องการจะ **Track traffic** การเรียกใช้ Google sheets API ด้วย
ดังนั้นก่อนอื่นเราต้องไป `Enable APIs and Services` เปิดใช้ Google sheet api ใน [console.developers.google.com](https://console.cloud.google.com/) ซะก่อน หากไม่ต้องการก็สามารถข้ามไปขั้นตอนการสร้าง API key ได้เลย

เข้าเว็บมาใหม่ๆถ้ายังไม่เคยเปิดใช้าน api อะไรเลยจะเจอหน้าโล่งๆแบบนี้คลิกที่ `Enable APIs and Services` ได้เลย

![Google APIs & Services {caption: Google APIs & Services}](https://cdn-images-1.medium.com/max/800/1*YJmboExGCmg2tyzvBolgbw.png)

Google มี api ของบริการต่างๆมากมายให้เราเล่น สิ่งที่เราสนใจคือ Google Sheets API ใครหาไม่เจอใช้ช่องค้นหาด้านบนได้

![Services น่าเล่นมากมายของ GoogleGoogle {caption: Services น่าเล่นมากมายของ GoogleGoogle}](https://cdn-images-1.medium.com/max/800/1*SWcTD7pjY0i2Bg9oaj-FBw.png)

![Sheets API](https://cdn-images-1.medium.com/max/800/1*kBNTbwP3esq8iT8wKoRmTA.png)

กด `Enable` เล้ยยย

จะได้หน้านี้มาซึ่งเอาไว้ดูทราฟฟิกการใช้งานต่างๆได้

![หน้ารายละเอียดการใช้งาน Google sheets api {caption: หน้ารายละเอียดการใช้งาน Google sheets api}](https://cdn-images-1.medium.com/max/800/1*eAmc8KGLUwhBEBuYNGEYfA.png)

### สร้าง API key กัน

ที่หน้า Dashboard ไปที่เมนู `Credentials` → `Create Credentials` → `API Key`

![Create API Credentials {caption: Create API Credentials}](https://cdn-images-1.medium.com/max/800/1*BMO6cUk4mnlcW6UCneKJQQ.png)

จะได้ `API Key` แบบนี้มาซึ่งเราต้อง `Restrict` ให้มันใช้ได้กับ Google sheet Api ของเรา กดไปที่ `Restrict key`

![API key created {caption: API key created}](https://cdn-images-1.medium.com/max/800/1*QkQZ0ztcEe87MXDV8Grc2A.png)

![Restrict สำหรับ Google sheets API ที่เราเปิดใช้ไว้เมื่อตะกี้ {caption: Restrict สำหรับ Google sheets API ที่เราเปิดใช้ไว้เมื่อตะกี้}](https://cdn-images-1.medium.com/max/800/1*ktc1GL-seztHCEsnGVstyQ.png)

#### เสร็จแล้ว ได้ API key มา เอาไปเรียกดู

`https://sheets.googleapis.com/v4/spreadsheets/{SheetsID}/values/Sheet1!A2:K2?key={API key}`

**ผ่ามมม!!!**

```json
// 20200708145312
// https://sheets.googleapis.com/v4/spreadsheets/{SheetID}/values/Sheet1!A2:K2?key={API key}
{
  "error": {
    "code": 403,
    "message": "The caller does not have permission",
    "status": "PERMISSION_DENIED"
  }
}
```

ที่เป็นแบบนี้เพราะเรายังไม่ได้แชร์ Sheet ของเราให้เป็น Public ครับ

![กดแชร์ Sheet ของเราให้เรียบร้อย, {caption: กดแชร์ Sheet ของเราให้เรียบร้อย}](https://cdn-images-1.medium.com/max/800/1*OovfGXDW22qPhS30fX3prQ.png)

พอแชร์เสร็จเรียบร้อยเรียกใหม่

```json
// 20200708145918
// https://sheets.googleapis.com/v4/spreadsheets/{Sheets ID}/values/Sheet1!A2:K2?key={API key}
{
  "range": "Sheet1!A2:K2",
  "majorDimension": "ROWS",
  "values": [
    [
      "85,565,700.00",
      "263,435.10",
      "45",
      "15,250.00",
      "300",
      "100",
      "269",
      "",
      "97,146,178.28",
      "400,000,000.00",
      "24.28654457"
    ]
  ]
}
```

##### มาแล้ววววว เย้ยยยย

...

จากนั้นก็เอาไปเรียกใช้บนเว็บของเราได้แล้วววว… 0–0

...

Part I → [Google Sheet แสดงยอดเงินบริจาคบนเว็บไซต์ แบบง่ายๆไวๆ [Part I]](/blog/google_sheet-donation_amount-part1)
PS. ใครมีเครื่องไม้เครื่องมืออะไรที่ ฟรี เจ๋ง ง่าย ก็เอามาแชร์กันนะครับ
PS2. ใครมีคำแนะนำเรื่องการตั้งค่าที่ไม่ค่อยจะปลอดภัยก็ฝากชี้แนะครับผม
