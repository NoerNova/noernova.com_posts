---
tags: ["code"]
date: "May 30, 2024"
title: "NextJS Typescript .env"
link: "note/nextjs_ts_env"
description: "How to config .env for NextJS typescript"
---

In NextJS typescript I used env.local to test local environment but got warning

``Type 'string | undefined' is not assignable to type 'string'.``

so here's to config type

```Typescript
  namespace NodeJS {
    interface ProcessEnv {
      NEXT_KEY1: string;
      NEXT_KEY2: string;
    }
  }
```

then add ``env.d.ts`` to ``tsconfig.json``

```Javascript
  {
    "compilerOptions": {
      // options
    },
    "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", "env.d.ts"]
  }
```
