---
tag: "code"
date: "May 28, 2024"
title: "Some useful Bash script"
link: "note/bash_script_command_line"
description: "some useful bash script"
---

ffmpeg converter

 ```bash
  for f in /path/to/folder/*; do ffmpeg -i "$f" "${f%.*}.webp"; done
 ```

find and delete pattern

 ```bash
  find /path/to/folder -type f ! -name "*.webp" -delete
 ```
