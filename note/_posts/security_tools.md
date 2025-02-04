---
tags: ["security"]
date: "May 28, 2024"
title: "Some useful security tools"
link: "note/security_tools"
description: "some useful security tools"
---

NMAP
-p port

- Service and OS detection
  - -sV (service, version) -Pn (no-ping, assume host is active)
  - -A (OS detection)

Directory and Web Object Enumeration

- Dirb
  - /usr/share/dirb/wordlists/vulns
  - dirb URL [wordlist-file default-common.txt]

- Nikto
  - Web server scanning
  - outdated components
  - misconfigurations
  - insecure and dangerous files
  - nikto -list-plugins
  - nikto -h target_host
  - -p target_port
  - -ssl
  - Output
    - -o /path
    - -Format {csv/htm/json/txt} -o /path

#### OpenVAS

- VA Scanner
- sudo gvm-feed-update
- sudo gvm-start

- CMSeek
- WPScan
- CMSmap
- Skipfish
- Joomscan
- Wapiti
- owasp-zap
- [https://sitereport.netcraft.com](https://sitereport.netcraft.com)
- [https://github.com/D35m0nd142/LFISuite/blob/master/pathtotest.txt](https://github.com/D35m0nd142/LFISuite/blob/master/pathtotest.txt)
- [https://github.com/pentestmonkey/php-reverse-shell](https://github.com/pentestmonkey/php-reverse-shell)
- [https://osintframework.com/](https://osintframework.com/)

### Test

testphp.vulnweb.com
