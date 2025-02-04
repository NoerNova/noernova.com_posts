---
tags: ["security"]
date: "May 28, 2024"
title: "Bypass windows password with Kali linux"
link: "note/bypass_windows_password_with_kali_linux"
description: "Bypass windows password with Kali linux"
---

- First of all we need to open windows and try login even with the wrong passwords. This will enable the editing mode on the SAM (Security Accounts Manager) file,. If this step skipped then we wont be able to open the file later.
- Restart the computer and boot Kali-Linux with bootable USB
- Navigating to the Windows file system.
- In the path C:\WINDOWS\system32\config there's contained a file name SAM

```bash
  sudo chntpw -l SAM
  // this command will list all the existing users on the computer
```

```bash
  sudo chntpw -u USERNAME SAM
  
  // this will enable an editing dialog with the following options
  // 1 - Clear (blank) user password
  // 2 — Unlock and enable user account (probably locked now)
  // 3 — Promote user (make user administrator)
  // 4 — Add user to a group
  // 5 — Remove user from a group
  // q — Quit editing user, back to user select
```

- Reboot.
