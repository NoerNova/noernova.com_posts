---
tags: ["shan-language"]
date: "May 30, 2024"
title: "Shan language in CLDR and LID"
link: "note/shan_language_cldr_and_lid"
description: "Shan language in term of Unicode encoding, CLDR and Language Identifier"
---

- Unicode Standard: Unicode is a standard for encoding, representing, and handling text in digital form. It assigns unique code points (integer values) to every character in a wide range of writing systems, including alphabets, ideograms, and symbols.

- Unicode Language Identifier (ULI): Unicode Language Identifier (ULI) is a string that identifies a language or language family in a Unicode environment. It typically consists of a two-letter ISO 639 language code, optionally followed by a two-letter ISO 3166 country code. For example, "en" represents English, "fr" represents French, "zh" represents Chinese.

  - To represent identifier for Shan language, there is no included ISO 639-1 for Shan language but three-letter ISO 639-2 code as ["Shn"](https://www.loc.gov/standards/iso639-2/php/langcodes_name.php?code_ID=399)

  - ISO 639-1 code "sh" is assigned to Serbo-Croatian language, which is a collective term for several closely related South Slavic languages spoken in the Balkans. However, the code "sh" is also used for Shan language, which is a Tai-Kadai language spoken in Myanmar, Thailand, and China.

  - ISO 639-2 There're 2 interested identifier, ["shn"](https://www.loc.gov/standards/iso639-2/php/langcodes_name.php?code_ID=436) and ["tai"](https://www.loc.gov/standards/iso639-2/php/langcodes_name.php?code_ID=436)
    - "shn" is identifier for Shan language and it's type is "Living Language"
    - "tai" is identifier for Tai language and it's type is "Collective"
    - "tai" is used for remainder group language for "Ka-Tai" language group

  - **ISO 3166-2:MM** is the entry for [Myanmar](https://en.wikipedia.org/wiki/Myanmar) in [ISO 3166-2](https://en.wikipedia.org/wiki/ISO_3166-2), part of the [ISO 3166](https://en.wikipedia.org/wiki/ISO_3166) [standard](https://en.wikipedia.org/wiki/Standardization) published by the [International Organization for Standardization](https://en.wikipedia.org/wiki/International_Organization_for_Standardization) (ISO), which defines [codes](https://en.wikipedia.org/wiki/Code) for the names of the principal [subdivisions](https://en.wikipedia.org/wiki/Country_subdivision) (e.g., [provinces](https://en.wikipedia.org/wiki/Province) or [states](https://en.wikipedia.org/wiki/State_(administrative_division))) of all [countries](https://en.wikipedia.org/wiki/Country) coded in [ISO 3166-1](https://en.wikipedia.org/wiki/ISO_3166-1).
    - As of 2020, Myanmar ISO 3166-2 codes are defined for [7 regions, 7 states, and 1 union territory](https://en.wikipedia.org/wiki/Regions_and_states_of_Burma).
    - [Reference Link](https://en.wikipedia.org/wiki/ISO_3166-2:MM)

  - ISO 15924: **ISO 15924**, **Codes for the representation of names of scripts**
    - Shan script language has 3 script system represent on [scriptsource.org](https://scriptsource.org/cms/scripts/page.php?item_id=subtag_detail&uid=eea9c6hvfb)
    - shn-Mymr [Myanmar (Burmese) [Mymr]](https://scriptsource.org/cms/scripts/page.php?item_id=script_detail&key=Mymr) - this is most detail script [Reference](https://scriptsource.org/cms/scripts/page.php?item_id=wrSys_detail&key=shn-Mymr)
    - shn-Thai [Thai [Thai]](https://scriptsource.org/cms/scripts/page.php?item_id=script_detail&key=Thai) [Reference](https://scriptsource.org/cms/scripts/page.php?item_id=subtag_detail&uid=eea9c6hvfb)
    - shn-Tale [Tai Le [Tale]](https://scriptsource.org/cms/scripts/page.php?item_id=script_detail&key=Tale) [Reference](https://scriptsource.org/cms/scripts/page.php?item_id=wrSys_detail&key=shn-Tale)
    - [Reference Link](https://scriptsource.org/cms/scripts/page.php?item_id=language_detail&key=shn)

#### Conclusion refer to Unicode Language Identifier (ULI) document

- for Shan language we can use 2 identifier
  - shn-MM for Shan language spoken in Myanmar
  - shn-TH for Shan language spoken in Thailand
    - **because those are 2 countries mostly spoken in Shan**
    - Unicode Language Identifier (ULI) is used for help identify the language used in a document or application

- [To updating CLDR (Language/Script/Region Subtags)](https://cldr.unicode.org/development/updating-codes/update-languagescriptregion-subtags)
