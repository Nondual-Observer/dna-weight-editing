# Third-Party Licenses and Copyright Notices

Original author code, the `sgne_cc_dna_delta.npz` weight diff patch, author-created prompts (`MultiSpeech-64` and the 7 long-form prose items in `FreshBlind56`), author-generated `.jsonl` model outputs, figures, and manuscript text in this repository are licensed under the **Apache License, Version 2.0** (see [`LICENSE`](LICENSE)).

Bundled or adapted third-party code snippets, benchmark datasets, and external reference logs retain their original licenses and copyright notices set forth below.

---

## 1. `RikiyaT/lm_understand-logs` (Code: MIT License · Logs/Data: CC BY 4.0)

* **Upstream repository:** https://github.com/RikiyaT/lm_understand-logs (`LICENSE`: https://github.com/RikiyaT/lm_understand-logs/blob/main/LICENSE)
* **Checkpoint card:** https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
* **Scope in this repository:**
  1. `traces/mit_official_math500_ref/mit_unedited_merged.json`, `traces/mit_official_math500_ref/mit_p50c_merged.json`, and the derived 500-item summary `traces/mit_official_math500_ref/mit_p50c_official_500_per_problem.json` are governed by **Creative Commons Attribution 4.0 International (`CC BY 4.0`)**: https://creativecommons.org/licenses/by/4.0/.
  2. Answer-extraction helper functions adapted in `scripts/verify_paper_metrics.py` (`mit_last_boxed`, `mit_remove_boxes`, `mit_extract_boxed`, `mit_last_answer_part`, `mit_lenient_correct`, adapted from `exp8_math/code/mathqa.py` and `exp8_math/code/evaluate.py`) are governed by the **MIT License**:

```text
MIT License

Copyright (c) 2026 the authors of "LLMs can understand and write neural network weights"

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Hendrycks `MATH` Dataset (`hendrycks/math` — MIT License)

* **Upstream repository:** https://github.com/hendrycks/math (`LICENSE`: https://github.com/hendrycks/math/blob/main/LICENSE)
* **500-problem test split reference:** OpenAI PRM800K / https://huggingface.co/datasets/HuggingFaceH4/MATH-500
* **Scope in this repository:** Problem statements, gold answers, subject/level metadata in `suites/suite_math500_full.json` (500 items), stratified 100-item subset in `suites/suite_math500_strat100.json`, 49 untouched math holdout items in `suites/fresh_blind56_32k_suite.json`, and `\boxed{}` brace-matching logic originally from `hendrycks/math`.

```text
MIT License

Copyright (c) 2021 Dan Hendrycks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 3. Multilingual Grade School Math (`MGSM` — CC BY-SA 4.0)

* **Upstream dataset:** https://huggingface.co/datasets/juletxara/mgsm (Freda Shi et al., *"Language Models are Multilingual Chain-of-Thought Reasoners"*, Google Research)
* **License:** Creative Commons Attribution-ShareAlike 4.0 International (`CC BY-SA 4.0`) — https://creativecommons.org/licenses/by-sa/4.0/legalcode
* **Scope and modifications in this repository:** `suites/suite_unseen_mgsm240_bbh160.json` includes an adapted subset of 240 `MGSM` records (40 base grade-school math problems across 6 languages: `ru, de, fr, es, zh, ja`, formatted into JSON evaluation entries with `\boxed{}` prompt instructions) together with prompt echoes in `traces/mgsm240_bbh160_*.jsonl`. Those `MGSM` portions remain licensed under **CC BY-SA 4.0**.

---

## 4. `BIG-Bench Hard` (`suzgunmirac/BIG-Bench-Hard` — MIT License)

* **Upstream repository:** https://github.com/suzgunmirac/BIG-Bench-Hard (`LICENSE`: https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/LICENSE)
* **Scope in this repository:** 160 selected items (20 items across each of 8 tasks: `boolean_expressions`, `causal_judgement`, `date_understanding`, `disambiguation_qa`, `formal_fallacies`, `logical_deduction_five_objects`, `navigate`, `web_of_lies`) formatted in `suites/suite_unseen_mgsm240_bbh160.json`.

```text
MIT License

Copyright (c) 2022 suzgunmirac

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 5. OpenAI `HumanEval` (`openai/human-eval` — MIT License)

* **Upstream repository:** https://github.com/openai/human-eval (`LICENSE`: https://github.com/openai/human-eval/blob/master/LICENSE)
* **Scope in this repository:** All 164 canonical Python programming problems and unit tests in `suites/suite_unseen_humaneval164.json`.

```text
The MIT License

Copyright (c) OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## 6. Base Model Reference (`Qwen/Qwen3.5-0.8B` — Apache License 2.0)

* **Upstream model:** https://huggingface.co/Qwen/Qwen3.5-0.8B
* **Copyright:** Copyright Alibaba Cloud Qwen Team
* **License:** Apache License, Version 2.0 (see [`LICENSE`](LICENSE))
