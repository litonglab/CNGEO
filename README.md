# CNGEO

CNGEO is a Chinese-language question dataset for research on Generative Engine Optimization (GEO). This release contains 600 questions across six domains, with 100 questions in each domain.

## Dataset

| Domain | Domain code | File | Questions |
|---|---|---|---:|
| Education and Academia | `education` | `dataset/education_and_academia.csv` | 100 |
| Finance and Investment | `finance` | `dataset/finance_and_investment.csv` | 100 |
| Healthcare | `health` | `dataset/healthcare.csv` | 100 |
| Law and Government | `law_gov` | `dataset/law_and_government.csv` | 100 |
| Life, Culture, and Society | `life_society` | `dataset/life_culture_and_society.csv` | 100 |
| Technology and Digital | `tech_digital` | `dataset/technology_and_digital.csv` | 100 |
| **Total** |  |  | **600** |

## Repository Structure

```text
CNGEO/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── configs/
│   └── baidu_search.example.json
├── dataset/
│   ├── education_and_academia.csv
│   ├── finance_and_investment.csv
│   ├── healthcare.csv
│   ├── law_and_government.csv
│   ├── life_culture_and_society.csv
│   └── technology_and_digital.csv
├── prompts/
│   ├── common_rewrite_rules.txt
│   ├── source_cleaning.txt
│   └── rewrite_prompts/
├── 00_build_query_file.py
├── 01_search_sources.py
├── 02_clean_sources.py
├── 03_rewrite_sources.py
├── 04_compact_rewrites.py
├── 05_generate_answers.py
├── 06_finalize_answers.py
└── 07_compute_visibility_metrics.py
```

## CSV Schema

All files are UTF-8 encoded and use the same three-column schema:

| Column | Description |
|---|---|
| `query_id` | Unique identifier for the question |
| `domain` | Canonical domain code |
| `query_zh` | Question text in Chinese |

Each row represents one question. The header row is not included in the question count.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Experiment Pipeline

### 1. Build the combined experiment input

The released questions remain split by domain. Build the combined file expected by the experiment scripts with:

```bash
python 00_build_query_file.py
```

This creates `data/queries.csv`.

### 2. Retrieve and crawl source pages

```bash
python 01_search_sources.py \
  --config configs/baidu_search.example.json
```

If a Baidu session cookie is required in your environment, provide it through `BAIDU_COOKIE`. Do not place it directly in a tracked configuration file.

### 3. Clean crawled webpage bodies

The raw crawler output can contain navigation labels, advertisements, recommendation modules, login prompts, footers, repeated text, and other page-layout noise. The cleaning step uses the prompt in `prompts/source_cleaning.txt` and calls `deepseek-v4-pro` by default.

```bash
export BAILIAN_API_KEY="your-api-key"

python 02_clean_sources.py --dry-run
python 02_clean_sources.py
```

The script reads `data/search_results.jsonl` and writes cleaned source bodies to `data/search_results_cleaned.jsonl`. It preserves substantive facts, lists, conditions, source information, and risk disclosures while removing page-level noise. Failed or invalid cleaning results are logged but are not written to the cleaned dataset, so they can be retried safely.

### 4. Generate GEO rewrites

Set the API key through the environment and review a dry run before starting a full experiment:

```bash
python 03_rewrite_sources.py --dry-run
python 03_rewrite_sources.py
python 04_compact_rewrites.py
```

The Chinese cleaning and rewriting prompt files under `prompts/` are part of the experimental design and are intentionally included in Chinese.

### 5. Generate fixed-source answers

The included platform configuration uses environment variables only:

- `BAILIAN_API_KEY` for DeepSeek and Qwen through Bailian
- `ARK_API_KEY` for Doubao through Ark
- `QIANFAN_API_KEY` for ERNIE through Qianfan

```bash
python 05_generate_answers.py \
  --platforms DP,TYQW,DB,WXY \
  --repeats 5 \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-tokens 8192 \
  --dry-run

python 05_generate_answers.py \
  --platforms DP,TYQW,DB,WXY \
  --repeats 5 \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-tokens 8192

python 06_finalize_answers.py \
  --platforms DP,TYQW,DB,WXY \
  --repeats 5
```

Every answer row contains a `run_id` from 1 through 5. Resume detection includes `run_id`, so an interrupted experiment generates only the missing repetitions.

The complete main experiment contains:

- 12,000 baseline answers: 600 questions x 4 platforms x 5 runs
- 480,000 strategy answers: 600 questions x 5 target ranks x 8 strategies x 4 platforms x 5 runs
- 492,000 answer generations in total

API usage may incur substantial cost. Always inspect the dry-run task count and confirm model availability, pricing, rate limits, and provider terms before running the full dataset.

### 6. Compute visibility metrics

```bash
python 07_compute_visibility_metrics.py \
  --platforms DP,TYQW,DB,WXY \
  --repeats 5 \
  --bootstrap-samples 2000 \
  --bootstrap-domain-size 100 \
  --bootstrap-seed 20260727
```

The statistical order is fixed:

1. Compute CS, PAWS, and composite visibility for every individual answer generation.
2. Average the five baseline runs and five strategy runs within each question, platform, strategy, and target-rank condition.
3. Compute absolute gains and pooled relative gains from the averaged metrics. The pipeline does not average five precomputed relative-gain ratios.
4. Estimate confidence intervals with a question-cluster Bootstrap. Each replicate samples 100 questions with replacement within each of the six domains and keeps all platforms, strategies, target ranks, and averaged baseline/strategy results for a sampled question together.

The script writes:

- `data/metrics_runs.jsonl`: metrics for every individual generation
- `data/metrics.jsonl`: condition-level metrics after averaging the five runs
- `outputs/reports/`: aggregate reports with 2,000-replicate Bootstrap confidence intervals

## Quick Dataset Usage

```python
from pathlib import Path

import pandas as pd

files = sorted(Path("dataset").glob("*.csv"))
queries = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

print(queries.shape)                # (600, 3)
print(queries["domain"].value_counts())
```

## Security

- No API keys, cookies, user paths, or account identifiers are stored in the public code.
- Credentials are read only from environment variables.
- `.env`, generated data, logs, and local credential-bearing configuration are excluded by `.gitignore`.
- Never commit provider credentials or authenticated browser cookies.

## Data Integrity

- 600 questions in total
- 100 questions per domain
- 600 unique `query_id` values
- No missing values in the three required columns

## Citation

If you use CNGEO in your research, please cite our paper:

```bibtex
@article{li2026cngeo,
  title   = {Chinese Generative Engine Optimization},
  author  = {Li, Tong and Wu, Rongbang and Li, Qinghao and Jia, Gui and Gao, Xiangyu and Fan, Ju and Xu, Ke},
  journal = {Journal of Software},
  year    = {2026},
  note    = {Accepted, to appear}
}
```

The citation will be updated with volume, issue, pages, and DOI once the final publication information becomes available.

## License

The source code, prompts, and dataset in this repository are released under the MIT License. See [LICENSE](LICENSE) for details.

## Disclaimer

The questions are provided for research and evaluation purposes. They do not constitute medical, legal, financial, or other professional advice.
