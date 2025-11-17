## Anonymized Supplementary Repository

This repository contains anonymized code and data for a research project on multilingual language resources and documentation bias.
It implements a population-normalized Resource Density Index (RDI), constructs catalogue-based baselines from the LRE Map and the Linguistic Data Consortium (LDC), and applies a citation-mining pipeline over the Semantic Scholar corpus to identify under-documented datasets for high-population languages.

### Repository layout

- **`data/`**
  - **`population.tsv`**: Speaker population statistics for the 200 most widely spoken languages (Ethnologue 200 list).
  - **`lre/`**
    - **`raw/`**: HTML pages scraped from the LRE Map by language.
    - **`lre_table_clean.tsv`**, **`metadata.jsonl`**: Parsed and cleaned LRE Map metadata used to compute catalogue-side dataset counts.
  - **`ldc/`**
    - **`ldc_datasets_metadata.csv`**, **`ldc_language_statistics.csv`**: LDC catalogue export and per-language statistics.
  - **`ours/`**
    - **`top100.tsv`**, **`top100.dedup.tsv`**, **`top101_200.tsv`**, **`top101_200.dedup.tsv`**, **`checked.tsv`**: Citation-mined candidate datasets, semantically deduplicated with SemHash and manually validated; these files form the final dataset inventory used in the analysis.

- **`scripts/`**
  - **`LRE_Map_fetch_data.py`**: Asynchronously scrape language-specific result pages from the LRE Map into `data/lre/raw/`.
  - **`LRE_Map_parse_metadata.py`**: Parse HTML pages into structured metadata (`metadata.jsonl`, `lre_table_clean.tsv`), including basic language and resource attributes.
  - **`UPenn_LDC_download.py`**: Query and export LDC catalogue statistics to `data/ldc/`.
  - **`fetch_semanticscholar_metadata.py`**: Resolve citation identifiers via the Semantic Scholar Graph API and attach metadata (year, venue, citation counts) to candidate dataset mentions.
  - **`deduplicate_semhash.py`**: Semantically deduplicate citation-mined rows using `semhash`, producing the `*.dedup.tsv` files.
  - **`plots/`**
    - **`plot_rdi_bins.py`**: Produce the histogram of catalogue-based RDI values.
    - **`plot_modality_sankey.py`** and **`modality_sankey_seaborn.pdf`**: Build the modality–task–language Sankey diagram summarizing dataset distributions.
    - **`task_mix_heatmap.py`**: Plot task–language distributions for the discovered datasets.

- **`pyproject.toml`**, **`uv.lock`**: Project configuration and locked dependencies (Python ≥ 3.12, managed with `uv`).
- **`secrets.env`**: Local configuration for API keys (for example, Semantic Scholar and LLM providers). This file is not tracked and must be created by the user.

### Installation

This project targets **Python 3.12+** and uses **uv** for dependency management.

```bash
cd /path/to/repo
uv sync
```

### Basic usage

Most of the raw data required to reproduce the main analyses is already included under `data/`.
To re-run the full pipeline from scratch, the overall workflow is:

1. **LRE Map scraping and parsing**
   - (Optional; the repository already ships pre-scraped HTML.)
   - Scrape raw pages:
     - `python scripts/LRE_Map_fetch_data.py ...`
   - Parse metadata:
     - `python scripts/LRE_Map_parse_metadata.py --input data/lre/raw --output data/lre/metadata.jsonl`
   - Aggregate to per-language counts using the language normalization rules described in the accompanying manuscript.

2. **LDC catalogue export**
   - Run `python scripts/UPenn_LDC_download.py` to export dataset and language statistics into `data/ldc/`.

3. **Citation-based dataset discovery**
   - Prepare a TSV of candidate citations for the target languages (top 200 by population).
   - Resolve citation metadata from Semantic Scholar:
     - `python scripts/fetch_semanticscholar_metadata.py --input <candidates.tsv> --output <with_metadata.tsv>`
   - Deduplicate semantically similar rows:
     - `python scripts/deduplicate_semhash.py --input <with_metadata.tsv> --output <dedup.tsv> --text-column Features`
   - Manually validate and label the deduplicated candidates, producing the curated files under `data/ours/` (`checked.tsv` is the final inventory of validated datasets).

4. **Recompute RDI and analysis tables**
   - Combine `population.tsv`, LRE Map counts, LDC counts, and the curated dataset inventory to recompute:
     - Catalogue-based RDI per language.
     - Citation-based RDI per language.

5. **Reproduce figures**
   - RDI distribution, modality Sankey diagram, and task–language heatmap:
     - `python scripts/plots/plot_rdi_bins.py`
     - `python scripts/plots/plot_modality_sankey.py`
     - `python scripts/plots/task_mix_heatmap.py`

### Environment and secrets

Some steps (in particular Semantic Scholar queries) may require API keys.
Secrets and private configuration are not committed; users are responsible for their own credentials and for complying with API usage policies.

### Citation

This repository is part of an anonymized supplementary package for a manuscript currently under review.
If you build on this work, please cite the final published version of the manuscript once it becomes available.
