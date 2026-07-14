# Open-Source Offline-Deployable Retrieval-Augmented Large Language Model for Assisting Pancreatic Cancer Staging
In this project, a retrieval-augmented generation (RAG) framework is used to enable large language models (LLMs) to perform guideline-based staging of pancreatic cancer CT findings. This repository contains all experimental code. If you are running the code for the first time, please begin with the [Preparation](#preparation) section.

- If you only want to inspect the results from our experiments as they were at the time of the experiment, please check out the `v1.0.0-with-results` tag or visit the following URL: https://github.com/mohehe1234/local-rag/tree/v1.0.0-with-results
- If you want to repeat the experiments, follow the instructions at [Repeating Experiments](#repeating-experiments) section.
- If you want to run the experiments with your own data, follow the instructions at [Adding Custom Files](#adding-custom-files) section.

## Citation
If you use this work, please cite it.
```bibtex
@article {Johno2026,
	author = {Johno, Hisashi and Amakawa, Akitomo and Komaba, Atsushi and Tozuka, Ryota and Johno, Yuki and Sato, Junichi and Yoshimura, Kentaro and Nakamoto, Kazunori and Ichikawa, Shintaro},
	title = {Open-Source Offline-Deployable Retrieval-Augmented Large Language Model for Assisting Pancreatic Cancer Staging},
	journal = {Jpn J Radiol},
	year = {2026},
    month = jul,
    note = {doi: 10.1007/s11604-026-02049-8}
}
```

## Preparation
Follow the steps below to set up and run the project locally.
### 1. Install Ollama

Download and install Ollama from the https://ollama.ai. 

### 2. Clone this repository
```
git clone URL
```
### 3. Move to the project directory
```
cd this_repository_name
```
### 4. Prepare models and API key
First, pull the required Ollama model:

```
ollama pull model_name
```
Next, if you want to use OpenAI models, set up your OpenAI API key. Create an `api_key.txt` file in the root directory and enter your API key in it. You can obtain your key from https://platform.openai.com/account/api-keys

### 5. Install dependencies
You can install the dependencies using either `uv` or `pip`, following the commands below.

**Before installation**, open the pyproject.toml file and replace all occurrences of `cu126` with your own CUDA version.
Then, update `cuXXX` in the command below to match your CUDA version before running it. If you are using CPU instead of CUDA, replace `cuXXX` with cpu before running the command.

**If using `pip`**
- macOS or Linux

    ```
    python -m venv venv
    source venv/bin/activate
    pip install .[cuXXX]
    ```
- Windows

    ```
    python -m venv venv
    venv\Scripts\activate
    pip install .[cuXXX]
    ```

**If using `uv`**

```
uv sync --extra cuXXX
```

## Running the LLM for Staging
You can run the program by specifying the reliable external knowledge (REK) and the file containing the CT findings.

### Basic command
```
python main.py [options]
```
### Arguments
| Argument | Required | Description |
| --- | --- | --- |
| `input` | Yes | Path to the input JSON file |
| `output` | Yes | Directory to save the results |
| `--rek REK` | Yes | Path to the reliable external knowledge (REK) |
| `--rag` | | Enable retrieval-augmented generation (RAG) |
| `--model-openai MODEL_OPENAI` | Conditionally* | Name of the OpenAI model to use |
| `--model-ollama MODEL_OLLAMA` | Conditionally* | Name of the Ollama model to use |
| `--api-key API_KEY` | | Path to the file containing the API key |
| `--case-min CASE_MIN` | | Minimum Case_number to include |
| `--case-max CASE_MAX` | | Maximum Case_number to include |
| `-h, --help` | | Show help message and exit |


Please refer to the `scripts` directory for example runs.
If you want to provide your own new REK or CT findings files, follow the instructions in [Adding Custom Files](#adding-custom-files) section.

## Repeating Experiments
**If using `pip`**
- macOS or Linux

    ```
    chmod u+x scripts/run_pip.sh
    ./scripts/run_pip.sh
    ```
- Windows

    ```
    .\scripts\run_pip.ps1
    ```

**If using `uv`**

Please replace `run_pip` with `run_uv` in the commands above and then execute them.

## Adding Custom Files
This section explains how to replace the REK and CT findings files and execute the code with your own files.
### Replacing the REK
Create a new file in Markdown format based on `REK.txt`. When running the code, refer to [Running the LLM for Staging](#running-the-llm-for-staging) section and specify your file using the `--rek` argument.
### Replacing CT findings
Prepare a JSON file following the schema below. When running the code, refer to [Running the LLM for Staging](#running-the-llm-for-staging) section and specify your file using the `input` argument.
```
{ 
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "Case_number": { "type": "integer" },
            "Findings": { "type": "string" },
            "T": { "enum": ["T0", "Tis", "T1a", "T1b", "T1c", "T2", "T3", "T4"] },
            "CH": { "enum": ["CH0", "CH1"] },
            "DU": { "enum": ["DU0", "DU1"] },
            "S": { "enum": ["S0", "S1"] },
            "RP": { "enum": ["RP0", "RP1"] },
            "PV": { "enum": ["PV0", "PV1"] },
            "A": { "enum": ["A0", "A1"] },
            "PL": { "enum": ["PL0", "PL1"] },
            "OO": { "enum": ["OO0", "OO1"] },
            "N": { "enum": ["N0", "N1a", "N1b"] },
            "M": { "enum": ["M0", "M1"] },
            "Resectability": { "enum": ["R", "BR", "UR"] }
        },
        "required": [
            "Case_number",
            "Findings",
            "T",
            "CH",
            "DU",
            "S",
            "RP",
            "PV",
            "A",
            "PL",
            "OO",
            "N",
            "M",
            "Resectability"
        ]
    }
}
```

## License

This repository contains:
- Code and scripts under the MIT License (`LICENSE`).
- Adapted textual data from an external guideline licensed under
  Creative Commons Attribution 4.0 International (CC BY 4.0).

### CC BY 4.0 Attribution
The following file is licensed under CC BY 4.0:
- `REK.txt`

This file include reformatted and chunked versions of the original article text.
Changes were made to ensure experimental transparency and reproducibility.

**Original guideline:**
- **Title**: Japanese classification of pancreatic carcinoma by the Japan Pancreas Society: Eighth edition
- **Author(s)**: Masaharu Ishida, Tsutomu Fujii, Masashi Kishiwada, Kazuto Shibuya, Sohei Satoi, Makoto Ueno, Kohei Nakata, Shigetsugu Takano, Katsunori Uchida, Nobuyuki Ohike, Yohei Masugi, Toru Furukawa, Kenichi Hirabayashi, Noriyoshi Fukushima, Shuang-Qin Yi, Hiroyuki Isayama, Takao Itoi, Takao Ohtsuka, Takuji Okusaka, Dai Inoue, Hirohisa Kitagawa, Kyoichi Takaori, Masaji Tani, Yuichi Nagakawa, Hideyuki Yoshitomi, Michiaki Unno, Yoshifumi Takeyama
- **Journal**: Journal of Hepato-Biliary-Pancreatic Sciences
- **Publisher**: Wiley
- **Source**: https://onlinelibrary.wiley.com/doi/10.1002/jhbp.12056
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Copyright © The Author(s).