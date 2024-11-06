# HuAMR

You can access trained models and datasets used in this project from the following links:

- [Link to trained models](#) (Replace with actual link)
- [Link to datasets](#) (Replace with actual link)

## Setup

```bash
conda create --name my-env python=3.12.0
conda activate my-env

pip install -e .
```

### Environment Variables

Set the `HF_TOKEN` (Hugging Face token) environment variable to enable Hugging Face model loading and saving functionalities across all entry points:

```bash
export HF_TOKEN='your_hugging_face_token'
```

## Workflow Overview

Here is the recommended sequence for running each script in the project to produce synthetic AMR data, train models, and evaluate results:

1. **Generate Input Data for Synthetic AMR**:
   - Run `gen_input_for_synth_data.py` to prepare a CSV of selected sentences for synthetic data generation.
   - Alternatively, use `gen_europarl_synth_data.py` to prepare synthetic data from the Europarl dataset.
2. **Generate Synthetic AMR Data**:

   - Run `generate_synthetic_data.py` to produce AMR graphs for the prepared sentences, using an AMR model adapter.

3. **Train AMR Models**:

   - Use `train.py` or `train_s2s.py` (for Seq2Seq models) to train models on AMR datasets with configurations as needed.

4. **Generate AMR for Evaluation**:

   - After training, run `generate.py` or `generate_s2s.py` to create AMR graphs for evaluation.

5. **Evaluate Models**:
   - Run `evaluate.py` to calculate Smatch++ scores (F1, Precision, Recall) between reference and predicted AMR graphs.
   - For machine translation evaluation, use `comet_score.py` to calculate COMET scores between source and translated sentences.

## Entrypoints

### train.py

#### Running the Training Script

To run `train.py`, provide a configuration file in YAML format (e.g., `llm_config.yaml`) containing the training parameters. Execute the script from the command line as follows:

```bash
python train.py path/to/llm_config.yaml
```

#### Configuration

Refer to `llm_config.yaml` for a complete list of configurable options and their descriptions.

### train_s2s.py

#### Running the Seq2Seq Training Script

To train a sequence-to-sequence (Seq2Seq) model using `train_s2s.py`, provide a configuration file in YAML format (e.g., `s2s_train_config.yaml`) containing the necessary training and model parameters. Run the script from the command line as follows:

```bash
python train_s2s.py path/to/s2s_train_config.yaml
```

#### Configuration

Refer to `s2s_config.yaml` for a complete list of configurable options and their descriptions.

### generate.py

#### Running the Generation Script

To perform AMR generation using `generate.py`, you need a configuration file (e.g., `generate_config.yaml`) along with a specified adapter and output path. Execute the script as follows:

```bash
python generate.py path/to/generate_config.yaml path/to/adapter path/to/output_directory [batch_size]
```

- **`config_path`**: Path to the configuration file.
- **`adapter_path`**: Path to the adapter model for loading into the main model.
- **`output_path`**: Directory where the output CSV with generated AMR graphs will be saved.
- **`batch_size`** (optional): Batch size for inference (default is 32).

#### Configuration

Refer to `llm_config.yaml` for a complete list of configurable options.

#### Output

The script will save the output file (`generated.csv`) containing sentences and their corresponding generated AMR graphs to the specified output directory.

### generate_s2s.py

#### Running the Seq2Seq Generation Script

To run `generate_s2s.py`, provide a configuration file in YAML format (e.g., `s2s_config.yaml`) along with an output directory path. This script performs inference and saves the generated results.

```bash
python generate_s2s.py path/to/s2s_config.yaml path/to/output_directory
```

- **`config_path`**: Path to the configuration file (e.g., `s2s_config.yaml`).
- **`output_path`**: Directory where the output CSV with generated AMR graphs will be saved.

#### Configuration

Refer to `s2s_config.yaml` for a complete list of configurable options and their descriptions.

#### Output

The script will save the output file (`generated.csv`) containing the test set sentences and their generated AMR graphs to the specified output directory.

### generate_synthetic_data.py

#### Running the Synthetic Data Generation Script

To generate synthetic AMR data with `generate_synthetic_data.py`, provide a configuration file (e.g., `generate_synth_config.yaml`), an adapter, an input file, and an output directory. The script processes the input file in batches and appends generated AMR graphs to the output file.

```bash
python generate_synthetic_data.py path/to/generate_synth_config.yaml path/to/adapter path/to/input_file path/to/output_directory [batch_size]
```

- **`config_path`**: Path to the configuration file (e.g., `generate_synth_config.yaml`).
- **`adapter_path`**: Path to the adapter model for loading into the main model.
- **`input_file`**: Path to a CSV file containing sentences for which synthetic AMR graphs will be generated.
- **`output_path`**: Directory where the output CSV with generated synthetic AMR data will be saved.
- **`batch_size`** (optional): Batch size for inference (default is 32).

#### Input

The `input_file` should be a CSV file with a `sentence` column containing sentences for AMR generation.

#### Output

The script will generate `generated.csv` in the specified `output_path`, containing:

- **`sentence`**: Original input sentences.
- **`generated_amr`**: Corresponding generated AMR graphs.

### evaluate.py

#### Running the Evaluation Script

`evaluate.py` calculates the Smatch++ F1, Precision, and Recall scores between reference and predicted AMR graphs in a CSV file. The script processes each row, removing unwanted elements (e.g., `:wiki` attributes) from the graphs and handling errors where graphs cannot be parsed.

```bash
python evaluate.py path/to/data.csv ref_column pred_column path/to/output_directory
```

- **`data_path`**: Path to the CSV file containing the reference and predicted AMR graphs.
- **`ref_column`**: Column name in `data.csv` for the reference AMR graphs.
- **`pred_column`**: Column name in `data.csv` for the predicted AMR graphs.
- **`out_path`**: Directory where the output CSV with evaluation results will be saved.

#### Input

The `data.csv` file should contain columns for the reference and predicted AMR graphs specified by `ref_column` and `pred_column`.

#### Output

The script will generate `evaluated.csv` in the specified `out_path`, containing:

- **`smatchpp_f1`**: F1 score for each row.
- **`smatchpp_prec`**: Precision score for each row.
- **`smatchpp_rec`**: Recall score for each row.

Additionally, the script outputs the following metrics to the console:

- **Mean Smatch++ F1 score** across all entries.
- **Mean Precision** across all entries.
- **Mean Recall** across all entries.
- **Count of unparsable AMR graphs**.

### comet_score.py

#### Running the COMET Scoring Script

`comet_score.py` calculates COMET scores for evaluating machine translation quality. The script uses the `Unbabel/wmt22-cometkiwi-da` model to predict scores based on pairs of source and target sentences.

```bash
python comet_score.py path/to/input_folder
```

- **`input_folder`**: Path to a directory containing CSV files with source and target sentences.

#### Input

The `input_folder` should contain one or more CSV files. Each CSV file should include:

- **`sentence`**: Column for source sentences.
- **`hu_sentence`**: Column for the target or machine-translated sentences.

The script will load all CSV files in the folder and concatenate them for scoring.

#### Output

The script prints the COMET scores for each sentence pair to the console.

### gen_input_for_synth_data.py

#### Running the Input Generation Script

`gen_input_for_synth_data.py` prepares a dataset for synthetic AMR generation by filtering and tokenizing sentences from the "HunSum-2-abstractive" dataset. The output is a CSV file with selected sentences ready for use as input in AMR generation tasks.

```bash
python gen_input_for_synth_data.py
```

#### Requirements

Ensure the necessary datasets and tokenizers are available. This script uses the `SZTAKI-HLT/HunSum-2-abstractive` dataset, which will be automatically downloaded if not already present.

#### Output

The script generates a file named `synthetic_data_input.csv`, containing:

- **`uuid`**: Unique identifier for each entry.
- **`url`**: URL of the article source.
- **`sentence`**: First sentence of the tokenized lead section.

### gen_europarl_synth_data.py

#### Running the Synthetic Data Generation Script for Europarl

`gen_europarl_synth_data.py` generates AMR graphs from English-Hungarian sentence pairs in the Europarl dataset. This script uses an AMR parser model to create AMR graphs for English sentences, then saves the generated graphs along with the corresponding Hungarian sentences.

```bash
python gen_europarl_synth_data.py path/to/stog_model_dir path/to/europarl_folder path/to/output_directory [batch_size]
```

- **`stog_model_dir`**: Path to the directory containing the sentence-to-graph (STOG) model for AMR parsing.
- **`europarl_folder`**: Directory containing the Europarl dataset files (`europarl-v7.hu-en.en` and `europarl-v7.hu-en.hu`).
- **`output_path`**: Directory where the output file with generated AMR graphs will be saved.
- **`batch_size`** (optional): Batch size for inference (default is 32).

#### Input

The `europarl_folder` should contain:

- **`europarl-v7.hu-en.en`**: File with English sentences.
- **`europarl-v7.hu-en.hu`**: File with corresponding Hungarian sentences.

#### Output

The script will create `europarl_generated.txt` in the specified `output_path`. Each entry in this file will contain:

- **AMR Graph**: Generated from the English sentence.
- **Hungarian Sentence**: Aligned with the AMR graph for reference.

## Citation

If you use our dataset or models, please cite the following papers:

TODO
