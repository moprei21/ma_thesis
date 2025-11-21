# Contribution of Each File to the Thesis

## Python Scripts

**conversational_client.py** – Implements inference for the GPT-based client and provides a command-line interface (CLI) to configure and execute different generation strategies. Chapter 4.3

**create_instruction_data.py** – Performs data preprocessing required for instruction tuning on the server, including formatting and structuring the training examples. Chapter 4.3.3

**data_check.py** – Generates and reports dataset statistics to support validation and quality assurance. Chapter 5.1

**inference_hf.py** – Executes inference using the Gemma-3 models through the Hugging Face interface. Chapter 5.4

**prompting.py** – Applies prompt modifications within the conversational client, adapting them to the selected generation strategy. Chapter 4.3

**reporting.py** – Manages the organization and storage of generated outputs, ensuring reproducibility and traceability of experiment results. Chapter 4.3

**text_similarity.py** – Contains the implementation of MAUVE-based text similarity experiments across model configurations. Chapter 4.4.2

**train_fasttext.py** – Trains and evaluates FastText classifiers used for downstream classification tasks. Chapter 4.4.1

## Shell Script

**run_mauve.sh** – Automates MAUVE evaluation across different generation strategies and dialect variations, ensuring consistent and reproducible experiment runs. Chapter 4.4.2
