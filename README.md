# ma_thesis
Explanation of contribution of each file to the Thesis

## Python
conversational_client.py --> inference of GPT client, CLI to determine generation strategy
create_instruction_data.py --> pre-processing of data for instruction tuning on server
data_check.py --> copy of data statistics 
inference_hf.py --> inference for the Gemma-3 models
prompting.py -->  script used in  conversational client to modify prompts for the generation strategies
reporting.py --> script used in conversational client to specify the location of generated data
text_similarity.py --> MAUVE experiments
train_fasttext.py --> FastText classifications

## Shell Script
run_mauve.sh --> running mauve script for generation strategies and dialects



