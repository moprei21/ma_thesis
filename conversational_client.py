from dotenv import load_dotenv
import openai
import os
from reporting import LLMReporter
from prompting import PromptingEngine
from data import SwissDialDataset
import argparse


PROMPT_BASE = """
Only output raw training data — do not explain, translate, or include any meta-commentary. Do not switch languages. Maintain a consistent dialect.

Generate one sentence in Swiss German using the #dialect_insert# 
Each example should be realistic, diverse in phrasing, and aligned with how real people talk in daily life.

Follow these steps:
1. Generate a sentence in Swiss German using the #dialect_insert# 
2. Check if the sentence is indeed in the #dialect_insert#; if not, generate a new one.
3. Compare the sentence with other dialects, and if it's too similar, generate a new one.

Do these steps one after the other.

Only output raw training data — do not explain, translate, or include any meta-commentary. Do not switch languages. Maintain a consistent dialect.

Output format:
10 sentences plain text on one line separated by a newline, no special formatting or JSON.

Begin now.
"""
PROMPT_BASE_ZèRICH = """
Only output raw training data — do not explain, translate, or include any meta-commentary. Do not switch languages. Maintain a consistent dialect.

Generate one sentence in Swiss German using the Zürich dialect 
Each example should be realistic, diverse in phrasing, and aligned with how real people talk in daily life.

Follow these steps:
1. Generate a sentence in Swiss German using the Zürich dialect 
2. Check if the sentence is indeed in the #dialect_insert#; if not, generate a new one.
3. Compare the sentence with other dialects, and if it's too similar, generate a new one.

Do these steps one after the other.

Only output raw training data — do not explain, translate, or include any meta-commentary. Do not switch languages. Maintain a consistent dialect.

Output format:
10 sentence plain text on one line separated by a newline, no special formatting or JSON.

Begin now.
"""

system_prompt = """You are a data generation model specialized in producing synthetic Swiss German (Schweizerdeutsch) training data for natural language processing tasks. """

class GPTConversationalClient:
    def __init__(self, model_name="gpt-4.1", temperature=2.0):
        load_dotenv(override=True)
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.top_p = 0.9
        self.temperature = temperature
        self.max_tokens = 300
        self.conversation = []

    def set_conversation(self, conversation):
        """Initialize the conversation history"""
        self.conversation = conversation

    def query(self, response_format=None):
        """Send the conversation to the API and return the assistant's response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                messages=self.conversation,
                max_tokens=self.max_tokens,
                **({"response_format": response_format} if response_format else {})
            )
            finish_reason = response.choices[0].finish_reason
            message_content = response.choices[0].message.content

            if finish_reason == "stop":
                return message_content
            elif finish_reason == "length":
                print("Warning: Response was truncated due to length limits.")
                return message_content
            elif finish_reason == "content_filter":
                raise ValueError("Response was filtered due to content restrictions.")
            else:
                raise ValueError(f"Unexpected finish reason: {finish_reason}")
        except Exception as e:
            raise RuntimeError(f"API query failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="A simple CLI tool to process files."
    )

    # Positional argument (required)
    parser.add_argument(
        "-d","--dialect",
        type=str,
        help="Dialect to use for processing (e.g., Basel, Zürich, Bern)"
    )

    # Optional argument with a value
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to the output file"
    )

    # Optional flag (boolean)
    parser.add_argument(
        "-s", "--strategy",
        type=str,
        choices=["few-shot", "few-shot-conversational", 'zero-shot'],
        help="Strategy used in prompting"
    )
    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate (default: 10)"
    )


    # Parse arguments
    args = parser.parse_args()
    # init client
    client = GPTConversationalClient(model_name="gpt-4.1", temperature=2.0)

    # init reporter
    reporter = LLMReporter(base_dir=args.output, use_timestamps=True)

    dialect = args.dialect
    # few-shot dataset
    swiss_dial_dataset = SwissDialDataset(file_path="data/swissdial/sentences_ch_de_numerics.json", dialect=dialect)
    df = swiss_dial_dataset.load_from_json_file()
    swiss_dial_dataset.dialect = dialect
    
    # Create the Hugging Face dataset
    dataset = swiss_dial_dataset.create_huggingface_dataset(df)


    sampled_data = swiss_dial_dataset.sample_dataset(dataset['train'], num_samples=10)
    strategy = args.strategy  # or "few-shot" based on your requirement
    # init prompt

    if strategy == "few-shot-conversational":
        prompt = PromptingEngine(strategy, sampled_data)
        prompt.set_system_prompt(system_prompt)
        few_shot_prompt = "Please generate a sentence in Swiss German using the #dialect_insert#"
        prompt.generate_prompt(dialect,few_shot_prompt)

    elif strategy == "few-shot":
        prompt = PromptingEngine('few-shot', sampled_data)
        prompt.set_system_prompt(system_prompt)
        prompt.generate_prompt(dialect,PROMPT_BASE)

    else:
        prompt = PromptingEngine(strategy=strategy)
        prompt.set_system_prompt(system_prompt)
        prompt.generate_prompt(dialect,PROMPT_BASE)

    client.set_conversation(prompt.conversation)
    reporter.register_file("response", filename=f"{dialect.lower()}.txt")
    for i in range(args.num_samples):
        response = client.query()
        reporter.write("response", response, newline=True)

if __name__ == "__main__":
    main()
