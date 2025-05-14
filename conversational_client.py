from dotenv import load_dotenv
import openai
import os
from reporting import LLMReporter



class GPTConversationalClient:
    def __init__(self, model_name="gpt-4.1", temperature=1.0):
        load_dotenv(override=True)
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.top_p = 0.9
        self.temperature = temperature
        self.max_tokens = 200
        self.conversation = []

    def set_conversation(self, conversation):
        """Initialize the conversation history"""
        self.conversation = conversation

    def add_message(self, role, content):
        """Add a message to the conversation"""
        self.conversation.append({"role": role, "content": content})

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
    client = GPTConversationalClient()
    reporter = LLMReporter(base_dir="synthetic_data", use_timestamps=True)
    system_prompt = """You are a data generation model specialized in producing synthetic Swiss German (Schweizerdeutsch) training data for natural language processing tasks. All output must be in spoken Swiss German, not High German or English. Use casual, natural phrasing from everyday speech in German-speaking Switzerland.

Never explain, translate, or break character. Only generate raw training data. Use Zürich dialect unless otherwise instructed.

Output realistic, as spoken by Swiss locals. Stay strictly in Swiss German.
"""

    prompt = """You are a data generation model specialized in producing synthetic Swiss German (Schweizerdeutsch) training data for natural language processing tasks. Your responses must be written only in spoken Swiss German, not High German or English. Always use casual, natural phrasing as heard in everyday speech in the German-speaking parts of Switzerland.

Only output raw training data — do not explain, translate, or include any meta-commentary. Do not switch languages. Maintain a consistent dialect (Zürich-style unless otherwise specified).

Generate 10 sentences in Swiss German.
Each example should be realistic, diverse in phrasing, and aligned with how real people talk in daily life.

Output format:
plain text on one line, no special formatting or JSON.

Begin now.
"""
    # Set up the initial conversation
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    client.set_conversation(conversation)
    reporter.register_file("response", filename="all.txt")
    for i in range(1):
        response = client.query()
        reporter.write("response", response, newline=True)

if __name__ == "__main__":
    main()
