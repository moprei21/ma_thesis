from dotenv import load_dotenv
import openai
import os



class GPTConversationalClient:
    def __init__(self, model_name="gpt-4o", temperature=1.0):
        load_dotenv(override=True)
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
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
                messages=self.conversation,
                **({"response_format": response_format} if response_format else {})
            )

            finish_reason = response.choices[0].finish_reason
            message_content = response.choices[0].message.content

            if finish_reason == "stop":
                self.add_message("assistant", message_content)
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

    # Set up the initial conversation
    conversation = [
        {"role": "system", "content": "Du bisch en Expert im rede vo Schwiizerdütsch. Bitte antworte mir nur uf Schwiizerdütsch und uf kei anderi Sprach."},
        {"role": "user", "content": "Wie warm isches im Sommer in Züri?"}
    ]

    client.set_conversation(conversation)
    response = client.query()
    print(response)

if __name__ == "__main__":
    main()
