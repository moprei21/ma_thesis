from transformers import pipeline
import torch

model_id_gpt = "openai/gpt-oss-120b"


SYSTEM_PROMPT_TRANSLATION = (
    "You are a Swiss German translator. Translate the following Standard "
    "German sentence into various Swiss dialects: "
    "St. Gallen (ch_sg), Bern (ch_be), Graubünden (ch_gr), Zürich (ch_zh), "
    "Valais (ch_vs), Basel (ch_bs), Aargau (ch_ag), and Lucerne (ch_lu)."
)

SYSTEM_PROMPT = """You are a person speaking different Swiss German dialects in many different varieties."""

# Load a chat-capable text model
pipe = pipeline(
    "text-generation",
    model=model_id_gpt,  # or "google/gemma-7b-it"
    device="auto",
    torch_dtype="auto",  # Use "cuda" for GPU if available
)

# Keep chat history as a list of messages
chat_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

print("Chatbot is ready! Type 'exit' to quit.")

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    # Add user message
    chat_history.append({"role": "user", "content": user_input})

    # Format into pipeline input (newer transformers versions support this directly)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id_gpt)

    # Apply chat template manually — just like during fine-tuning
    formatted_prompt = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True  # This tells the model to complete as assistant
    ).strip()

    # Now generate using the raw prompt string
    output = pipe(
        formatted_prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=1.0,
        return_full_text=False,
    )


    assistant_response = output[0]["generated_text"]
    print("\nGenerated response:")
    print(output)
    print(f"Assistant: {assistant_response}")

    # Save assistant response into history
    chat_history.append({"role": "assistant", "content": assistant_response})
