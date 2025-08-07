import json
import argparse
from pathlib import Path


dialect_mapping = {'ch_bs':'Basel',  'ch_zh':'Zürich', 'ch_be': 'Bern', 'ch_lu':'Luzern', "ch_sg":"Sankt Gallen", 'ch_gr':'Graubünden','ch_vs':'Wallis','ch_ag':'Aargau'}

SYSTEM_PROMPT = """You are a person speaking different Swiss German dialects in many different varieties."""
USER_PROMPT = """
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
1 sentence plain text on one line separated by a newline, no special formatting or JSON.

Begin now.
"""

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Swiss German dataset to chat-style JSONL."
    )
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        data = data[:2760]

    instruction_data = []

    for entry in data:
        # Identify dialect keys (exclude known keys)
        dialect_keys = [k for k in entry if k not in ['id', 'de', 'thema', 'code_switching']]

        for dialect_key in dialect_keys:
            dialect_label = dialect_mapping.get(dialect_key, dialect_key)  # fallback to dialect_key if not in mapping

            new_entry = {
                "id": f"{entry['id']}_{dialect_key}",  # make unique ID per dialect
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": USER_PROMPT.replace("#dialect_insert#", f"'{dialect_label}'"),
                "output": entry[dialect_key],
                "metadata": {
                    "thema": entry.get("thema"),
                    "dialect": dialect_key,
                    "source": f"original_id_{entry['id']}"
                }
            }
            instruction_data.append(new_entry)


    chat_entries = []

    for entry in instruction_data:
        chat_entry = {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": entry["system_prompt"]}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": entry["user_prompt"]}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": entry["output"]}]
                }
            ]
        }
        chat_entries.append(chat_entry)

    # Write all at once (more efficient and safer than opening inside the loop)
    with out_path.open("w", encoding="utf-8") as f:
        for line in chat_entries:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()