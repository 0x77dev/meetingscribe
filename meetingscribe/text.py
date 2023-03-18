from typing import List

import openai

def split_text_into_chunks(text: str, max_tokens: int = 3500) -> List[str]:
    words = text.split(" ")
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > max_tokens:
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_chunk(chunk: str, model: str) -> str:
    openai_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that helps with meeting notes, you receive meeting conversation from TTS and should provide compressed and logical summary/output from chaotic conversation, without missing details, mention time-codes"},
            {"role": "user", "content": f"{chunk}\n---\ndo not miss any important information."},
        ],
        temperature=0.05,
    )

    summary = openai_response.choices[0].message["content"]
    return summary.strip()
