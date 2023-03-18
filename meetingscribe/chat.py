from meetingscribe.text import split_text_into_chunks


import openai
import pysrt
from rich import print
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from tqdm import tqdm


import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from meetingscribe.text import process_chunk

def start_chat(srt_file, model, temperature: float = 0.5):
    with open(srt_file, "r") as file:
        srt_data = file.read()

    srt_subs = pysrt.from_string(srt_data)
    text = " ".join([sub.text for sub in srt_subs])

    text_chunks = split_text_into_chunks(text)
    compressed_chunks = []

    logging.info(f"Splitted text into {len(text_chunks)} chunks.")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_chunk, chunk, model)
                   for chunk in text_chunks]
        progress_bar = tqdm(as_completed(futures), total=len(
            text_chunks), desc="Compressing chunks", unit="chunk")

        for future in progress_bar:
            compressed_chunk = future.result()
            compressed_chunks.append(compressed_chunk)

    compressed_text = " ".join(compressed_chunks)

    default_question = "What is it about?"

    while True:
        question = Prompt.ask("Question", default=default_question)
        default_question = None

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the given transcript. The transcript has been compressed for better understanding."},
                {"role": "user", "content": f"{compressed_text}\n---\nAnswer the following question: {question}."},
            ],
            temperature=temperature,
        )

        answer = response.choices[0].message["content"]

        print(Panel(Text(answer), title=question))
