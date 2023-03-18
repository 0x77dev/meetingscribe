from meetingscribe.chat import start_chat
from meetingscribe.text import split_text_into_chunks, process_chunk
from meetingscribe.audio import process_segment, split_audio_into_segments
import openai
import concurrent.futures
import pysrt
import typer
import logging
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from tqdm import tqdm
import srt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s]: %(message)s")


app = typer.Typer(help="MeetingScribe is an AI-driven command-line tool designed to streamline your meeting experience by handling transcription, translation, and note-taking. Effortlessly generate accurate translation/transcription in English from audio file. Additionally, the tool intelligently creates meeting notes, summaries, and identifies action items.")


@app.command(help="Chat for answering questions based on the provided SRT file")
def interactive(input_srt_file: str = "output.srt", model: str = "gpt-3.5-turbo", temperature: float = 0.5):
    logging.debug(f"Starting Chat TUI using SRT file: {input_srt_file}")
    start_chat(input_srt_file, model, temperature)


def summarize_file(srt_file: str, model: str = "gpt-3.5-turbo", yes: bool = False) -> str:
    with open(srt_file, "r") as file:
        srt_data = file.read()

    srt_subs = pysrt.from_string(srt_data)
    text = " ".join([sub.text for sub in srt_subs])

    text_chunks = split_text_into_chunks(text)
    summaries = []

    logging.info(f"Splitted text into {len(text_chunks)} chunks.")

    total_tokens = sum([len(chunk.split())
                       for chunk in text_chunks]) + len(text_chunks) * 2
    tokens_per_chunk = 3500
    total_tokens += (tokens_per_chunk + 1) * 2

    cost_per_token = 0.000002
    total_cost = cost_per_token * total_tokens

    logging.warn(
        f"The estimated cost of summarizing this file is ~${total_cost:.2f}. {total_tokens} tokens, and cost per token is ${cost_per_token:.6f}.")
    if not yes:
        confirmation = input("Do you want to proceed? (y/n): ")
        if confirmation.lower() != 'y':
            print("Aborting.")
            exit()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_chunk, chunk, model)
                   for chunk in text_chunks]
        progress_bar = tqdm(as_completed(futures), total=len(
            text_chunks), desc="Summarizing chunks", unit="chunk")

        for future in progress_bar:
            summary = future.result()
            summaries.append(summary)

    combined_summary = "\n".join(summaries)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(
            description="Summarizing chunks into single answer", total=None)
        openai_response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant, that helps with meeting notes, you receive meeting conversation from TTS and should provide useful and informative notes and summary. Your task is to assemble the meeting notes from the one meeting conversation chunks."},
                {"role": "user",
                    "content": f"{combined_summary}\n---\nassemble chunks into one text in markdown format, with summary, notes, conversation breakdown, action items (if any) from conversation, etc. Remember this is a one meeting not multiple meetings."},
            ],
            temperature=0.01,
        )

    refined_summary = openai_response.choices[0].message["content"]

    return refined_summary.strip()


@app.command(help="Generate meeting summary, notes, and action items from SRT file")
def summarize(input_srt_file: str = "output.srt", output_summary_file: str = "output.md", yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"), model: str = "gpt-3.5-turbo"):
    logging.debug(f"Summarizing SRT file: {input_srt_file}")

    summary = summarize_file(input_srt_file, model, yes)

    with open(output_summary_file, "w") as file:
        file.write(summary)

    logging.info(f"Summary saved to: {output_summary_file}")


@app.command(help="Transform SRT file to TXT file")
def srt2txt(srt_file: str = "output.srt", output_file: str = "output.txt"):
    with open(srt_file) as f:
        srt_file = f.read()
    subtitles = srt.parse(srt_file)

    # Extract the text from the subtitles
    text = ""
    for subtitle in subtitles:
        text += subtitle.content.replace("\n", " ") + " "

    # Write the text to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    logging.info(f"TXT file saved to: {output_file}")


@app.command(short_help="Transcribe (and optionally translate to English) audio file into SRT file", help="Transcribe (and optionally translate) audio file into SRT file\nTranslation will translate from source language to English")
def process(input_audio_file: str, output_srt_file: str = "output.srt", source_language: Optional[str] = None, segment_length: int = 10 * 60 * 1000, yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"), model: str = "whisper-1"):
    logging.debug(f"Loading audio file {input_audio_file}")
    is_translating = source_language is not None
    action = "Translating" if is_translating else "Transcribing"

    audio = AudioSegment.from_file(input_audio_file)

    logging.info("Splitting audio file into segments")

    segments = split_audio_into_segments(audio, segment_length)

    total_duration = sum(segment.duration_seconds for segment in segments)
    avg_duration = total_duration / len(segments)

    logging.info(
        f"Average duration of audio segments: {avg_duration:.2f} seconds")
    logging.info(
        f"Total duration of audio segments: {total_duration:.2f} seconds")

    cost_per_minute = 0.006

    total_cost = cost_per_minute * total_duration / 60

    logging.warn(
        f"The estimated cost of {action.lower()} this audio file is ${total_cost:.2f}. {cost_per_minute} USD per minute of audio.")
    if not yes:
        confirmation = input("Do you want to proceed? (y/n): ")
        if confirmation.lower() != 'y':
            print("Aborting.")
            exit()

    logging.info(f"{action} audio segments")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_segment = {executor.submit(
            process_segment, segment, i, source_language, model): i for i, segment in enumerate(segments)}
        srt_results = []

        progress_bar = tqdm(concurrent.futures.as_completed(future_to_segment), total=len(
            segments), desc="Processing audio", unit="segment")

        for future in progress_bar:
            index = future_to_segment[future]
            try:
                srt_results.append(future.result())
            except Exception as exc:
                logging.error(f"Segment {index} generated an exception: {exc}")
                exit(1)

    logging.info("Joining SRT files with correct time frames")

    with open(output_srt_file, "w") as output:
        total_duration_ms = 0
        for index, srt_data in srt_results:
            srt_chunk = pysrt.from_string(srt_data)

            for subtitle in srt_chunk:
                subtitle.start.ordinal += total_duration_ms
                subtitle.end.ordinal += total_duration_ms

                output.write(str(subtitle))

            total_duration_ms += int(segments[index].duration_seconds * 1000)

    logging.info(f"Joined SRT files, and saved as {output_srt_file}")


if __name__ == "__main__":
    app()
