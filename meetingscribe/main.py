import openai
import concurrent.futures
import pysrt
import typer
import logging
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
from tqdm import tqdm
import srt
import io

app = typer.Typer(help="MeetingScribe is an AI-driven command-line tool designed to streamline your meeting experience by handling transcription, translation, and note-taking. Effortlessly generate accurate translation/transcription in English from audio file. Additionally, the tool intelligently creates meeting notes, summaries, and identifies action items.")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s]: %(message)s")


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


def summarize_file(srt_file: str, model: str = "gpt-3.5-turbo") -> str:
    with open(srt_file, "r") as file:
        srt_data = file.read()

    srt_subs = pysrt.from_string(srt_data)
    text = " ".join([sub.text for sub in srt_subs])

    text_chunks = split_text_into_chunks(text)
    summaries = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_chunk, chunk, model)
                   for chunk in text_chunks]
        progress_bar = tqdm(as_completed(futures), total=len(
            text_chunks), desc="Summarizing", unit="chunk")

        for future in progress_bar:
            summary = future.result()
            summaries.append(summary)

    combined_summary = "\n".join(summaries)

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
def summarize(input_srt_file: str = "output.srt", output_summary_file: str = "output.md"):
    logging.debug(f"Summarizing SRT file: {input_srt_file}")

    summary = summarize_file(input_srt_file)

    with open(output_summary_file, "w") as file:
        file.write(summary)

    logging.info(f"Summary saved to: {output_summary_file}")


def split_audio_into_segments(audio: AudioSegment, segment_length: int) -> List[AudioSegment]:
    segments = []
    audio_length = len(audio)

    for i in range(0, audio_length, segment_length):
        segments.append(audio[i:i + segment_length])

    return segments

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

@app.command(short_help="Transcribe (and optionally translate to English) audio file into SRT file", help="Transcribe (and optionally translate) audio file into SRT file\nTranslation will translate from source language to English")
def process(input_audio_file: str, output_srt_file: str = "output.srt", source_language: Optional[str] = None, segment_length: int = 10 * 60 * 1000):
    logging.debug(f"Loading audio file {input_audio_file}")
    is_transcribing = source_language is not None
    action = "Transcribing" if is_transcribing else "Translating"

    audio = AudioSegment.from_file(input_audio_file)

    logging.info("Splitting audio file into segments")

    segments = split_audio_into_segments(audio, segment_length)

    total_duration = sum(segment.duration_seconds for segment in segments)
    avg_duration = total_duration / len(segments)

    logging.info(
        f"Average duration of audio segments: {avg_duration:.2f} seconds")

    logging.info(f"{action} audio segments")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_segment = {executor.submit(
            process_segment, segment, i, source_language): i for i, segment in enumerate(segments)}
        srt_results = []

        progress_bar = tqdm(concurrent.futures.as_completed(future_to_segment), total=len(
            segments), desc="Processing audio", unit="segment")

        for future in progress_bar:
            index = future_to_segment[future]
            try:
                srt_results.append(future.result())
            except Exception as exc:
                logging.error(f"Segment {index} generated an exception: {exc}")

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


class NamedBytesIO(io.BytesIO):
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop("name", "buffer")
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return self._name

def process_segment(segment: AudioSegment, index: int, source_language: Optional[str] = None) -> Tuple[int, str]:
    temp_buffer = NamedBytesIO(name=f"temp_segment_{index}.mp3")
    segment.export(temp_buffer, format="mp3")
    temp_buffer.seek(0)

    if source_language is not None:
        translated_srt = openai.Audio.translate(
            "whisper-1", temp_buffer, source_language=source_language, response_format="srt")
    else:
        translated_srt = openai.Audio.transcribe(
            "whisper-1", temp_buffer, response_format="srt")

    logging.debug(f"SRT chunk {index}:\n{translated_srt}")

    return index, translated_srt


if __name__ == "__main__":
    app()
