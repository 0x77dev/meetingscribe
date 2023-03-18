from meetingscribe.misc import NamedBytesIO

import openai
from pydub import AudioSegment

import logging
from typing import Optional, Tuple

from typing import List


def split_audio_into_segments(audio: AudioSegment, segment_length: int) -> List[AudioSegment]:
    segments = []
    audio_length = len(audio)

    for i in range(0, audio_length, segment_length):
        segments.append(audio[i:i + segment_length])

    return segments


def process_segment(segment: AudioSegment, index: int, source_language: Optional[str] = None, model: str = "whisper-1") -> Tuple[int, str]:
    temp_buffer = NamedBytesIO(name=f"temp_segment_{index}.mp3")
    segment.export(temp_buffer, format="mp3")
    temp_buffer.seek(0)

    if source_language is not None:
        translated_srt = openai.Audio.translate(
            model, temp_buffer, source_language=source_language, response_format="srt")
    else:
        translated_srt = openai.Audio.transcribe(
            model, temp_buffer, response_format="srt")

    logging.debug(f"SRT chunk {index}:\n{translated_srt}")

    return index, translated_srt
