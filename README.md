# MeetingScribe

MeetingScribe is an AI-driven command-line tool designed to streamline your meeting experience by handling transcription, translation, and note-taking. Effortlessly generate accurate translation/transcription in English from audio file. Additionally, the tool intelligently creates meeting notes, summaries, and identifies action items.

**Prerequisites**:

1. Specify [OpenAI API Key](https://platform.openai.com/account/api-keys):

```console
export OPENAI_API_KEY=<your-openai-api-key>
```

2. Install [FFmpeg](https://ffmpeg.org/download.html)

**Installation**:

<details>

<summary>
using <code>pip</code>
</summary>

```console
pip install meetingscribe
```

</details>

<details>

<summary>
using <code>docker</code>
</summary>

```console
export OPENAI_API_KEY=<your-openai-api-key>

docker run -it -e OPENAI_API_KEY=$OPENAI_API_KEY ghcr.io/0x77dev/meetingscribe --help
```

</details>

**Usage**:

```console
meeting [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `interactive`: Chat for answering questions based on the...
* `process`: Transcribe (and optionally translate to English) audio file into SRT file
* `srt2txt`: Transform SRT file to TXT file
* `summarize`: Generate meeting summary, notes, and...

## `meeting interactive`

Chat for answering questions based on the provided SRT file

**Usage**:

```console
meeting interactive [OPTIONS]
```

**Options**:

* `--input-srt-file TEXT`: [default: output.srt]
* `--model TEXT`: [default: gpt-3.5-turbo]
* `--temperature FLOAT`: [default: 0.5]
* `--help`: Show this message and exit.

## `meeting process`

Transcribe (and optionally translate) audio file into SRT file
Translation will translate from source language to English

**Usage**:

```console
meeting process [OPTIONS] INPUT_AUDIO_FILE
```

**Arguments**:

* `INPUT_AUDIO_FILE`: [required]

**Options**:

* `--output-srt-file TEXT`: [default: output.srt]
* `--source-language TEXT`
* `--segment-length INTEGER`: [default: 600000]
* `-y, --yes`: Skip confirmation prompt
* `--model TEXT`: [default: whisper-1]
* `--help`: Show this message and exit.

## `meeting srt2txt`

Transform SRT file to TXT file

**Usage**:

```console
meeting srt2txt [OPTIONS]
```

**Options**:

* `--srt-file TEXT`: [default: output.srt]
* `--output-file TEXT`: [default: output.txt]
* `--help`: Show this message and exit.

## `meeting summarize`

Generate meeting summary, notes, and action items from SRT file

**Usage**:

```console
meeting summarize [OPTIONS]
```

**Options**:

* `--input-srt-file TEXT`: [default: output.srt]
* `--output-summary-file TEXT`: [default: output.md]
* `-y, --yes`: Skip confirmation prompt
* `--model TEXT`: [default: gpt-3.5-turbo]
* `--help`: Show this message and exit.
