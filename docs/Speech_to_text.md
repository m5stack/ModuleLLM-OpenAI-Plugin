# Create transcription
`post http://192.168.20.186:8000/v1/audio/transcriptions`
Transcribes audio into the input language.

```python
from openai import OpenAI
client = OpenAI(
    api_key="sk-",
    base_url="http://192.168.20.186:8000/v1"
)

audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-tiny",
  language="en",
  file=audio_file
)
```

## Request body

### file file Required
The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.

### model `string` <span style="color: red;">Required</span>
ID of the model to use. The options are `whisper-tiny`, `whisper-base`, and `whisper-small`.

### language `string` <span style="color: red;">Required</span>
The language of the input audio. Supplying the input language in ISO-639-1 (e.g. en) format will improve accuracy and latency.

### response_format string Optional
Defaults to json
`Currently only supported format is json.`