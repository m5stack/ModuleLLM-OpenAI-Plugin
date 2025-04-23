# Audio
Learn how to turn audio into text or text into audio.

# Create speech
post https://192.168.20.186:8000/v1/audio/speech

Generates audio from the input text.

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key="sk-",
    base_url="http://192.168.20.186:8000/v1"
)

speech_file_path = Path(__file__).parent / "speech.mp3"
with client.audio.speech.with_streaming_response.create(
  model="melotts-en-us",
  voice="alloy",
  input="The quick brown fox jumped over the lazy dog."
) as response:
  response.stream_to_file(speech_file_path)
```

## Request body

### input `string` <span style="color: red;">Required</span>
The text to generate audio for. The maximum length is `1024` characters.

### model `string` <span style="color: red;">Required</span>
One of the available TTS models: `melotts-zh-cn`, `melotts-en-us`.

### voice 
`Voice selection is not currently supported`

### response_format `string` Optional Defaults to mp3
The format to audio in. Supported formats are `mp3`, `opus`, `aac`, `flac`, `wav`, and `pcm`.

### speed `number` Optional Defaults to 1
The speed of the generated audio. Select a value from `0.25` to `2.0`. `1.0` is the default.

## Returns
The audio file content.