# Chat Completions
The Chat Completions API endpoint will generate a model response from a list of messages comprising a conversation.

# Create chat completion
`post https://api.openai.com/v1/chat/completions`

```python
from openai import OpenAI
openai = OpenAI(
    api_key="sk-",
    base_url="http://192.168.20.186:8000/v1"
)

completion = client.chat.completions.create(
  model="qwen2.5-0.5B-p256-ax630c",
  messages=[
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

## Request body

### messages `array` <span style="color: red;">Required</span>
A list of messages comprising the conversation so far. Depending on the model you use, different message types (modalities) are supported, like text, images, and audio.

### model `string` <span style="color: red;">Required</span>
Model ID used to generate the response, like `qwen2.5-0.5B-p256-ax630c` or `deepseek-r1-1.5B-p256-ax630c`. StackFlow offers a wide range of models with different capabilities, performance characteristics. Refer to the model Docs to browse and compare available models.

### audio
`Audio output is not currently supported`

### function_call
`function_call is not currently supported`

### max_tokens `integer` Optional
The maximum number of tokens that can be generated in the chat completion.

### response_format `object` Optional
An object specifying the format that the model must output.
`Currently only supported format is json_object.`