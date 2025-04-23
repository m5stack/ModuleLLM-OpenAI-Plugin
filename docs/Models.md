# Models
List and describe the various models available in the API. You can refer to the Models documentation to understand what models are available and the differences between them.

# List models
`get http://192.168.20.186:8000/v1/models`

Lists the currently available models, and provides basic information about each one such as the owner and availability.

```python
from openai import OpenAI
client = OpenAI(
    api_key="sk-",
    base_url="http://192.168.20.186:8000/v1"
)

client.models.list()
print(client.models.list())
```

## Returns
A list of model objects.