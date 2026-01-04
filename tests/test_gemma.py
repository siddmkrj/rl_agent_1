from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(
    model="gemma3:1b",
    messages=[
        {
            "role": "user",
            "content": "hello how are you?",
        },
    ],
    stream=False,
)
print(response.message.content)