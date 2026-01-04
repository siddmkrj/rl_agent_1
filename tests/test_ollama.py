from ollama import chat

stream = chat(
    model="gemma3:1b",
    messages=[
        {
            "role": "user",
            "content": "Write me a function that outputs the fibonacci sequence",
        },
    ],
    stream=True,
)
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
