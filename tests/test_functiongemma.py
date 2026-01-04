import json
from ollama import chat


def get_weather(city: str) -> str:
    """
    Get the current weather for a city.

    Args:
      city: The name of the city

    Returns:
      A string describing the weather
    """
    return json.dumps(
        {"city": city, "temperature": 22, "unit": "celsius", "condition": "sunny"}
    )


messages = [{"role": "user", "content": "What is the weather in Paris?"}]
print("Prompt:", messages[0]["content"])

response = chat("functiongemma", messages=messages, tools=[get_weather])

if response.message.tool_calls:
    tool = response.message.tool_calls[0]
    print(f"Calling: {tool.function.name}({tool.function.arguments})")

    result = get_weather(**tool.function.arguments)
    print(f"Result: {result}")

    messages.append(response.message)
    messages.append({"role": "tool", "content": result})

    final = chat("functiongemma", messages=messages)
    print("Response:", final.message.content)
else:
    print("Response:", response.message.content)
