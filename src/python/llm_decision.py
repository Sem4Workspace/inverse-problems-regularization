import openai
import json

def query_llm(prompt, model="gpt-4"):
    """
    Queries the LLM with a structured prompt and returns parsed JSON output.
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a scientific reasoning assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    content = response["choices"][0]["message"]["content"]

    try:
        decision = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError("LLM output is not valid JSON")

    return decision
