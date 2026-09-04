from foundry_local import FoundryLocalManager
from openai import OpenAI

MODEL_ALIAS = "phi-3.5-mini"


def get_client():
    manager = FoundryLocalManager(MODEL_ALIAS)
    client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)
    return client, manager


def hello_model():
    client, manager = get_client()
    model_id = manager.get_model_info(MODEL_ALIAS).id

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "Hello, world"}],
    )
    print(response.choices[0].message.content)


def main():
    hello_model()


if __name__ == "__main__":
    main()
