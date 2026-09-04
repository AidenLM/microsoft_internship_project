from foundry_local_sdk import (
    ChatSession,
    Configuration,
    FoundryLocalManager,
    MessageItem,
    Request,
    TextItemType,
)

MODEL_ALIAS = "qwen3-0.6b"


def load_model(manager):
    model = manager.catalog.get_model(MODEL_ALIAS)
    if model is None:
        raise RuntimeError(f"Model '{MODEL_ALIAS}' not found in catalog")
    if not model.is_cached:
        print(f"Downloading {MODEL_ALIAS}, this only happens once...")
        model.download()
    model.load()
    return model


def extract_answer(message):
    for part in message.parts:
        if getattr(part, "type", None) == TextItemType.DEFAULT:
            return part.text
    return message.parts[-1].text


def hello_model():
    manager = FoundryLocalManager(Configuration(app_name="local-rag-assistant"))
    model = load_model(manager)

    with ChatSession(model) as session:
        request = Request().add_item(MessageItem.user("Hello, world"))
        with session.process_request(request) as response:
            print(extract_answer(response.get_item(0)))


def main():
    hello_model()


if __name__ == "__main__":
    main()
