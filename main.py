from foundry_local_sdk import (
    ChatSession,
    Configuration,
    FoundryLocalManager,
    MessageItem,
    Request,
    TextItemType,
)

MODEL_ALIAS = "mistral-7b-v0.2"


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


def chat():
    manager = FoundryLocalManager(Configuration(app_name="local-rag-assistant"))
    model = load_model(manager)

    print(f"Chatting with {MODEL_ALIAS}. Type 'exit' to quit.\n")

    with ChatSession(model) as session:
        while True:
            question = input("you> ").strip()
            if not question or question.lower() in ("exit", "quit"):
                break

            request = Request().add_item(MessageItem.user(question))
            with session.process_request(request) as response:
                print(f"model> {extract_answer(response.get_item(0))}\n")


def main():
    chat()


if __name__ == "__main__":
    main()
