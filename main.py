from foundry_local_sdk import (
    ChatSession,
    Configuration,
    FoundryLocalManager,
    MessageItem,
    Request,
    TextItemType,
)

from retrieval import get_top_chunks

MODEL_ALIAS = "mistral-7b-v0.2"

# Below this, the best match is basically unrelated to the question - seen
# ~0.77-0.83 for genuinely relevant chunks and ~0.38-0.39 for unrelated
# ones during testing, so 0.5 sits comfortably between the two.
MIN_SIMILARITY = 0.5
NO_ANSWER = "I don't have information about that in my documents."

PROMPT_TEMPLATE = """Answer the question using only the context below.
If the answer isn't in the context, say you don't know instead of guessing.
Keep the answer short.

Context:
{context}

Question: {question}
"""


def load_model(manager, alias):
    model = manager.catalog.get_model(alias)
    if model is None:
        raise RuntimeError(f"Model '{alias}' not found in catalog")
    if not model.is_cached:
        print(f"Downloading {alias}, this only happens once...")
        model.download()
    model.load()
    return model


def extract_answer(message):
    for part in message.parts:
        if getattr(part, "type", None) == TextItemType.DEFAULT:
            return part.text
    return message.parts[-1].text


def answer_query(manager, chat_model, question):
    chunks = get_top_chunks(manager, question, top_k=3)

    if not chunks or chunks[0][0] < MIN_SIMILARITY:
        return NO_ANSWER, chunks

    context = "\n".join(f"- {content}" for _, _, _, content in chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    with ChatSession(chat_model) as session:
        request = Request().add_item(MessageItem.user(prompt))
        with session.process_request(request) as response:
            return extract_answer(response.get_item(0)), chunks


def chat():
    manager = FoundryLocalManager(Configuration(app_name="local-rag-assistant"))
    chat_model = load_model(manager, MODEL_ALIAS)

    print(f"RAG assistant ready ({MODEL_ALIAS}, answering from docs/). Type 'exit' to quit.\n")

    while True:
        question = input("you> ").strip()
        if not question or question.lower() in ("exit", "quit"):
            break

        answer, chunks = answer_query(manager, chat_model, question)
        print(f"model> {answer}")
        sources = ", ".join(f"{source} ({score:.2f})" for score, _, source, _ in chunks)
        print(f"  sources: {sources}\n")


def main():
    chat()


if __name__ == "__main__":
    main()
