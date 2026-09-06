import time

from foundry_local_sdk import Configuration, FoundryLocalManager

from main import MODEL_ALIAS, answer_query, load_model

DECLINE_PHRASES = (
    "don't have information",
    "don't have enough",
    "not mentioned in the context",
    "no mention of",
    "doesn't mention",
    "not in the context",
    "no information",
    "i don't know",
)

TEST_CASES = [
    # (question, should_be_answerable)
    ("What is RAG?", True),
    ("What is Foundry Local?", True),
    ("What is SQLite used for in this project?", True),
    ("What is vibe coding?", True),
    ("What is the main risk of vibe coding?", True),
    ("What practices make vibe coding safer?", True),
    ("What is the capital of Japan?", False),
    ("How do I bake a chocolate cake?", False),
    ("What's the best programming language?", False),
]


def main():
    manager = FoundryLocalManager(Configuration(app_name="local-rag-assistant"))
    chat_model = load_model(manager, MODEL_ALIAS)

    passed = 0
    for question, should_be_answerable in TEST_CASES:
        t0 = time.time()
        answer, chunks = answer_query(manager, chat_model, question)
        elapsed = time.time() - t0

        declined = any(phrase in answer.lower() for phrase in DECLINE_PHRASES)
        got_answerable = not declined
        correct = got_answerable == should_be_answerable

        passed += correct
        status = "PASS" if correct else "FAIL"
        print(f"[{status}] ({elapsed:.1f}s) {question!r}")
        print(f"  expected answerable={should_be_answerable}, got answerable={got_answerable}")
        print(f"  answer: {answer}\n")

    print(f"{passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    main()
