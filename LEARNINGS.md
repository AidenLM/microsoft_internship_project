# Learnings

Running notes on what we figured out while building this, mainly so we don't
forget it by the time we write the final report. Updating this as we go
through each week instead of trying to remember everything at the end.

## Week 1

**RAG in one sentence**: retrieve relevant text from your own documents,
stick it into the prompt as context, then let the model generate an answer
using that context instead of just guessing from what it memorized during
training. That's basically why RAG answers are more accurate and can cite
sources - the model isn't working from memory alone.

**The Foundry Local SDK on PyPI right now is not what the tutorials show.**
Every guide we found online (including the blog post this whole project is
based on) uses an older API: `from foundry_local import FoundryLocalManager`,
then a `.endpoint` / `.api_key` you hand to an OpenAI client. `pip install
foundry-local-sdk` currently installs a completely different, newer version -
import path is `foundry_local_sdk` (not `foundry_local`), and there's no
OpenAI-compatible client at all anymore. Instead you go through
`FoundryLocalManager -> catalog.get_model(alias) -> model.load()`, then open
a `ChatSession`, build a `Request` with a `MessageItem`, and call
`session.process_request(request)`. Had to read the actual installed
package source to figure this out since there's no up-to-date doc for it
yet. Lesson: don't trust a tutorial's exact code snippets blindly, check
what's actually installed.

**Chat models vs. embedding models are trained for different jobs, even
though both are transformers and both work with vectors internally.**
- A chat model (Mistral, in our case) turns text into vectors, but ends with
  a probability distribution over the next token and generates word by
  word. It's trained to predict/continue text well.
- An embedding model skips all of that - no next-token prediction, no text
  generation. It just squashes the whole input into one fixed-size vector,
  trained so that semantically similar text ends up with similar vectors
  (so cosine similarity actually means something). That's what makes it
  usable for search - you can't reliably use a chat model's internals for
  that without this specific training.

Foundry Local's catalog only exposes embedding-task models under separate
aliases (`qwen3-embedding-0.6b`, `qwen3-embedding-8b`) - there's no
"embedding mode" for chat models like Mistral, at least not in this catalog.

**Model size vs. what the machine can actually handle.** Went through a
few models to find a workable size:
- `qwen3-0.6b` - loads and answers in ~2s, but got basic factual questions
  wrong. Too small to be useful for real answers.
- `qwen3-4b` - noticeably better, still fast (a few seconds per answer).
- `mistral-7b-v0.2` (~4.2GB on disk) - best answers so far, still only
  takes a few seconds and runs fine.
- Tried pushing to `qwen3-8b` (~5.8GB) and it got risky memory-wise on this
  machine (16GB RAM total) - backed off from that. `mistral-7b-v0.2` seems
  like the sweet spot for now.

Went with `mistral-7b-v0.2` as the default chat model, `qwen3-embedding-0.6b`
planned for embeddings in Week 2.

## Week 2

**Seeing an actual embedding made the "similarity" idea click.** Ran
`qwen3-embedding-0.6b` on a few test sentences and compared them with cosine
similarity:

```
"The cat sat on the mat."  vs  "A feline was resting on the rug."   -> 0.73
"The cat sat on the mat."  vs  "The stock market crashed yesterday." -> 0.35
```

The first pair shares basically no words in common ("cat"/"feline",
"mat"/"rug", "sat"/"resting") but scored high, because the *meaning* is the
same. The second pair scored low because it's actually unrelated. That's the
whole point of using embeddings for retrieval instead of plain keyword
matching - the model is comparing meaning, not exact word overlap. This is
the mechanism RAG relies on: embed the user's question, compare it against
every stored document chunk's embedding, and pull back whichever chunks
score highest - regardless of whether they share any literal words with the
question.

Each sentence became a vector of 1024 numbers (`shape=[1024]`), returned as
raw float32 bytes from the SDK - had to unpack it with `numpy.frombuffer`
to get actual floats out.
