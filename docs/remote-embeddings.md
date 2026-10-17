# Embeddings on a server

By default DigUp runs EmbeddingGemma 2 on your computer with llama.cpp. You can send the embedding work to a server
instead, for example a GPU box on your network that runs vLLM, llama.cpp's `llama-server`, Ollama or Text Embeddings
Inference. DigUp then downloads no model and uses no model memory. Your files' text, pictures and sound go to that
server, so use only a server you trust.

The server must have an OpenAI-compatible `POST <url>/embeddings` endpoint.

## Settings

Put `embedder.json` in DigUp's folder:

- Mac: `~/Library/Application Support/DigUp/embedder.json`
- Windows: `%LOCALAPPDATA%\DigUp\embedder.json`
- Linux: `$XDG_DATA_HOME/digup/embedder.json` (usually `~/.local/share/digup/embedder.json`)

```json
{
  "url": "http://spark.local:8000/v1",
  "api_key": "optional",
  "model": "google/embeddinggemma-2",
  "media": "messages"
}
```

Environment variables override the file: `DIGUP_EMBED_URL`, `DIGUP_EMBED_API_KEY`, `DIGUP_EMBED_MODEL` and
`DIGUP_EMBED_MEDIA`. Without a URL, DigUp runs the model on your computer. Restart DigUp after you change the file.

| Key | Default | What it does |
|---|---|---|
| `url` | (none: local) | The base URL. DigUp adds `/embeddings`. |
| `api_key` | (none) | Sent as `Authorization: Bearer <key>`. |
| `model` | `google/embeddinggemma-2` | The `model` field of each request. |
| `media` | `messages` | How pictures and sound go to the server (see below). |

`media`:

- `messages`: chat-style content parts with data URLs (`image_url`, `audio_url`), as vLLM takes them for multimodal
  embeddings.
- `llama`: llama-server's prompt objects, `{"prompt_string": "<__media__>", "multimodal_data": ["<base64>"]}`.
- `none`: text only. Pictures, video frames and sound get no vector. DigUp still finds them by name and by their text.

## What the server must do

- Give vectors of at least 768 numbers. Longer vectors are cut to 768 and normalized again. That is correct only for
  Matryoshka models such as EmbeddingGemma and Qwen3-Embedding.
- Not add its own prompt. DigUp sends text with EmbeddingGemma 2's task prompts already in it
  (`task: search result | query: ...`, `title: ... | text: ...`).
- Size pictures itself. The local runtime fits each picture to a token budget (70 to 1120 vision tokens); a server
  uses its own image processor.

## Example: EmbeddingGemma 2 with llama-server

```sh
llama-server -hf ggml-org/embeddinggemma-2-GGUF --embeddings --pooling mean --host 0.0.0.0 --port 8080
```

```json
{"url": "http://spark.local:8080/v1", "media": "llama"}
```

## The index

The index records where its vectors come from: the model name and `openai-api`, not the server's address. A change
between the local model and a server, or to another model, makes DigUp embed every file again. A move of the same
model to another address keeps the index. Vectors from a server are near the local ones but not equal, so do not mix
them in one index.
