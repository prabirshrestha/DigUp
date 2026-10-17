# Embeddings on a server

By default DigUp runs EmbeddingGemma 2 on your computer with llama.cpp. You can send the embedding work to a server
instead, for example a GPU box on your network that runs llama.cpp's `llama-server` or vLLM. DigUp then downloads
no model and uses no model memory. Your files' text, pictures and sound go to that server, so use only a server you
trust.

The server must have an OpenAI-compatible `POST <url>/embeddings` endpoint.

## Settings

Choose "On a server" on the first step of setup, or later in Settings > Model (Mac, Windows and Linux). Fill in the
URL, the API key if the server needs one, the model and the server type, then click Test. Test sends one text, one
small picture and one short sound, and says what works. Save (or Continue in setup) stores the server and restarts
DigUp's helpers, so DigUp switches at once and downloads no model. "On this computer" goes back to the built-in
model and downloads it if it isn't there yet.

The settings go to `embedder.json` in DigUp's folder, which you can also write yourself:

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
`DIGUP_EMBED_MEDIA`. Without a URL, DigUp runs the model on your computer. Restart DigUp after you change the file by
hand. `digup embedder get`, `digup embedder set` and `digup embedder test` show, save and try the server from the
command line; the last two read the JSON above on stdin.

| Key | Default | What it does |
|---|---|---|
| `url` | (none: local) | The base URL. DigUp adds `/embeddings`. |
| `api_key` | (none) | Sent as `Authorization: Bearer <key>`. |
| `model` | `google/embeddinggemma-2` | The `model` field of each request. |
| `media` | `messages` | How pictures and sound go to the server (see below). |

`media` (the server type in Settings) is how pictures and sound are sent. DigUp always sends text, pictures and sound;
text goes as `input` either way.

- `llama`, "Media prompt (llama-server)": llama-server's prompt objects, `{"prompt_string": "<marker>",
  "multimodal_data": ["<base64>"]}`. DigUp reads the marker from the server's `GET /props` (newer builds pick a
  random one) and falls back to `<__media__>`. Tested end to end with EmbeddingGemma 2 for text, pictures and sound.
- `messages`, "Chat messages (vLLM)": chat-style content parts with data URLs (`image_url`, `audio_url`) in
  `messages`. vLLM takes `messages` on `/v1/embeddings` only with a model-specific `--chat-template`, and its docs
  list images, not sound, for embedding models. Not tested against vLLM here: use Test to see what your server takes.

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
