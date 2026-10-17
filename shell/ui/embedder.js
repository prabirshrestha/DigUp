// Where the embeddings come from: the model on this computer (the default), or an OpenAI-compatible server
// (docs/remote-embeddings.md). Settings → Model shows it with a Save button. Setup shows the choice on its first step
// and the server's fields on a step of their own, and saves on Continue.
// The form edits a draft, compared with what's saved to tell whether there's anything to save; Test checks the draft.
"use strict";

const DEFAULT_EMBEDDING_MODEL = "google/embeddinggemma-2";
/// The server's kind, which says how pictures and sound go to it (the helper's `media`).
const SERVER_TYPES = [["messages", "Chat messages (vLLM)"], ["llama", "Media prompt (llama-server)"]];

/// Builds the choice in `choiceContainer` and the server's fields in `fieldsContainer` (they may be the same one).
/// `options.save`: a Save button of its own (Settings), else the page calls `save()`; `options.changed`: told whenever
/// the draft or what's saved changes. Returns what the page needs of it.
function embedderForm(choiceContainer, fieldsContainer, options = {}) {
  const invoke = (command, args) => window.__TAURI__.core.invoke(command, args);
  const state = {
    saved: null,                         // the server saved, null for this computer's model
    draft: draftOf(null),
    test: { state: "idle" },             // idle | testing | done (with ok and message)
    saving: false,
    problem: "",
  };
  let testNumber = 0;
  let built = false;   // `changed` isn't told before the page has the form

  const make = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  };
  const field = (label, input) => {
    const row = make("label", "server-field");
    row.append(make("span", "server-label", label), input);
    return row;
  };
  const choice = (value, text, info) => {
    const label = make("label", "choice");
    const radio = make("input");
    radio.type = "radio";
    radio.name = "embedder-where";
    radio.value = value;
    radio.addEventListener("change", () => edit({ where: value }));
    label.append(radio, make("span", "", text));
    if (info) label.append(make("span", "choice-info", info));
    return label;
  };

  const here = choice("here", "On this computer", "downloads the model");
  const server = choice("server", "On a server", "OpenAI-compatible, such as vLLM or llama-server");
  const url = make("input", "field");
  url.type = "url";
  url.placeholder = "http://spark.local:8080/v1";
  const key = make("input", "field");
  key.type = "password";
  key.placeholder = "Optional";
  const model = make("input", "field");
  model.placeholder = DEFAULT_EMBEDDING_MODEL;
  const media = make("select", "field");
  media.append(...SERVER_TYPES.map(([value, text]) => {
    const option = make("option", "", text);
    option.value = value;
    return option;
  }));
  for (const input of [url, key, model, media]) {
    input.spellcheck = false;
    input.autocomplete = "off";
    input.addEventListener("input", () => edit({ url: url.value, apiKey: key.value, model: model.value, media: media.value }));
  }
  const mediaHelp = make("p", "note server-help", "How pictures and sound are sent. Text is the same for both. Test "
    + "shows what your server takes.");
  const warning = make("p", "note server-help", "Your files' text, pictures and sound go to this server to be "
    + "understood. Use only a server you trust.");
  const testButton = make("button", "", "Test");
  const testResult = make("span", "test-result");
  testButton.addEventListener("click", runTest);
  const testRow = make("div", "test-row");
  testRow.append(testButton, testResult);
  const fields = make("div", "server-fields");
  fields.append(field("URL", url), field("API key", key), field("Model", model), field("Server type", media),
                mediaHelp, testRow, warning);
  const saveRow = make("div", "save-row");
  const saveNote = make("span", "save-note");
  const saveButton = make("button", "primary", "Save");
  saveButton.addEventListener("click", () => save().catch(() => {}));
  saveRow.append(saveNote, saveButton);
  const note = make("p", "note server-help", "Changing this has DigUp index every file again.");
  // Settings only: on a first launch nothing is indexed yet.
  const others = options.save ? [note, saveRow] : [];
  if (choiceContainer === fieldsContainer) {
    choiceContainer.replaceChildren(here, server, fields, ...others);
  } else {
    choiceContainer.replaceChildren(here, server);
    fieldsContainer.replaceChildren(fields, ...others);
  }

  function edit(change) {
    state.draft = { ...state.draft, ...change };
    state.test = { state: "idle" };
    testNumber += 1;
    state.problem = "";
    render();
  }

  function render() {
    const { draft, test } = state;
    here.querySelector("input").checked = draft.where === "here";
    server.querySelector("input").checked = draft.where === "server";
    fields.hidden = draft.where !== "server";
    // Set only when they differ, so typing keeps its place.
    if (url.value !== draft.url) url.value = draft.url;
    if (key.value !== draft.apiKey) key.value = draft.apiKey;
    if (model.value !== draft.model) model.value = draft.model;
    media.value = draft.media;
    testButton.disabled = test.state === "testing" || !draft.url.trim();
    testResult.className = "test-result" + (test.state === "done" ? (test.ok ? " ok" : " failed") : "");
    testResult.textContent = test.state === "testing" ? "Testing…" : test.state === "done" ? test.message : "";
    note.hidden = !changed();
    saveButton.disabled = state.saving || !changed() || !ready();
    saveNote.className = "save-note" + (state.problem ? " failed" : "");
    saveNote.textContent = state.saving ? "Saving…" : state.problem;
    if (built) options.changed?.();
  }

  async function runTest() {
    const number = ++testNumber;
    state.test = { state: "testing" };
    render();
    let result;
    try {
      result = await invoke("test_embedder", { server: serverOf(state.draft) });
    } catch (error) {
      result = { ok: false, message: String(error) };
    }
    if (number !== testNumber) return;   // the draft changed meanwhile
    state.test = { state: "done", ok: result.ok, message: result.message };
    render();
  }

  /// What's saved differs from the draft.
  function changed() {
    return JSON.stringify(serverOf(state.draft)) !== JSON.stringify(serverOf(draftOf(state.saved)));
  }

  /// The draft can be saved: this computer, or a server with a URL.
  function ready() {
    return state.draft.where === "here" || Boolean(state.draft.url.trim());
  }

  /// Saves the draft (the shell restarts its helpers with it). Throws, after showing why, when it can't.
  async function save() {
    if (!changed() || !ready()) return;
    state.saving = true;
    state.problem = "";
    render();
    try {
      const chosen = serverOf(state.draft);
      await invoke("save_embedder", { server: chosen });
      // The fields stay as typed: going back to the server they were for needs no typing again.
      state.saved = chosen;
    } catch (error) {
      state.problem = String(error);
      throw error;
    } finally {
      state.saving = false;
      render();
    }
  }

  async function load() {
    try {
      state.saved = await invoke("embedder");
    } catch (error) {
      state.problem = String(error);
    }
    state.draft = draftOf(state.saved);
    render();
  }

  render();
  built = true;
  return { load, save, changed, ready, saved: () => state.saved, choosingServer: () => state.draft.where === "server",
           problem: () => state.problem };
}

/// The form's fields for a saved server (or this computer's model).
function draftOf(server) {
  return { where: server ? "server" : "here", url: server?.url || "", apiKey: server?.api_key || "",
           model: server?.model || DEFAULT_EMBEDDING_MODEL, media: server?.media || "messages" };
}

/// The server a draft describes, as the shell takes it (null: this computer's model).
function serverOf(draft) {
  if (draft.where !== "server") return null;
  const server = { url: draft.url.trim() };
  if (draft.apiKey) server.api_key = draft.apiKey;
  server.model = draft.model.trim() || DEFAULT_EMBEDDING_MODEL;
  server.media = draft.media;
  return server;
}
