//! Where the embeddings come from: the model on this computer (the default), or an OpenAI-compatible server
//! (docs/remote-embeddings.md). The choice is the helper's embedder.json, read and written only by `digup embedder
//! get|set|test` (Sources/digup/EmbedderSettings.swift), so the shell keeps no path or format of its own. Each call
//! starts a process and may wait on the network: callers run them off the main thread.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// An embedding server, as the helper reads and writes it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Server {
    /// The base URL; the helper adds `/embeddings`.
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    pub model: String,
    pub media: Media,
}

/// How pictures and sound go to the server (text always goes as `input`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Media {
    /// vLLM's chat-style content parts.
    Messages,
    /// llama-server's prompt objects.
    Llama,
}

/// What a test (or a save) says: whether it worked, and what to show.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    pub ok: bool,
    #[serde(default)]
    pub message: String,
}

/// The server saved, or None for the model on this computer.
pub fn saved(helper: &Path) -> Result<Option<Server>, String> {
    #[derive(Deserialize)]
    struct Saved {
        server: Option<Server>,
    }
    let saved: Saved = run(helper, "get", None)?;
    Ok(saved.server)
}

/// Saves the server for every helper to read; None goes back to the model on this computer.
pub fn save(helper: &Path, server: Option<&Server>) -> Result<(), String> {
    let empty = Server { url: String::new(), api_key: None, model: String::new(), media: Media::Messages };
    let outcome: Outcome = run(helper, "set", Some(server.unwrap_or(&empty)))?;
    if outcome.ok { Ok(()) } else { Err(format!("can't save the embedding server: {}", outcome.message)) }
}

/// One text request, one picture and one sound (about 20 seconds at most each).
pub fn test(helper: &Path, server: &Server) -> Result<Outcome, String> {
    run(helper, "test", Some(server))
}

fn run<T: serde::de::DeserializeOwned>(helper: &Path, verb: &str, input: Option<&Server>) -> Result<T, String> {
    let mut command = Command::new(helper);
    command.args(["embedder", verb]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = command.spawn().map_err(|error| format!("can't run {}: {error}", helper.display()))?;
    crate::helper::adopt(&child);
    let mut stdin = child.stdin.take().ok_or("no stdin")?;
    if let Some(input) = input {
        let _ = stdin.write_all(&serde_json::to_vec(input).map_err(|error| error.to_string())?);
    }
    drop(stdin);
    let output = child.wait_with_output().map_err(|error| format!("can't run {}: {error}", helper.display()))?;
    let line = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(line.trim()).map_err(|_| {
        let said = String::from_utf8_lossy(&output.stderr).trim().to_string();
        format!("digup embedder {verb} failed: {}", if said.is_empty() { output.status.to_string() } else { said })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_reads_and_writes_the_helpers_json() {
        let server: Server = serde_json::from_str(
            r#"{"api_key":"k","media":"messages","model":"google/embeddinggemma-2","url":"http://spark.local:8080/v1"}"#,
        )
        .unwrap();
        assert_eq!(server, Server {
            url: "http://spark.local:8080/v1".into(),
            api_key: Some("k".into()),
            model: "google/embeddinggemma-2".into(),
            media: Media::Messages,
        });
        let keyless = Server { api_key: None, media: Media::Llama, ..server };
        assert_eq!(serde_json::to_string(&keyless).unwrap(),
                   r#"{"url":"http://spark.local:8080/v1","model":"google/embeddinggemma-2","media":"llama"}"#);
        assert!(serde_json::from_str::<Media>(r#""none""#).is_err());
    }

    /// The real helper, when it's built (`swift build`), against a scratch file: save, read back, and back to built-in.
    #[test]
    fn helper_saves_and_reads_the_server() {
        let helper = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.build/debug/digup");
        if !helper.is_file() {
            eprintln!("skipped: no {}", helper.display());
            return;
        }
        let file = std::env::temp_dir().join(format!("digup-embedder-test-{}.json", std::process::id()));
        std::env::set_var("DIGUP_EMBEDDER_FILE", &file);
        let server = Server { url: "http://127.0.0.1:9/v1".into(), api_key: None, model: "m".into(), media: Media::Llama };
        save(&helper, Some(&server)).unwrap();
        assert_eq!(saved(&helper).unwrap(), Some(server));
        save(&helper, None).unwrap();
        assert_eq!(saved(&helper).unwrap(), None);
        assert!(!file.exists());
    }
}
