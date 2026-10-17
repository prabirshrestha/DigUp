import DigUpKit
import Foundation

/// `digup embedder get|set|test`: the embedding server for shells that can't link DigUpKit (`RemoteEmbedding`). Each
/// prints one JSON line; `set` and `test` read the server from stdin, `{"url": …, "api_key": …, "model": …,
/// "media": …}`, and `set` with no URL goes back to the built-in model.
///
///     get   {"file": …, "server": {…} | null}
///     set   {"ok": true} | {"ok": false, "message": …}
///     test  {"ok": Bool, "message": …}
func embedderSettings(_ args: Arguments) throws -> Int32 {
    func server() throws -> RemoteEmbedding {
        try JSONDecoder().decode(RemoteEmbedding.self, from: FileHandle.standardInput.readDataToEndOfFile())
    }
    func print(_ value: some Encodable) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        try FileHandle.standardOutput.write(contentsOf: encoder.encode(value) + [0x0A])
    }
    struct Saved: Encodable {
        let file: String
        let server: RemoteEmbedding?
    }
    switch args.positional.first {
    case "get":
        try print(Saved(file: RemoteEmbedding.file.path, server: RemoteEmbedding.saved()))
    case "set":
        do {
            try RemoteEmbedding.save(try server())
            try print(RemoteEmbedding.TestResult(ok: true, message: ""))
        } catch {
            try print(RemoteEmbedding.TestResult(ok: false, message: "\(error)"))
            return 1
        }
    case "test":
        try print(try server().test())
    default:
        throw CLIError("digup embedder get|set|test")
    }
    return 0
}
