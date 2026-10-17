#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Foundation

/// Where to send embedding requests instead of running the model here: a server with an OpenAI-compatible
/// `/embeddings` endpoint (vLLM, llama.cpp's llama-server, Ollama, Text Embeddings Inference and the like).
///
/// Read from `embedder.json` in DigUp's own folder (`AppPaths.applicationSupport`), then the environment, which wins:
///
///     {"url": "http://spark:8000/v1", "api_key": "...", "model": "google/embeddinggemma-2", "media": "messages"}
///
///     DIGUP_EMBED_URL  DIGUP_EMBED_API_KEY  DIGUP_EMBED_MODEL  DIGUP_EMBED_MEDIA
///
/// Without a URL, DigUp runs the model here (the default). The server must give vectors of at least
/// `Vectors.dimension` (768) numbers; longer ones are cut to 768 and normalized again, which suits Matryoshka models
/// (EmbeddingGemma, Qwen3-Embedding) only. Text arrives already prompted (`Prompts`): the server must not add its own.
public struct RemoteEmbedding: Codable, Sendable, Equatable {
    /// How images and audio go to the server. Text always goes as `input`.
    public enum Media: String, Codable, Sendable {
        /// Chat-style content parts in `messages` (`image_url`, `audio_url` as data URLs): vLLM's multimodal embeddings.
        case messages
        /// llama-server's prompt objects: `{"prompt_string": "<__media__>", "multimodal_data": [base64]}`.
        case llama
        /// Text only: images and audio get no vector (they're still found by name and text).
        case off = "none"
    }

    public var url: String
    public var apiKey: String?
    public var model: String
    public var media: Media

    public static let defaultModel = "google/embeddinggemma-2"
    public static let fileName = "embedder.json"

    public init(url: String, apiKey: String? = nil, model: String = defaultModel, media: Media = .messages) {
        self.url = url
        self.apiKey = apiKey
        self.model = model
        self.media = media
    }

    enum CodingKeys: String, CodingKey {
        case url, apiKey = "api_key", model, media
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        url = try container.decodeIfPresent(String.self, forKey: .url) ?? ""
        apiKey = try container.decodeIfPresent(String.self, forKey: .apiKey)
        model = try container.decodeIfPresent(String.self, forKey: .model) ?? Self.defaultModel
        media = try container.decodeIfPresent(Media.self, forKey: .media) ?? .messages
    }

    public static var file: URL { AppPaths.applicationSupport.appendingPathComponent(fileName) }

    /// The configured server, or nil to run the model here. A file that doesn't parse is an error, not a silent
    /// fallback to the local model.
    public static func load(file: URL = file,
                            environment: [String: String] = ProcessInfo.processInfo.environment) throws -> RemoteEmbedding? {
        var config = RemoteEmbedding(url: "")
        if let data = FileManager.default.contents(atPath: file.path) {
            do {
                config = try JSONDecoder().decode(RemoteEmbedding.self, from: data)
            } catch {
                throw WorkerError.failed("can't read \(file.path): \(error)")
            }
        }
        func value(_ name: String) -> String? { environment[name].flatMap { $0.isEmpty ? nil : $0 } }
        if let url = value("DIGUP_EMBED_URL") { config.url = url }
        if let key = value("DIGUP_EMBED_API_KEY") { config.apiKey = key }
        if let model = value("DIGUP_EMBED_MODEL") { config.model = model }
        if let media = value("DIGUP_EMBED_MEDIA") {
            guard let parsed = Media(rawValue: media) else {
                throw WorkerError.failed("DIGUP_EMBED_MEDIA must be messages, llama or none, not \(media)")
            }
            config.media = parsed
        }
        config.url = config.url.trimmingCharacters(in: .whitespaces)
        return config.url.isEmpty ? nil : config
    }

    /// `load()` for callers that only ask whether a server is set; a broken file counts as set, so starting the
    /// embedder reports it.
    public static var isConfigured: Bool {
        do { return try load() != nil } catch { return true }
    }

    /// What this server's vectors are, for index fingerprints: the model, not the host, so moving the server keeps the
    /// index.
    public var info: WorkerInfo {
        WorkerInfo(model: model, revision: "remote", runtime: "openai-api", dtype: "api", dim: Vectors.dimension,
                   textOnly: false, loadSeconds: nil)
    }
}

/// An `Embedder` that sends each request to an OpenAI-compatible server (`RemoteEmbedding`). Nothing loads here, so it
/// starts at once and uses no model memory; one thread at a time, like the others.
public final class RemoteEmbedder: Embedder {
    public private(set) var info: WorkerInfo
    private let config: RemoteEmbedding
    private let endpoint: URL
    private let textOnly: Bool
    private let session: URLSession
    private static let textBatch = 32

    public init(_ config: RemoteEmbedding, textOnly: Bool) throws {
        let base = config.url.hasSuffix("/") ? String(config.url.dropLast()) : config.url
        guard let endpoint = URL(string: base + "/embeddings"), endpoint.scheme?.hasPrefix("http") == true else {
            throw WorkerError.failed("embedding server URL isn't http(s): \(config.url)")
        }
        let started = Date()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 300
        session = URLSession(configuration: configuration)
        self.config = config
        self.endpoint = endpoint
        self.textOnly = textOnly
        info = config.info
        // One small request now: a wrong URL, key or model shows up here, not halfway through indexing.
        _ = try embedTexts([Prompts.query("warm up")])
        let seconds = (Date().timeIntervalSince(started) * 100).rounded() / 100
        info = WorkerInfo(model: info.model, revision: info.revision, runtime: info.runtime, dtype: info.dtype,
                          dim: info.dim, textOnly: textOnly, loadSeconds: seconds)
    }

    public func embedTexts(_ texts: [String]) throws -> [[Float]] {
        var vectors: [[Float]] = []
        for start in stride(from: 0, to: texts.count, by: Self.textBatch) {
            let batch = Array(texts[start..<min(start + Self.textBatch, texts.count)])
            let got = try post(["input": batch])
            guard got.count == batch.count else {
                throw WorkerError.failed("the server sent \(got.count) vectors for \(batch.count) texts")
            }
            vectors += got
        }
        return vectors
    }

    public func embedImages(_ paths: [String], budget: Int) throws -> [Embedding] {
        try paths.map { try embedMedia($0, image: true) }
    }

    public func embedAudio(_ paths: [String]) throws -> [Embedding] {
        try paths.map { try embedMedia($0, image: false) }
    }

    public func close() {
        session.invalidateAndCancel()
    }

    // MARK: Requests

    /// One file's vector, or why it has none (a file the server can't read must not sink its neighbours).
    private func embedMedia(_ path: String, image: Bool) throws -> Embedding {
        guard !textOnly else { throw WorkerError.failed("this embedder was started text-only") }
        guard config.media != .off else {
            return Embedding(vector: nil, error: "the embedding server is set to text only")
        }
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: path)).base64EncodedString()
            let body: [String: Any]
            switch config.media {
            case .messages:
                let mime = Self.mimeType(path, image: image)
                let part: [String: Any] = image
                    ? ["type": "image_url", "image_url": ["url": "data:\(mime);base64,\(data)"]]
                    : ["type": "audio_url", "audio_url": ["url": "data:\(mime);base64,\(data)"]]
                body = ["messages": [["role": "user", "content": [part]]]]
            case .llama, .off:
                body = ["input": [["prompt_string": "<__media__>", "multimodal_data": [data]]]]
            }
            guard let vector = try post(body).first else { throw WorkerError.failed("the server sent no vector") }
            return Embedding(vector: vector, error: nil)
        } catch {
            return Embedding(vector: nil, error: String("\(error)".prefix(300)))
        }
    }

    private static func mimeType(_ path: String, image: Bool) -> String {
        switch URL(fileURLWithPath: path).pathExtension.lowercased() {
        case "jpg", "jpeg": "image/jpeg"
        case "png": "image/png"
        case "webp": "image/webp"
        case "gif": "image/gif"
        case "wav": "audio/wav"
        case "mp3": "audio/mpeg"
        case "flac": "audio/flac"
        case "m4a": "audio/mp4"
        case "ogg": "audio/ogg"
        default: image ? "image/png" : "audio/wav"
        }
    }

    private struct Reply: Decodable {
        struct Item: Decodable {
            let index: Int?
            let embedding: [Float]
        }
        let data: [Item]
    }

    /// POSTs `body` (plus the model and float encoding) and returns the vectors in input order, each cut to
    /// `Vectors.dimension` and L2-normalized.
    private func post(_ body: [String: Any]) throws -> [[Float]] {
        var body = body
        body["model"] = config.model
        body["encoding_format"] = "float"
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = config.apiKey, !key.isEmpty { request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let result = Self.send(request, on: session)
        let (data, response) = try result.get()
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else {
            let text = String(decoding: data.prefix(300), as: UTF8.self)
            throw WorkerError.failed("embedding server said \(status) at \(endpoint.absoluteString): \(text)")
        }
        let reply: Reply
        do {
            reply = try JSONDecoder().decode(Reply.self, from: data)
        } catch {
            throw WorkerError.failed("embedding server reply isn't an OpenAI embeddings reply: \(error)")
        }
        let items = reply.data.enumerated().sorted { ($0.element.index ?? $0.offset) < ($1.element.index ?? $1.offset) }
        return try items.map { try Self.normalized($0.element.embedding) }
    }

    static func normalized(_ vector: [Float]) throws -> [Float] {
        guard vector.count >= Vectors.dimension else {
            throw WorkerError.failed("the server's vectors have \(vector.count) numbers; DigUp needs \(Vectors.dimension)")
        }
        let cut = vector.count == Vectors.dimension ? vector : Array(vector.prefix(Vectors.dimension))
        let norm = cut.reduce(0) { $0 + Double($1) * Double($1) }.squareRoot()
        guard norm > 0 else { throw WorkerError.failed("the server sent an empty vector") }
        return cut.map { Float(Double($0) / norm) }
    }

    /// URLSession is asynchronous; `Embedder` calls are not. Waits for the one request.
    private static func send(_ request: URLRequest, on session: URLSession) -> Result<(Data, URLResponse), Error> {
        final class Box: @unchecked Sendable { var result: Result<(Data, URLResponse), Error>? }
        let box = Box()
        let done = DispatchSemaphore(value: 0)
        session.dataTask(with: request) { data, response, error in
            if let data, let response {
                box.result = .success((data, response))
            } else {
                box.result = .failure(error ?? WorkerError.failed("no reply from the embedding server"))
            }
            done.signal()
        }.resume()
        done.wait()
        return box.result!
    }
}
