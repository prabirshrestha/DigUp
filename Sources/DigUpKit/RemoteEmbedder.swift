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

    /// `embedder.json` in DigUp's folder; $DIGUP_EMBEDDER_FILE moves it (dev runs, which mustn't touch the real one).
    public static var file: URL {
        if let path = ProcessInfo.processInfo.environment["DIGUP_EMBEDDER_FILE"], !path.isEmpty {
            return URL(fileURLWithPath: path)
        }
        return AppPaths.applicationSupport.appendingPathComponent(fileName)
    }

    /// What the settings screens show: the file alone, without the environment. nil for the built-in model, and for a
    /// file that doesn't parse (saving over it fixes it).
    public static func saved(file: URL = file) -> RemoteEmbedding? {
        try? load(file: file, environment: [:])
    }

    /// Saves the server for every helper to read, or with nil (or an empty URL) removes the file: the built-in model.
    public static func save(_ config: RemoteEmbedding?, file: URL = file) throws {
        guard var config, !config.url.trimmingCharacters(in: .whitespaces).isEmpty else {
            if FileManager.default.fileExists(atPath: file.path) { try FileManager.default.removeItem(at: file) }
            return
        }
        config.url = config.url.trimmingCharacters(in: .whitespaces)
        if config.apiKey?.isEmpty == true { config.apiKey = nil }
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(config).write(to: file, options: .atomic)
    }

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
                throw WorkerError.failed("DIGUP_EMBED_MEDIA must be messages or llama, not \(media)")
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

    /// What the Test button shows: whether the server gives usable vectors for text, a picture and a sound.
    public struct TestResult: Codable, Sendable, Equatable {
        public var ok: Bool
        public var message: String

        public init(ok: Bool, message: String) {
            self.ok = ok
            self.message = message
        }
    }

    /// One request for each kind DigUp sends: text, a small picture and a short sound, each waiting `timeout` seconds
    /// at most.
    public func test(timeout: TimeInterval = 20) -> TestResult {
        let embedder: RemoteEmbedder
        do {
            embedder = try RemoteEmbedder(self, textOnly: false, timeout: timeout)
        } catch {
            return TestResult(ok: false, message: reason(error))
        }
        defer { embedder.close() }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("digup-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let picture = folder.appendingPathComponent("test.png")
        let sound = folder.appendingPathComponent("test.wav")
        var failed: [String] = []
        do {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try Data(base64Encoded: Self.testPicture)!.write(to: picture)
            try Self.testSound().write(to: sound)
            if let error = try embedder.embedImages([picture.path], budget: 0).first?.error { failed.append("pictures: \(error)") }
            if let error = try embedder.embedAudio([sound.path]).first?.error { failed.append("sound: \(error)") }
        } catch {
            return TestResult(ok: false, message: reason(error))
        }
        guard failed.isEmpty else {
            return TestResult(ok: false, message: "Text works, but not " + failed.joined(separator: "; "))
        }
        return TestResult(ok: true, message: "Text, pictures and sound work.")
    }

    /// One second of a 440 Hz tone: 16 kHz, 16-bit mono WAV.
    static func testSound() -> Data {
        let rate = 16_000
        let samples = (0..<rate).map { Int16(8_000 * sin(2 * Double.pi * 440 * Double($0) / Double(rate))) }
        func le<T: FixedWidthInteger>(_ value: T) -> Data { withUnsafeBytes(of: value.littleEndian) { Data($0) } }
        var data = Data("RIFF".utf8) + le(UInt32(36 + samples.count * 2)) + Data("WAVEfmt ".utf8)
        data += le(UInt32(16)) + le(UInt16(1)) + le(UInt16(1)) + le(UInt32(rate)) + le(UInt32(rate * 2))
        data += le(UInt16(2)) + le(UInt16(16)) + Data("data".utf8) + le(UInt32(samples.count * 2))
        for sample in samples { data += le(sample) }
        return data
    }

    /// A 32x32 orange PNG.
    private static let testPicture = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKklEQVR42mM4UaFBU8QwasGoBaMWjFowasGoBaMWjFowasGoBaMWDBULAIvioExdDtm1AAAAAElFTkSuQmCC"

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
    /// Where a picture or sound goes in a llama-server prompt: newer servers pick a random marker and name it in
    /// `GET /props`.
    private var mediaMarker = "<__media__>"
    private static let textBatch = 32

    public init(_ config: RemoteEmbedding, textOnly: Bool, timeout: TimeInterval = 300) throws {
        let base = config.url.hasSuffix("/") ? String(config.url.dropLast()) : config.url
        guard let endpoint = URL(string: base + "/embeddings"), endpoint.scheme?.hasPrefix("http") == true else {
            throw WorkerError.failed("embedding server URL isn't http(s): \(config.url)")
        }
        let started = Date()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        session = URLSession(configuration: configuration)
        self.config = config
        self.endpoint = endpoint
        self.textOnly = textOnly
        info = config.info
        if config.media == .llama && !textOnly { mediaMarker = Self.mediaMarker(base: base, session: session) ?? mediaMarker }
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
            case .llama:
                body = ["input": [["prompt_string": mediaMarker, "multimodal_data": [data]]]]
            }
            guard let vector = try post(body).first else { throw WorkerError.failed("the server sent no vector") }
            return Embedding(vector: vector, error: nil)
        } catch {
            return Embedding(vector: nil, error: String(reason(error).prefix(300)))
        }
    }

    /// llama-server's `media_marker`, from `/props` at the server's root (the URL without `/v1`).
    private static func mediaMarker(base: String, session: URLSession) -> String? {
        struct Props: Decodable {
            let media_marker: String?
        }
        let root = base.hasSuffix("/v1") ? String(base.dropLast(3)) : base
        guard let url = URL(string: root + "/props"),
              case .success(let (data, response)) = send(URLRequest(url: url), on: session),
              (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return (try? JSONDecoder().decode(Props.self, from: data))?.media_marker
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
                let url = request.url?.absoluteString ?? ""
                box.result = .failure(WorkerError.failed(error.map { "can't reach \(url): \($0.localizedDescription)" }
                                                         ?? "no reply from \(url)"))
            }
            done.signal()
        }.resume()
        done.wait()
        return box.result!
    }
}

/// An error as the person reads it: a server's own words without the worker prefix.
private func reason(_ error: Error) -> String {
    if case WorkerError.failed(let why) = error { return why }
    return "\(error)"
}
