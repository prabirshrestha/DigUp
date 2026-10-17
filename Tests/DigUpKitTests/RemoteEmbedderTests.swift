import Foundation
import Testing
@testable import DigUpKit

@Suite struct RemoteEmbedderTests {
    private func file(_ json: String?) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("embedder-\(UUID().uuidString).json")
        if let json { try Data(json.utf8).write(to: url) }
        return url
    }

    @Test func localWithoutAURL() throws {
        #expect(try RemoteEmbedding.load(file: file(nil), environment: [:]) == nil)
        #expect(try RemoteEmbedding.load(file: file(#"{"url": "  "}"#), environment: [:]) == nil)
    }

    @Test func fileThenEnvironment() throws {
        let config = try file(#"{"url": "http://spark:8000/v1", "api_key": "k", "media": "llama"}"#)
        let fromFile = try #require(try RemoteEmbedding.load(file: config, environment: [:]))
        #expect(fromFile == RemoteEmbedding(url: "http://spark:8000/v1", apiKey: "k", model: RemoteEmbedding.defaultModel,
                                            media: .llama))
        let overridden = try #require(try RemoteEmbedding.load(file: config, environment: [
            "DIGUP_EMBED_URL": "http://other/v1", "DIGUP_EMBED_MODEL": "m", "DIGUP_EMBED_MEDIA": "none",
        ]))
        #expect(overridden == RemoteEmbedding(url: "http://other/v1", apiKey: "k", model: "m", media: .off))
    }

    @Test func brokenFileIsAnError() throws {
        #expect(throws: WorkerError.self) { try RemoteEmbedding.load(file: file("{"), environment: [:]) }
        #expect(throws: WorkerError.self) {
            try RemoteEmbedding.load(file: file(nil), environment: ["DIGUP_EMBED_URL": "http://x", "DIGUP_EMBED_MEDIA": "video"])
        }
    }

    @Test func sourceIgnoresTheHost() {
        #expect(RemoteEmbedding(url: "http://a/v1").info.vectorSource == RemoteEmbedding(url: "http://b/v1").info.vectorSource)
    }

    @Test func vectorsAreCutAndNormalized() throws {
        let long = (0..<1024).map { Float($0 % 7) + 1 }
        let vector = try RemoteEmbedder.normalized(long)
        #expect(vector.count == Vectors.dimension)
        #expect(abs(vector.reduce(0) { $0 + $1 * $1 } - 1) < 1e-4)
        #expect(throws: WorkerError.self) { try RemoteEmbedder.normalized([Float](repeating: 1, count: 384)) }
    }
}
