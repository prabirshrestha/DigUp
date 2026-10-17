import DigUpKit
import Observation
import SwiftUI

/// Where embeddings come from, as Settings → Model and onboarding edit it: the model on this Mac (the default) or an
/// OpenAI-compatible server (`RemoteEmbedding`). Edits stay a draft until `save()`; Test tries the draft.
@Observable
final class EmbedderForm {
    enum Choice: Hashable {
        case builtIn, server
    }

    /// The form's fields. Kept while you switch between the choices, so going back to the server keeps what you typed.
    struct Draft: Equatable {
        var choice: Choice
        var url: String
        var apiKey: String
        var model: String
        var media: RemoteEmbedding.Media

        init(_ saved: RemoteEmbedding?) {
            choice = saved == nil ? .builtIn : .server
            url = saved?.url ?? ""
            apiKey = saved?.apiKey ?? ""
            model = saved?.model ?? RemoteEmbedding.defaultModel
            media = saved?.media ?? .messages
        }

        var server: RemoteEmbedding {
            let model = model.trimmingCharacters(in: .whitespaces)
            return RemoteEmbedding(url: url.trimmingCharacters(in: .whitespaces),
                                   apiKey: apiKey.isEmpty ? nil : apiKey,
                                   model: model.isEmpty ? RemoteEmbedding.defaultModel : model, media: media)
        }

        /// What saving writes: nil for the model on this Mac.
        var config: RemoteEmbedding? { choice == .builtIn ? nil : server }
    }

    enum Test: Equatable {
        case idle
        case testing
        case result(RemoteEmbedding.TestResult)
    }

    var draft: Draft {
        didSet { if draft.config != oldValue.config { test = .idle } }   // a result is only about what was tested
    }
    private(set) var saved: RemoteEmbedding?
    private(set) var test = Test.idle
    /// Why the last save failed.
    private(set) var problem: String?
    /// Saves the setting and restarts what uses it (`AppController.setEmbedder`); nil when done, else why not.
    @ObservationIgnored private let apply: (RemoteEmbedding?) -> String?

    init(apply: @escaping (RemoteEmbedding?) -> String?) {
        self.apply = apply
        let saved = RemoteEmbedding.saved()
        self.saved = saved
        draft = Draft(saved)
    }

    var hasChanges: Bool { draft.config != saved }
    var canSave: Bool { hasChanges && (draft.choice == .builtIn || !draft.server.url.isEmpty) }
    var passed: Bool {
        if case .result(let result) = test { result.ok } else { false }
    }

    /// Back to what's saved (Settings opening again).
    func revert() {
        saved = RemoteEmbedding.saved()
        draft = Draft(saved)
        problem = nil
    }

    /// Sends one text, one small picture and one short sound to the draft's server, off the main thread (it waits on the network).
    func runTest() {
        let config = draft.server
        guard !config.url.isEmpty, test != .testing else { return }
        test = .testing
        Task.detached {
            let result = config.test()
            await MainActor.run { [weak self] in
                guard let self, draft.server == config else { return }   // edited meanwhile: the result is stale
                test = .result(result)
                log("embedder: test of \(config.url) (\(config.media.rawValue)): \(result.message)")
            }
        }
    }

    func save() {
        guard canSave else { return }
        problem = apply(draft.config)
        guard problem == nil else { return }
        saved = RemoteEmbedding.saved()
        draft.choice = saved == nil ? .builtIn : .server
    }
}

/// The model on this computer or a server, as two radio buttons, one under the other.
struct EmbedderChoicePicker: View {
    @Bindable var form: EmbedderForm

    var body: some View {
        Picker("Run the model", selection: $form.draft.choice) {
            Text("On this computer: downloads the model, \(Int((Double(ModelFiles.totalBytes) / 1e6).rounded())) MB")
                .tag(EmbedderForm.Choice.builtIn)
            Text("On a server: OpenAI-compatible, such as vLLM or llama-server").tag(EmbedderForm.Choice.server)
        }
        .pickerStyle(.radioGroup)
    }
}

/// The server's fields, Test and its result, and what using a server means, as rows of a `Form` (grouped in
/// Settings, columns in onboarding).
struct EmbedderServerFields: View {
    @Bindable var form: EmbedderForm

    var body: some View {
        TextField("Server URL", text: $form.draft.url, prompt: Text("http://spark.local:8080/v1"))
        SecureField("API key", text: $form.draft.apiKey, prompt: Text("Optional"))
        TextField("Model", text: $form.draft.model, prompt: Text(RemoteEmbedding.defaultModel))
        Picker(selection: $form.draft.media) {
            Text("Chat messages (vLLM)").tag(RemoteEmbedding.Media.messages)
            Text("Media prompt (llama-server)").tag(RemoteEmbedding.Media.llama)
        } label: {
            Text("Server type")
            Text("How pictures and sound are sent. Text is the same for both.")
        }
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Button("Test") { form.runTest() }
                .disabled(form.draft.server.url.isEmpty || form.test == .testing)
            switch form.test {
            case .idle:
                Text("Sends one text, one small picture and one short sound.")
                    .foregroundStyle(.secondary)
            case .testing:
                ProgressView().controlSize(.small)
                Text("Testing…").foregroundStyle(.secondary)
            case .result(let result):
                Image(systemName: result.ok ? "checkmark.circle.fill" : "xmark.octagon.fill")
                    .foregroundStyle(result.ok ? .green : .red)
                Text(result.message)
                    .foregroundStyle(result.ok ? AnyShapeStyle(.primary) : AnyShapeStyle(.red))
                    .lineLimit(3)   // a server's error can be long: all of it shows on hover
                    .fixedSize(horizontal: false, vertical: true)
                    .help(result.message)
                    .textSelection(.enabled)
            }
            Spacer(minLength: 0)
        }
        .font(.system(size: 12))
        Text("Your files' text, pictures and sound go to this server to be indexed, and your searches too. Use only a "
             + "server you trust.")
            .font(.system(size: 11))
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}
