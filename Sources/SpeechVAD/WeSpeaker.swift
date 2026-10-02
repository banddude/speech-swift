import Foundation
import MLXCommon
import MLX
import AudioCommon

#if canImport(CoreML)
import CoreML
#endif

/// Inference engine for WeSpeaker speaker embeddings.
public enum WeSpeakerEngine: String, Sendable {
    /// MLX backend — runs on GPU via Metal shaders.
    case mlx
    /// CoreML backend — runs on Neural Engine + CPU, freeing the GPU.
    case coreml
}

/// Speaker embedding model using WeSpeaker ResNet34-LM.
///
/// Produces 256-dimensional L2-normalized speaker embeddings from audio.
/// Uses 80-dim log-mel features at 16kHz.
///
/// Supports two backends:
/// - `.mlx`: GPU-based inference via MLX (default)
/// - `.coreml`: Neural Engine inference via CoreML (lower power, frees GPU)
///
/// This class is thread-safe: all properties are immutable after construction and
/// inference is pure computation with no mutable state. `MLModel.prediction(from:)` is
/// documented as thread-safe by Apple.
///
/// ```swift
/// let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
/// let embedding = model.embed(audio: samples, sampleRate: 16000)
/// // embedding: [Float] of length 256
/// ```
public final class WeSpeakerModel {

    /// The inference engine in use.
    public let engine: WeSpeakerEngine

    /// Whether the model weights are loaded and ready for inference.
    var _isLoaded = true

    /// The ResNet34 network (nil when using CoreML engine)
    let network: WeSpeakerNetwork?

    /// Mel feature extractor
    let melExtractor: MelFeatureExtractor

    #if canImport(CoreML)
    /// CoreML compiled model (nil when using MLX engine)
    var coremlModel: MLModel?
    #endif

    /// Default HuggingFace model ID (MLX weights)
    public static let defaultModelId = "aufklarer/WeSpeaker-ResNet34-LM-MLX"

    /// Default HuggingFace model ID (CoreML weights)
    public static let defaultCoreMLModelId = "aufklarer/WeSpeaker-ResNet34-LM-CoreML"

    /// Embedding dimension
    public let embeddingDimension: Int = 256

    /// Expected input sample rate
    public let inputSampleRate: Int = 16000

    /// Enumerated mel frame lengths supported by the CoreML model.
    static let enumeratedMelLengths = [20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]

    init(network: WeSpeakerNetwork) {
        self.engine = .mlx
        self.network = network
        self.melExtractor = MelFeatureExtractor()
        #if canImport(CoreML)
        self.coremlModel = nil
        #endif
    }

    #if canImport(CoreML)
    init(coremlModel: MLModel) {
        self.engine = .coreml
        self.network = nil
        self.coremlModel = coremlModel
        self.melExtractor = MelFeatureExtractor()
    }
    #endif

    /// Load a pre-trained speaker embedding model from HuggingFace.
    ///
    /// Downloads model weights on first use, then caches locally.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID (auto-selected by engine if not specified)
    ///   - engine: inference backend (`.mlx` or `.coreml`)
    ///   - progressHandler: callback for download progress
    /// - Returns: ready-to-use speaker embedding model
    public static func fromPretrained(
        modelId: String? = nil,
        engine: WeSpeakerEngine = .mlx,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> WeSpeakerModel {
        let resolvedModelId = modelId ?? (engine == .coreml ? defaultCoreMLModelId : defaultModelId)

        progressHandler?(0.0, "Downloading speaker embedding model...")

        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: resolvedModelId)

        switch engine {
        case .mlx:
            try await HuggingFaceDownloader.downloadWeights(
                modelId: resolvedModelId,
                to: cacheDir,
                offlineMode: offlineMode,
                progressHandler: { progress in
                    progressHandler?(progress * 0.8, "Downloading weights...")
                }
            )

            progressHandler?(0.8, "Loading model...")

            let network = WeSpeakerNetwork()
            try WeSpeakerWeightLoader.loadWeights(model: network, from: cacheDir)

            progressHandler?(1.0, "Ready")
            return WeSpeakerModel(network: network)

        case .coreml:
            #if canImport(CoreML)
            try await HuggingFaceDownloader.downloadWeights(
                modelId: resolvedModelId,
                to: cacheDir,
                additionalFiles: ["wespeaker.mlmodelc/**", "config.json"],
                offlineMode: offlineMode,
                progressHandler: { progress in
                    progressHandler?(progress * 0.8, "Downloading CoreML model...")
                }
            )

            progressHandler?(0.8, "Loading CoreML model...")

            let modelURL = cacheDir.appendingPathComponent("wespeaker.mlmodelc", isDirectory: true)
            guard FileManager.default.fileExists(atPath: modelURL.path) else {
                throw AudioModelError.modelLoadFailed(
                    modelId: resolvedModelId,
                    reason: "CoreML model not found at \(modelURL.path)")
            }

            let mlConfig = MLModelConfiguration()
            mlConfig.computeUnits = CoreMLComputeUnitsResolver.resolved(default: .cpuAndNeuralEngine)

            let model: MLModel
            do {
                model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
            } catch {
                throw AudioModelError.modelLoadFailed(
                    modelId: resolvedModelId,
                    reason: "Failed to load CoreML model",
                    underlying: error)
            }

            progressHandler?(1.0, "Ready")
            return WeSpeakerModel(coremlModel: model)
            #else
            throw AudioModelError.invalidConfiguration(
                model: "WeSpeaker", reason: "CoreML not available on this platform")
            #endif
        }
    }

    /// Extract a 256-dimensional speaker embedding from audio.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: sample rate of the input audio
    /// - Returns: 256-dim L2-normalized speaker embedding
    public func embed(audio: [Float], sampleRate: Int) -> [Float] {
        let samples: [Float]
        if sampleRate != inputSampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: inputSampleRate)
        } else {
            samples = audio
        }

        switch engine {
        case .mlx:
            return embedMLX(samples)
        case .coreml:
            #if canImport(CoreML)
            let (melSpec, nFrames) = melExtractor.extractRaw(samples)
            return (try? embedCoreML(melSpec: melSpec, nFrames: nFrames)) ?? [Float](repeating: 0, count: embeddingDimension)
            #else
            fatalError("CoreML not available on this platform")
            #endif
        }
    }

    /// MLX inference path.
    private func embedMLX(_ samples: [Float]) -> [Float] {
        guard let network else { fatalError("MLX network not loaded") }

        // Extract mel features: [T, 80]
        let mel = melExtractor.extract(samples)

        // Add batch and channel dims: [1, T, 80, 1]
        let input = mel.reshaped(1, mel.dim(0), mel.dim(1), 1)

        // Forward pass: [1, 256]
        let emb = network(input)
        eval(emb)

        return emb[0].asArray(Float.self)
    }

    /// Extract 256-dimensional speaker embeddings for a batch of clips in one
    /// forward pass per group (#2930: diarization used to run one forward pass
    /// per (window, speaker) crop, which is what made long recordings take
    /// hours — the model spends most of a small forward pass on dispatch
    /// overhead, not math).
    ///
    /// Clips are grouped by similar length so the zero padding added to make
    /// the batch rectangular stays small, and each row's statistics pooling is
    /// masked to its valid frames (see ``WeSpeakerNetwork.callAsFunction``),
    /// which reproduces ``embed(audio:sampleRate:)`` for every clip.
    ///
    /// - Parameters:
    ///   - audios: PCM Float32 clips
    ///   - sampleRate: sample rate of the input clips
    /// - Returns: one 256-dim L2-normalized embedding per input clip, in input order
    public func embedBatched(_ audios: [[Float]], sampleRate: Int) -> [[Float]] {
        guard !audios.isEmpty else { return [] }
        guard let network else { fatalError("MLX network not loaded") }

        struct Clip {
            let mel: MLXArray
            let frames: Int
        }

        // Extract mel per clip, then order by length: a batch of similar
        // lengths pads far less than the input order would.
        var clips: [Clip] = audios.map { audio in
            let samples = sampleRate == inputSampleRate
                ? audio
                : AudioFileLoader.resample(audio, from: sampleRate, to: inputSampleRate)
            let mel = melExtractor.extract(samples)
            return Clip(mel: mel, frames: mel.dim(0))
        }
        let byLength = clips.indices.sorted { clips[$0].frames > clips[$1].frames }

        var embeddings = [[Float]](repeating: [], count: audios.count)
        let groupSize = 32
        // Cap the FORWARD PASS by total mel frames, not just crop count: the
        // ResNet's activations scale with batch x frames, so 32 ten-second
        // crops would hold several GB of intermediate buffers at once (and
        // blew MLX's peak past 5 GB on a 36-minute file). ~8,000 frames is
        // 80 seconds of audio, the work of eight batch-of-1 passes, with a
        // peak that stays flat as recordings get longer.
        let maxFramesPerGroup = 8_000

        var position = 0
        while position < byLength.count {
            var end = position
            var frames = 0
            while end < byLength.count,
                  end - position < groupSize,
                  frames + clips[byLength[end]].frames <= maxFramesPerGroup {
                frames += clips[byLength[end]].frames
                end += 1
            }
            if end == position {
                // A single clip longer than the budget still runs alone.
                end = position + 1
            }
            let group = byLength[position..<end]
            position = end

            let maxFrames = group.map { clips[$0].frames }.max() ?? 0
            var flat = [Float](repeating: 0, count: group.count * maxFrames * 80)
            for (row, index) in group.enumerated() {
                let mel = clips[index].mel
                let frames = clips[index].frames
                let values = mel.asArray(Float.self)
                flat.replaceSubrange(
                    row * maxFrames * 80..<row * maxFrames * 80 + frames * 80,
                    with: values)
            }
            var validFrames = [Int]()
            for index in group { validFrames.append(clips[index].frames) }

            let input = MLXArray(flat, [group.count, maxFrames, 80, 1])
            let emb = network(input, validFrames: validFrames)
            eval(emb)
            for (row, index) in group.enumerated() {
                embeddings[index] = emb[row].asArray(Float.self)
            }
        }

        return embeddings
    }

    /// Compute cosine similarity between two embeddings.
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom > 0 ? dot / denom : 0
    }

}
