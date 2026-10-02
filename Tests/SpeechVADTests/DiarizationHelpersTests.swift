import XCTest
@testable import SpeechVAD
import AudioCommon

final class DiarizationHelpersTests: XCTestCase {

    // MARK: - mergeSegments

    func testMergeSegmentsEmpty() {
        let result = DiarizationHelpers.mergeSegments([], minSilence: 0.15)
        XCTAssertTrue(result.isEmpty)
    }

    func testMergeSegmentsSingleSegment() {
        let segs = [DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 0)]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].startTime, 1.0, accuracy: 0.001)
        XCTAssertEqual(result[0].endTime, 2.0, accuracy: 0.001)
    }

    func testMergeSegmentsAdjacentSameSpeaker() {
        // Two segments from same speaker with small gap → should merge
        let segs = [
            DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 0),
            DiarizedSegment(startTime: 2.1, endTime: 3.0, speakerId: 0),
        ]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].startTime, 1.0, accuracy: 0.001)
        XCTAssertEqual(result[0].endTime, 3.0, accuracy: 0.001)
    }

    func testMergeSegmentsLargeGapNotMerged() {
        // Two segments from same speaker with large gap → should NOT merge
        let segs = [
            DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 0),
            DiarizedSegment(startTime: 3.0, endTime: 4.0, speakerId: 0),
        ]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        XCTAssertEqual(result.count, 2)
    }

    func testMergeSegmentsDifferentSpeakers() {
        // Adjacent segments from different speakers → should NOT merge
        let segs = [
            DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 0),
            DiarizedSegment(startTime: 2.05, endTime: 3.0, speakerId: 1),
        ]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        XCTAssertEqual(result.count, 2)
    }

    func testMergeSegmentsMultipleSpeakersInterleaved() {
        // Interleaved speakers with small gaps within each speaker
        let segs = [
            DiarizedSegment(startTime: 0.0, endTime: 1.0, speakerId: 0),
            DiarizedSegment(startTime: 1.0, endTime: 2.0, speakerId: 1),
            DiarizedSegment(startTime: 2.0, endTime: 3.0, speakerId: 0),
            DiarizedSegment(startTime: 3.0, endTime: 4.0, speakerId: 1),
        ]
        let result = DiarizationHelpers.mergeSegments(segs, minSilence: 0.15)
        // Speaker 0 has gap of 1.0s between segments → not merged
        // Speaker 1 has gap of 1.0s between segments → not merged
        XCTAssertEqual(result.count, 4)
        // Should be sorted by start time
        for i in 1..<result.count {
            XCTAssertGreaterThanOrEqual(result[i].startTime, result[i-1].startTime)
        }
    }

    // MARK: - compactSpeakerIds

    func testCompactSpeakerIdsEmpty() {
        let result = DiarizationHelpers.compactSpeakerIds([])
        XCTAssertTrue(result.isEmpty)
    }

    func testCompactSpeakerIdsAlreadyContiguous() {
        let segs = [
            DiarizedSegment(startTime: 0, endTime: 1, speakerId: 0),
            DiarizedSegment(startTime: 1, endTime: 2, speakerId: 1),
        ]
        let result = DiarizationHelpers.compactSpeakerIds(segs)
        XCTAssertEqual(result[0].speakerId, 0)
        XCTAssertEqual(result[1].speakerId, 1)
    }

    func testCompactSpeakerIdsWithGaps() {
        // Speaker IDs 2, 5 → should become 0, 1
        let segs = [
            DiarizedSegment(startTime: 0, endTime: 1, speakerId: 2),
            DiarizedSegment(startTime: 1, endTime: 2, speakerId: 5),
            DiarizedSegment(startTime: 2, endTime: 3, speakerId: 2),
        ]
        let result = DiarizationHelpers.compactSpeakerIds(segs)
        XCTAssertEqual(result[0].speakerId, 0)
        XCTAssertEqual(result[1].speakerId, 1)
        XCTAssertEqual(result[2].speakerId, 0)
    }

    func testCompactSpeakerIdsPreservesTimestamps() {
        let segs = [
            DiarizedSegment(startTime: 1.5, endTime: 3.7, speakerId: 10),
        ]
        let result = DiarizationHelpers.compactSpeakerIds(segs)
        XCTAssertEqual(result[0].startTime, 1.5, accuracy: 0.001)
        XCTAssertEqual(result[0].endTime, 3.7, accuracy: 0.001)
        XCTAssertEqual(result[0].speakerId, 0)
    }

    func testCompactSpeakerIdsKeepsCentroidsAlignedAcrossGaps() {
        let segs = [
            DiarizedSegment(startTime: 0, endTime: 1, speakerId: 2),
            DiarizedSegment(startTime: 1, endTime: 2, speakerId: 5),
            DiarizedSegment(startTime: 2, endTime: 3, speakerId: 2),
        ]
        let embeddings: [[Float]] = (0...5).map { speakerId in
            [Float(speakerId), Float(speakerId) + 0.5]
        }

        let result = DiarizationHelpers.compactSpeakerIdsAndEmbeddings(
            segs, speakerEmbeddings: embeddings)

        XCTAssertEqual(result.segments.map(\.speakerId), [0, 1, 0])
        XCTAssertEqual(result.speakerEmbeddings, [embeddings[2], embeddings[5]])
    }

    func testCompactSpeakerIdsPadsOnlyMissingActiveCentroid() {
        let segs = [
            DiarizedSegment(startTime: 0, endTime: 1, speakerId: 1),
            DiarizedSegment(startTime: 1, endTime: 2, speakerId: 4),
        ]

        let result = DiarizationHelpers.compactSpeakerIdsAndEmbeddings(
            segs,
            speakerEmbeddings: [[0, 0], [1, 1]],
            missingEmbeddingDimension: 2)

        XCTAssertEqual(result.segments.map(\.speakerId), [0, 1])
        XCTAssertEqual(result.speakerEmbeddings, [[1, 1], [0, 0]])
    }

    // MARK: - resample

    func testResampleSameRate() {
        let audio: [Float] = [1.0, 2.0, 3.0, 4.0]
        let result = DiarizationHelpers.resample(audio, from: 16000, to: 16000)
        XCTAssertEqual(result, audio)
    }

    func testResampleUpsample() {
        // Use longer signal for AVAudioConverter (short signals may have different frame counts)
        let audio: [Float] = (0..<100).map { Float($0) / 100.0 }
        let result = DiarizationHelpers.resample(audio, from: 8000, to: 16000)
        // 100 samples at 8kHz → ~200 samples at 16kHz
        XCTAssertGreaterThan(result.count, 150)
        XCTAssertLessThan(result.count, 250)
    }

    func testResampleDownsample() {
        let audio: [Float] = (0..<200).map { Float($0) / 200.0 }
        let result = DiarizationHelpers.resample(audio, from: 16000, to: 8000)
        // 200 samples at 16kHz → ~100 samples at 8kHz
        XCTAssertGreaterThan(result.count, 75)
        XCTAssertLessThan(result.count, 125)
    }

    func testResampleEmpty() {
        let result = DiarizationHelpers.resample([], from: 16000, to: 8000)
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - DiarizationConfig

    func testDiarizationConfigDefaultValues() {
        let config = DiarizationConfig.default
        XCTAssertEqual(config.onset, 0.5, accuracy: 0.001)
        XCTAssertEqual(config.offset, 0.3, accuracy: 0.001)
        XCTAssertEqual(config.minSpeechDuration, 0.3, accuracy: 0.001)
        XCTAssertEqual(config.minSilenceDuration, 0.15, accuracy: 0.001)
        XCTAssertEqual(config.clusteringThreshold, 0.715, accuracy: 0.001)
    }

    func testDiarizationConfigCustomValues() {
        let config = DiarizationConfig(
            onset: 0.7, offset: 0.4,
            minSpeechDuration: 0.5, minSilenceDuration: 0.2,
            clusteringThreshold: 0.5
        )
        XCTAssertEqual(config.onset, 0.7, accuracy: 0.001)
        XCTAssertEqual(config.offset, 0.4, accuracy: 0.001)
        XCTAssertEqual(config.minSpeechDuration, 0.5, accuracy: 0.001)
        XCTAssertEqual(config.minSilenceDuration, 0.2, accuracy: 0.001)
        XCTAssertEqual(config.clusteringThreshold, 0.5, accuracy: 0.001)
    }

    // MARK: - cosineDistance

    func testCosineDistanceSameVector() {
        let a: [Float] = [1, 0, 0]
        let dist = DiarizationHelpers.cosineDistance(a, a)
        XCTAssertEqual(dist, 0.0, accuracy: 0.001)
    }

    func testCosineDistanceOpposite() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [-1, 0, 0]
        let dist = DiarizationHelpers.cosineDistance(a, b)
        XCTAssertEqual(dist, 2.0, accuracy: 0.001)
    }

    func testCosineDistanceOrthogonal() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]
        let dist = DiarizationHelpers.cosineDistance(a, b)
        XCTAssertEqual(dist, 1.0, accuracy: 0.001)
    }

    func testCosineDistanceZeroVector() {
        let a: [Float] = [0, 0, 0]
        let b: [Float] = [1, 0, 0]
        let dist = DiarizationHelpers.cosineDistance(a, b)
        XCTAssertEqual(dist, 2.0, accuracy: 0.001)  // degenerate
    }

    // MARK: - constrainedAgglomerativeClustering

    func testClusteringEmpty() {
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: [], threshold: 0.5)
        XCTAssertTrue(assignment.isEmpty)
        XCTAssertTrue(centroids.isEmpty)
    }

    func testClusteringSingleItem() {
        let items = [DiarizationHelpers.ClusterItem(
            windowIndex: 0, localSpeakerId: 0, embedding: [1, 0, 0])]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 0.5)
        XCTAssertEqual(assignment, [0])
        XCTAssertEqual(centroids.count, 1)
    }

    func testClusteringSimilarEmbeddingsDifferentWindows() {
        // Two similar embeddings from different windows → should merge
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0,
                                           embedding: [1, 0, 0, 0]),
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 0,
                                           embedding: [0.99, 0.1, 0, 0]),
        ]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 0.5)

        // Should be merged into same cluster
        XCTAssertEqual(assignment[0], assignment[1])
        XCTAssertEqual(centroids.count, 1)
    }

    func testClusteringSameWindowNeverMerge() {
        // Two similar embeddings from SAME window → constraint prevents merge
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0,
                                           embedding: [1, 0, 0, 0]),
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 1,
                                           embedding: [0.99, 0.1, 0, 0]),
        ]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 2.0)  // Very permissive threshold

        // Must NOT merge despite being similar
        XCTAssertNotEqual(assignment[0], assignment[1])
        XCTAssertEqual(centroids.count, 2)
    }

    func testClusteringThresholdRespected() {
        // Two embeddings with cosine distance > threshold → should NOT merge
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0,
                                           embedding: [1, 0, 0, 0]),
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 0,
                                           embedding: [0, 1, 0, 0]),
        ]
        // Cosine distance = 1.0, threshold = 0.5 → should NOT merge
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 0.5)

        XCTAssertNotEqual(assignment[0], assignment[1])
        XCTAssertEqual(centroids.count, 2)
    }

    func testClusteringCentroidLinkage() {
        // After merging, centroid should be the weighted average
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0,
                                           embedding: [1, 0]),
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 0,
                                           embedding: [0.9, 0.1]),
        ]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 1.0)

        XCTAssertEqual(assignment[0], assignment[1])
        XCTAssertEqual(centroids.count, 1)
        // Centroid should be average: [(1+0.9)/2, (0+0.1)/2] = [0.95, 0.05]
        XCTAssertEqual(centroids[0][0], 0.95, accuracy: 0.001)
        XCTAssertEqual(centroids[0][1], 0.05, accuracy: 0.001)
    }

    func testClusteringTransitiveConstraints() {
        // Three items: A(win0), B(win1), C(win0)
        // A and B are similar (closest pair) → merge. Merged {A,B} inherits win0 from A,
        // so {A,B} cannot merge with C (both have win0).
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0,
                                           embedding: [1, 0, 0, 0]),  // A
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 0,
                                           embedding: [0.99, 0.01, 0, 0]),  // B (close to A)
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 1,
                                           embedding: [0, 0, 1, 0]),  // C (far from A and B)
        ]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 2.0)

        // A and B should merge (closest, different windows)
        XCTAssertEqual(assignment[0], assignment[1], "A and B should be in same cluster")
        // C cannot merge with {A,B} — both have window 0
        XCTAssertNotEqual(assignment[0], assignment[2], "C should not merge with {A,B}")
        XCTAssertEqual(centroids.count, 2)
    }

    func testClusteringUsesUpdatedCentroidDistances() {
        // Unit vectors at 0°, 20°, 50°. d(A,B)=0.060 merges first; the merged
        // centroid points at 10°, so d({A,B},C)=1-cos40°≈0.234 exceeds the
        // 0.2 threshold and C must stay out — even though d(B,C)=1-cos30°≈0.134
        // was below it. Centroid linkage requires the post-merge distance;
        // reusing B's pre-merge distance would degrade to single linkage.
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0,
                                           embedding: [1, 0]),                  // A: 0°
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 0,
                                           embedding: [0.9397, 0.3420]),        // B: 20°
            DiarizationHelpers.ClusterItem(windowIndex: 2, localSpeakerId: 0,
                                           embedding: [0.6428, 0.7660]),        // C: 50°
        ]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 0.2)

        XCTAssertEqual(assignment[0], assignment[1], "A and B should merge")
        XCTAssertNotEqual(assignment[2], assignment[0],
                          "C must be measured against the merged centroid, not its nearest member")
        XCTAssertEqual(centroids.count, 2)
    }

    func testClusteringMultipleSpeakers() {
        // 2 speakers across 3 windows — speaker embeddings are clearly separated
        let spk0: [Float] = [1, 0, 0, 0]
        let spk1: [Float] = [0, 0, 1, 0]
        let items = [
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 0, embedding: spk0),
            DiarizationHelpers.ClusterItem(windowIndex: 0, localSpeakerId: 1, embedding: spk1),
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 0, embedding: spk0),
            DiarizationHelpers.ClusterItem(windowIndex: 1, localSpeakerId: 1, embedding: spk1),
            DiarizationHelpers.ClusterItem(windowIndex: 2, localSpeakerId: 0, embedding: spk0),
            DiarizationHelpers.ClusterItem(windowIndex: 2, localSpeakerId: 1, embedding: spk1),
        ]
        let (assignment, centroids) = DiarizationHelpers.constrainedAgglomerativeClustering(
            items: items, threshold: 0.5)

        // Should produce exactly 2 clusters
        XCTAssertEqual(centroids.count, 2)

        // All spk0 items should be in same cluster
        XCTAssertEqual(assignment[0], assignment[2])
        XCTAssertEqual(assignment[0], assignment[4])

        // All spk1 items should be in same cluster
        XCTAssertEqual(assignment[1], assignment[3])
        XCTAssertEqual(assignment[1], assignment[5])

        // spk0 and spk1 should be in different clusters
        XCTAssertNotEqual(assignment[0], assignment[1])
    }

    // MARK: - DiarizationResult

    func testDiarizationResultEmpty() {
        let result = DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        XCTAssertTrue(result.segments.isEmpty)
        XCTAssertEqual(result.numSpeakers, 0)
        XCTAssertTrue(result.speakerEmbeddings.isEmpty)
    }

    // MARK: - Protocol Conformance

    func testPyannoteDiarizationPipelineTypealias() {
        // Verify the typealias compiles — DiarizationPipeline should be PyannoteDiarizationPipeline
        let _: DiarizationPipeline.Type = PyannoteDiarizationPipeline.self
    }

    func testSpeakerExtractionCapableConformance() {
        // PyannoteDiarizationPipeline conforms to SpeakerExtractionCapable
        XCTAssertTrue((PyannoteDiarizationPipeline.self as Any) is SpeakerExtractionCapable.Type)
    }

    func testSpeakerDiarizationModelConformance() {
        // PyannoteDiarizationPipeline conforms to SpeakerDiarizationModel
        XCTAssertTrue((PyannoteDiarizationPipeline.self as Any) is SpeakerDiarizationModel.Type)
    }

    #if canImport(CoreML)
    func testSortformerDiarizationModelConformance() {
        // SortformerDiarizer conforms to SpeakerDiarizationModel but NOT SpeakerExtractionCapable
        XCTAssertTrue((SortformerDiarizer.self as Any) is SpeakerDiarizationModel.Type)
        XCTAssertFalse((SortformerDiarizer.self as Any) is SpeakerExtractionCapable.Type)
    }
    #endif


    // MARK: - scalability + legacy equivalence (#2930)

    /// A verbatim copy of the pre-#2930 clustering (memoized n*n matrix plus a
    /// full active-pair scan per merge). The public entry points now cap the
    /// item count and stride-sample oversize inputs (see the DiarizationClustering
    /// file header), and the exact algorithm is what actually runs at or under
    /// the cap — so every case below uses sizes under ``maxClusterItems`` and
    /// must produce the identical partition, not a tolerance.
    private enum LegacyAgglomerative {

    /// Constrained agglomerative clustering with centroid linkage and cosine distance.
    ///
    /// Items from the same window can never be merged (same-window constraint).
    /// Merges closest unconstrained pair until distance exceeds threshold.
    ///
    /// - Parameters:
    ///   - items: per-window per-speaker embeddings
    ///   - threshold: cosine distance threshold (0–2). Pairs with distance >= threshold are not merged.
    /// - Returns: cluster assignment for each item, and cluster centroids
    static func constrainedAgglomerativeClustering(
        items: [DiarizationHelpers.ClusterItem],
        threshold: Float
    ) -> (clusterAssignment: [Int], centroids: [[Float]]) {
        guard !items.isEmpty else { return ([], []) }
        if items.count == 1 {
            return ([0], [items[0].embedding])
        }

        let n = items.count
        let dim = items[0].embedding.count

        // Each item starts as its own cluster
        var clusterOf = Array(0..<n)  // item → cluster ID
        var centroids = items.map { $0.embedding }  // cluster ID → centroid
        var clusterMembers = (0..<n).map { [$0] }  // cluster ID → member items
        // Window indices per cluster (for constraint checking)
        var clusterWindows = items.map { Set([$0.windowIndex]) }
        var active = Set(0..<n)

        // Memoized pairwise distances (flat n*n, symmetric). Constrained pairs
        // are stored as +inf. A pair's distance (and its constraint state) only
        // changes when one side merges, and the merged cluster's row/column is
        // recomputed below — so the cache is always exact and the merge order
        // is identical to the original recompute-every-iteration loop, at
        // O(n^2) total distance work instead of O(n^3).
        var dist = [Float](repeating: .infinity, count: n * n)
        for i in 0..<n {
            for j in (i + 1)..<n where items[i].windowIndex != items[j].windowIndex {
                let d = cosineDistance(centroids[i], centroids[j])
                dist[i * n + j] = d
                dist[j * n + i] = d
            }
        }

        while active.count > 1 {
            // Find closest unconstrained pair
            var bestDist: Float = Float.greatestFiniteMagnitude
            var bestI = -1, bestJ = -1

            let activeList = active.sorted()
            for ai in 0..<activeList.count {
                for aj in (ai + 1)..<activeList.count {
                    let ci = activeList[ai], cj = activeList[aj]
                    let d = dist[ci * n + cj]
                    if d < bestDist {
                        bestDist = d
                        bestI = ci
                        bestJ = cj
                    }
                }
            }

            guard bestDist < threshold && bestI >= 0 else { break }

            // Merge bestJ into bestI
            let sizeI = clusterMembers[bestI].count
            let sizeJ = clusterMembers[bestJ].count
            let totalSize = Float(sizeI + sizeJ)

            // Weighted average centroid
            var newCentroid = [Float](repeating: 0, count: dim)
            for d in 0..<dim {
                newCentroid[d] = (centroids[bestI][d] * Float(sizeI) + centroids[bestJ][d] * Float(sizeJ)) / totalSize
            }
            centroids[bestI] = newCentroid

            // Transfer members
            for member in clusterMembers[bestJ] {
                clusterOf[member] = bestI
            }
            clusterMembers[bestI].append(contentsOf: clusterMembers[bestJ])

            // Propagate window constraints
            clusterWindows[bestI].formUnion(clusterWindows[bestJ])

            active.remove(bestJ)

            // Refresh the merged cluster's cached distances (its centroid and
            // window set both changed); everything else is untouched.
            for other in active where other != bestI {
                let d: Float = clusterWindows[bestI].isDisjoint(with: clusterWindows[other])
                    ? cosineDistance(centroids[bestI], centroids[other])
                    : .infinity
                dist[bestI * n + other] = d
                dist[other * n + bestI] = d
            }
        }

        // Build final compact assignment
        let activeSorted = active.sorted()
        var clusterMap = [Int: Int]()  // old cluster ID → new compact ID
        for (newId, oldId) in activeSorted.enumerated() {
            clusterMap[oldId] = newId
        }

        let assignment = (0..<n).map { clusterMap[clusterOf[$0]]! }
        let finalCentroids = activeSorted.map { centroids[$0] }

        return (assignment, finalCentroids)
    }

    /// Agglomerative clustering with centroid linkage and cosine distance.
    ///
    /// Unlike ``constrainedAgglomerativeClustering(items:threshold:)``, this
    /// first estimates global speaker clusters without applying a transitive
    /// same-window cannot-link constraint. This mirrors the reference
    /// pyannote pipeline, which clusters globally and only constrains the later
    /// per-window assignment step. A false local track can therefore merge
    /// back into its real speaker instead of forcing a new global identity.
    static func agglomerativeClustering(
        items: [DiarizationHelpers.ClusterItem],
        threshold: Float
    ) -> (clusterAssignment: [Int], centroids: [[Float]]) {
        guard !items.isEmpty else { return ([], []) }
        if items.count == 1 {
            return ([0], [items[0].embedding])
        }

        let n = items.count
        let dim = items[0].embedding.count
        var clusterOf = Array(0..<n)
        var centroids = items.map(\.embedding)
        var clusterMembers = (0..<n).map { [$0] }
        var active = Set(0..<n)

        var dist = [Float](repeating: .infinity, count: n * n)
        for i in 0..<n {
            for j in (i + 1)..<n {
                let d = cosineDistance(centroids[i], centroids[j])
                dist[i * n + j] = d
                dist[j * n + i] = d
            }
        }

        while active.count > 1 {
            var bestDist = Float.greatestFiniteMagnitude
            var bestI = -1
            var bestJ = -1

            let activeList = active.sorted()
            for ai in 0..<activeList.count {
                for aj in (ai + 1)..<activeList.count {
                    let ci = activeList[ai]
                    let cj = activeList[aj]
                    let d = dist[ci * n + cj]
                    if d < bestDist {
                        bestDist = d
                        bestI = ci
                        bestJ = cj
                    }
                }
            }

            guard bestDist < threshold && bestI >= 0 else { break }

            let sizeI = clusterMembers[bestI].count
            let sizeJ = clusterMembers[bestJ].count
            let totalSize = Float(sizeI + sizeJ)
            var newCentroid = [Float](repeating: 0, count: dim)
            for d in 0..<dim {
                newCentroid[d] = (
                    centroids[bestI][d] * Float(sizeI)
                        + centroids[bestJ][d] * Float(sizeJ)
                ) / totalSize
            }
            centroids[bestI] = newCentroid

            for member in clusterMembers[bestJ] {
                clusterOf[member] = bestI
            }
            clusterMembers[bestI].append(contentsOf: clusterMembers[bestJ])
            active.remove(bestJ)

            for other in active where other != bestI {
                let d = cosineDistance(centroids[bestI], centroids[other])
                dist[bestI * n + other] = d
                dist[other * n + bestI] = d
            }
        }

        let activeSorted = active.sorted()
        let clusterMap = Dictionary(
            uniqueKeysWithValues: activeSorted.enumerated().map { ($1, $0) })
        let assignment = (0..<n).map { clusterMap[clusterOf[$0]]! }
        let finalCentroids = activeSorted.map { centroids[$0] }
        return (assignment, finalCentroids)
    }

    /// Cosine distance between two vectors: 1 - cosine_similarity.
    /// Returns value in [0, 2].
    static func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        guard n > 0 else { return 2.0 }

        var dot: Float = 0, normA: Float = 0, normB: Float = 0
        for i in 0..<n {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        let denom = sqrt(normA) * sqrt(normB)
        guard denom > 1e-10 else { return 2.0 }
        return 1.0 - dot / denom
    }
    }

    private struct SeededRandom {
        var state: UInt64
        init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
        mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func unit() -> Float {
            Float(next() >> 40) / Float(1 << 24)
        }
    }

    /// Deterministic clustered embeddings: `speakers` unit-ish vectors around
    /// distinct directions plus per-item noise.
    private func syntheticItems(
        count: Int, speakers: Int, dimension: Int, windows: Int, seed: UInt64
    ) -> [DiarizationHelpers.ClusterItem] {
        var rng = SeededRandom(seed: seed)
        var centers = [[Float]]()
        for s in 0..<speakers {
            var v = (0..<dimension).map { _ in rng.unit() - 0.5 }
            let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
            v = v.map { $0 / norm }
            centers.append(v)
        }
        return (0..<count).map { i in
            let speaker = i % speakers
            let embedding = centers[speaker].map { value in
                value + (rng.unit() - 0.5) * 0.04
            }
            return DiarizationHelpers.ClusterItem(
                windowIndex: i % windows, localSpeakerId: speaker, embedding: embedding)
        }
    }

    private func assertSamePartition(
        _ a: [Int], _ b: [Int], file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.count, b.count, "assignment lengths differ", file: file, line: line)
        var mapping = [Int: Int]()
        for (x, y) in zip(a, b) {
            if let expected = mapping[x] {
                XCTAssertEqual(expected, y, "partitions differ at some item", file: file, line: line)
            } else {
                mapping[x] = y
            }
        }
    }

    func testClusteringMatchesLegacyOnRandomInputs() {
        let cases: [(count: Int, speakers: Int, dimension: Int, windows: Int, threshold: Float)] = [
            (50, 3, 32, 12, 0.715),
            (200, 2, 64, 40, 0.5),
            (200, 4, 16, 40, 1.2),
            (150, 2, 8, 75, 0.715),
        ]
        for testCase in cases {
            let items = syntheticItems(
                count: testCase.count, speakers: testCase.speakers,
                dimension: testCase.dimension, windows: testCase.windows, seed: 42)
            let (assignment, centroids) = DiarizationHelpers.agglomerativeClustering(
                items: items, threshold: testCase.threshold)
            let (legacyAssignment, legacyCentroids) = LegacyAgglomerative.agglomerativeClustering(
                items: items, threshold: testCase.threshold)
            assertSamePartition(assignment, legacyAssignment)
            XCTAssertEqual(centroids.count, legacyCentroids.count)

            let constrained = DiarizationHelpers.constrainedAgglomerativeClustering(
                items: items, threshold: testCase.threshold)
            let legacyConstrained = LegacyAgglomerative.constrainedAgglomerativeClustering(
                items: items, threshold: testCase.threshold)
            assertSamePartition(constrained.clusterAssignment, legacyConstrained.clusterAssignment)
            XCTAssertEqual(constrained.centroids.count, legacyConstrained.centroids.count)
        }
    }

    func testClusteringStaysFastAndBoundedAtScale() {
        // 4,000 embeddings exceeds maxClusterItems, so this runs the capped
        // path (stride sample of 3,000 clustered exactly, remainder assigned
        // to their nearest centroid): bounded memory, seconds of runtime, and
        // still three clean speakers. The pre-#2930 implementation needed a
        // 64 MB matrix and ~10^10 pair visits here.
        let items = syntheticItems(count: 4000, speakers: 3, dimension: 256, windows: 1300, seed: 7)
        let start = Date()
        let (assignment, centroids) = DiarizationHelpers.agglomerativeClustering(
            items: items, threshold: 0.715)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertEqual(assignment.count, items.count)
        XCTAssertEqual(Set(assignment).count, 3)
        XCTAssertEqual(centroids.count, 3)
        XCTAssertLessThan(elapsed, 60.0, "clustering regressed to a super-quadratic scan")
    }
}
