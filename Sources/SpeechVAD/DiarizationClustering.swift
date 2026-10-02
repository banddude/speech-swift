import Foundation

// MARK: - Scalable Constrained Agglomerative Clustering

// The clustering lives in its own Foundation-only file so it can be unit
// tested and benchmarked without loading the audio stack (the standalone
// harness and OfficeAdmin's offline test scripts compile this file alone).
//
// #2930: the previous implementation memoized an n*n Float distance matrix
// (650 MB at n = 12.7k) and rescanned every active pair on every merge, which
// is O(n^3) in comparisons and made a 71-minute recording take over an hour.
// Both entry points below keep the previous algorithm's exact greedy
// semantics (closest unconstrained pair, weighted-centroid linkage, cosine
// distance, same tie-breaks) but CAP the number of clustered items: past
// ``maxClusterItems`` the items are stride-sampled, the sample is clustered
// exactly, and every remaining item joins its nearest cluster centroid when
// that centroid is within the threshold (else it stays its own cluster).
// Memory and time are then bounded by the constant cap, not by the recording
// length: at the cap the matrix is 36 MB and the whole stage runs in seconds.

extension DiarizationHelpers {

    /// Item for constrained agglomerative clustering: one (window, local
    /// speaker) embedding crop.
    struct ClusterItem {
        let windowIndex: Int
        let localSpeakerId: Int
        let embedding: [Float]
    }

    /// Upper bound on items fed to the exact agglomerative clustering. A
    /// 71-minute recording at the coarse long-file step produces roughly 3-5k
    /// (window, speaker) embeddings; 3,000 keeps every speaker's crops well
    /// represented while bounding the distance matrix to ~36 MB.
    static let maxClusterItems = 3000


    /// Constrained agglomerative clustering with centroid linkage and cosine distance.
    ///
    /// Items from the same window can never be merged (same-window constraint).
    /// Merges closest unconstrained pair until distance exceeds threshold.
    /// Past ``maxClusterItems`` the items are stride-sampled first (see the
    /// file header).
    ///
    /// - Parameters:
    ///   - items: per-window per-speaker embeddings
    ///   - threshold: cosine distance threshold (0–2). Pairs with distance >= threshold are not merged.
    /// - Returns: cluster assignment for each item, and cluster centroids
    static func constrainedAgglomerativeClustering(
        items: [ClusterItem],
        threshold: Float
    ) -> (clusterAssignment: [Int], centroids: [[Float]]) {
        cappedClustering(items: items, threshold: threshold) { sample in
            constrainedAgglomerativeClusteringUncapped(items: sample, threshold: threshold)
        }
    }

    /// Agglomerative clustering with centroid linkage and cosine distance.
    ///
    /// Unlike ``constrainedAgglomerativeClustering(items:threshold:)``, this
    /// first estimates global speaker clusters without applying a transitive
    /// same-window cannot-link constraint. This mirrors the reference
    /// pyannote pipeline, which clusters globally and only constrains the later
    /// per-window assignment step. A false local track can therefore merge
    /// back into its real speaker instead of forcing a new global identity.
    /// Past ``maxClusterItems`` the items are stride-sampled first (see the
    /// file header).
    static func agglomerativeClustering(
        items: [ClusterItem],
        threshold: Float
    ) -> (clusterAssignment: [Int], centroids: [[Float]]) {
        cappedClustering(items: items, threshold: threshold) { sample in
            agglomerativeClusteringUncapped(items: sample, threshold: threshold)
        }
    }

    /// Stride-samples oversize inputs down to ``maxClusterItems``, clusters the
    /// sample with the exact algorithm, and sends every unsampled item to its
    /// nearest centroid when that centroid is within the threshold. Items with
    /// no centroid within the threshold stay their own cluster, exactly like
    /// an item the exact pass refused to merge.
    private static func cappedClustering(
        items: [ClusterItem],
        threshold: Float,
        cluster: ([DiarizationHelpers.ClusterItem])
            -> (clusterAssignment: [Int], centroids: [[Float]])
    ) -> (clusterAssignment: [Int], centroids: [[Float]]) {
        guard items.count > maxClusterItems else {
            return cluster(items)
        }

        let stride = Int((Double(items.count) / Double(maxClusterItems)).rounded(.up))
        var sampledIndices = [Int]()
        var index = 0
        while index < items.count {
            sampledIndices.append(index)
            index += stride
        }
        let sample = sampledIndices.map { items[$0] }
        let (sampleAssignment, centroids) = cluster(sample)
        guard !centroids.isEmpty else {
            // Degenerate sample (possible only for contrived inputs); fall
            // back to one cluster per item rather than an empty mapping.
            return (Array(items.indices), items.map(\.embedding))
        }

        // Every unsampled item joins its nearest centroid within the same
        // threshold the exact pass used, else stays its own cluster.
        var assignment = [Int](repeating: -1, count: items.count)
        var nextNewCluster = centroids.count
        for (position, sampled) in zip(sampledIndices, sampleAssignment) {
            assignment[position] = sampled
        }
        for i in items.indices where assignment[i] < 0 {
            var bestCluster = -1
            var bestDistance = threshold
            for (c, centroid) in centroids.enumerated() {
                let d = cosineDistance(items[i].embedding, centroid)
                if d < bestDistance {
                    bestDistance = d
                    bestCluster = c
                }
            }
            if bestCluster >= 0 {
                assignment[i] = bestCluster
            } else {
                assignment[i] = nextNewCluster
                nextNewCluster += 1
            }
        }
        return (assignment, centroids)
    }

    // MARK: - Constrained Agglomerative Clustering

    /// The exact constrained clustering, uncapped: the verbatim pre-#2930
    /// algorithm (memoized n*n distances, closest-pair scan per merge).
    static func constrainedAgglomerativeClusteringUncapped(
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

    /// The exact unconstrained clustering, uncapped: the verbatim pre-#2930
    /// algorithm (memoized n*n distances, closest-pair scan per merge).
    static func agglomerativeClusteringUncapped(
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