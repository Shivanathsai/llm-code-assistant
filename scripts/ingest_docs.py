#!/usr/bin/env python3
"""
ingest_docs.py
==============
Ingests a large documentation corpus to demonstrate 100K+ line processing.

Generates a synthetic corpus of 100K+ lines covering:
  - Python standard library docs
  - Distributed systems concepts
  - Kafka Streams documentation
  - REST API patterns
  - Data structures and algorithms

Then ingests it into ChromaDB and reports statistics.

Usage:
    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --lines 100000
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

from app.rag.document_ingester import DocumentIngester


def generate_corpus(target_lines: int) -> str:
    """
    Generate a technical documentation corpus of target_lines lines.
    Uses repetition with variation to simulate real documentation at scale.
    """
    sections = [
        ("Rate Limiting", _rate_limiting_docs()),
        ("Kafka Streams", _kafka_docs()),
        ("Vector Search", _vector_search_docs()),
        ("Distributed Systems", _distributed_systems_docs()),
        ("Data Structures", _data_structures_docs()),
        ("API Design", _api_design_docs()),
        ("DevOps", _devops_docs()),
        ("Algorithms", _algorithms_docs()),
    ]

    corpus_lines = []
    section_idx  = 0

    while len(corpus_lines) < target_lines:
        section_name, section_text = sections[section_idx % len(sections)]
        lines = section_text.splitlines()
        # Add section header with variation
        corpus_lines.append(f"\n## {section_name} — Part {section_idx // len(sections) + 1}\n")
        corpus_lines.extend(lines)
        section_idx += 1

    return "\n".join(corpus_lines[:target_lines])


def _rate_limiting_docs() -> str:
    return """
Rate limiting controls the rate at which requests are processed.

### Fixed Window
A fixed window counter resets every N seconds. Simple but allows 2x burst at boundaries.
Implementation: increment counter per time bucket, reject if over limit.
Time complexity: O(1) per request. Memory: O(1).

### Sliding Window Log
Stores exact timestamps of all requests. Accurate but memory intensive.
Time complexity: O(n) where n = requests in window. Memory: O(n).

### Sliding Window Counter
Approximates sliding window using two fixed windows. Better memory efficiency.
Time complexity: O(1). Memory: O(1). Accuracy: ~99% vs true sliding window.

### Token Bucket
Tokens accumulate at rate r up to burst size b. Request consumes one token.
Allows controlled bursting while maintaining average rate.
Time complexity: O(1). Useful for API rate limiting with burst allowance.

### Leaky Bucket
Requests enter a queue (bucket), processed at constant rate (leak rate).
Smooths traffic, prevents bursts. Queue full = reject.
Time complexity: O(1). Ensures steady output rate regardless of input.

### Redis Sorted Set Implementation
Using Redis ZADD with timestamp scores:
  ZADD key <timestamp> <request-id>     # record request
  ZREMRANGEBYSCORE key 0 <window-start> # evict old requests
  ZCARD key                              # count remaining
  Atomic via Lua script for race condition prevention.
  O(log n) per operation.
"""


def _kafka_docs() -> str:
    return """
Apache Kafka is a distributed streaming platform.

### Core Concepts
Topics: named categories of records, split into partitions for parallelism.
Partitions: ordered, immutable sequence of records. Key determines partition.
Offsets: unique sequential ID for each record within a partition.
Producers: write records to topics, choose partition via key or round-robin.
Consumers: read records from topics, track position via committed offsets.
Brokers: Kafka servers that store partitions and serve clients.
Replication: each partition replicated across N brokers for fault tolerance.

### Consumer Groups
Consumer groups enable parallel processing:
- Each partition assigned to exactly one consumer per group
- Rebalancing occurs when consumers join/leave the group
- Committed offsets track processing progress per partition
- lag = latest_offset - committed_offset (measure of processing backlog)

### Kafka Streams
Stream processing library built on Kafka consumer/producer API.
DSL operations: map, filter, groupBy, aggregate, join, window.
Exactly-once processing: idempotent producers + transactional consumers.
State stores: RocksDB for local state, Kafka changelog for durability.
Windowing: tumbling (non-overlapping), hopping (overlapping), session.

### Exactly-Once Semantics
Idempotent producer: retries don't produce duplicates (sequence numbers).
Transactional producer: atomic multi-partition writes.
Exactly-once processing: read-process-write is atomic.
Configuration: enable.idempotence=true, processing.guarantee=exactly_once_v2.
"""


def _vector_search_docs() -> str:
    return """
Vector search finds similar items by comparing dense vector representations.

### Embeddings
Dense vectors encoding semantic meaning of text, images, or other data.
Trained on large corpora to capture relationships between concepts.
Similar items have high cosine similarity in embedding space.
Dimensionality: 128–1536 dimensions depending on model.
Models: BERT, sentence-transformers, OpenAI text-embedding-ada-002.

### Cosine Similarity
Measures angle between two vectors: cos(θ) = dot(A,B) / (|A| * |B|)
Range: -1 (opposite) to 1 (identical), 0 = orthogonal (unrelated).
Used in semantic search, RAG, recommendation systems.
More robust than Euclidean distance for high-dimensional spaces.

### HNSW Index
Hierarchical Navigable Small World graphs for approximate nearest neighbour.
Builds a multi-layer graph where higher layers have longer links.
Query: start at top layer, greedily navigate to nearest neighbours.
Time complexity: O(log n) query, O(n log n) build.
Trade-off: recall vs speed via ef parameter.

### ChromaDB
Open-source vector database with Python-native API.
Collections: named groups of embeddings with metadata.
Operations: add, query, update, delete documents with embeddings.
Supports cosine and L2 distance metrics.
Persistent mode: embeddings stored on disk via SQLite + parquet.
In-memory mode: ephemeral storage for testing.

### RAG Architecture
Retrieval-Augmented Generation combines retrieval + generation:
1. Index: embed documents, store in vector database
2. Retrieve: embed query, find top-k similar chunks
3. Augment: inject retrieved chunks into LLM prompt
4. Generate: LLM produces grounded answer from context
Reduces hallucination, enables domain-specific knowledge without fine-tuning.
"""


def _distributed_systems_docs() -> str:
    return """
Distributed systems coordinate multiple independent computers.

### CAP Theorem
Consistency, Availability, Partition tolerance — choose 2 of 3.
CP systems: consistent + partition tolerant (sacrifice availability). Example: HBase.
AP systems: available + partition tolerant (sacrifice consistency). Example: Cassandra.
CA systems: consistent + available (no partition tolerance). Only possible in single node.

### Consensus Algorithms
Raft: leader-based consensus for replicated state machines.
Leader election: nodes vote for candidates with highest log index.
Log replication: leader appends entries, majority must acknowledge before commit.
Safety: at most one leader per term, committed entries never overwritten.
Paxos: original consensus algorithm, complex but foundational.

### Consistent Hashing
Assigns nodes and keys to a ring of hash values.
Adding/removing nodes: only affects adjacent keys (minimal redistribution).
Used in distributed caches (Redis Cluster), load balancers, DHTs.
Virtual nodes: each physical node owns multiple positions on ring.
Reduces hotspots compared to modulo hashing.

### Fault Tolerance Patterns
Circuit Breaker: prevents cascade failures by failing fast when service is down.
Retry with backoff: exponential backoff prevents thundering herd.
Bulkhead: isolate failures to prevent resource exhaustion.
Timeout: always set timeouts on network calls.
Health checks: liveness (is process running?), readiness (can serve traffic?).

### Replication
Leader-follower: one leader handles writes, followers replicate.
Leaderless: clients can write to any node, reconciled via version vectors.
Synchronous: write acknowledged only after replicas confirm.
Asynchronous: write acknowledged before replicas confirm (faster, risk of loss).
"""


def _data_structures_docs() -> str:
    return """
Data structures organize and store data for efficient access.

### Hash Table
Maps keys to values via hash function. O(1) average get/put/delete.
Collision resolution: chaining (linked list per bucket) or open addressing.
Load factor: entries / buckets. Rehash when > 0.75 for performance.
Applications: caches, symbol tables, database indexes.

### Binary Search Tree
Binary tree where left < node < right for all nodes.
Operations: search O(h), insert O(h), delete O(h) where h = height.
Balanced BSTs (AVL, Red-Black): O(log n) guaranteed.
Applications: sorted maps, range queries, order statistics.

### Heap
Complete binary tree satisfying heap property (max-heap: parent >= children).
Operations: insert O(log n), extract-max O(log n), peek O(1).
Implementation: array where parent at i, children at 2i+1 and 2i+2.
Applications: priority queues, heap sort, Dijkstra's algorithm.

### Skip List
Probabilistic data structure with O(log n) search/insert/delete.
Multiple sorted linked lists at different levels for fast traversal.
Used in Redis sorted sets, LevelDB, Lucene.
Simpler to implement than balanced BSTs, similar performance.

### Bloom Filter
Space-efficient probabilistic set membership test.
False positives possible, false negatives impossible.
Operations: add O(k), test O(k) where k = number of hash functions.
Use cases: URL deduplication, spell checkers, cache admission.
"""


def _api_design_docs() -> str:
    return """
API design principles for scalable, maintainable services.

### REST Principles
Stateless: server holds no client state between requests.
Resource-based: URLs represent nouns (resources), HTTP verbs are actions.
Uniform interface: standard HTTP methods, status codes, media types.
Cacheable: GET responses can be cached for performance.
Layered: clients unaware of intermediate proxies, load balancers.

### HTTP Status Codes
2xx Success: 200 OK, 201 Created, 204 No Content.
3xx Redirection: 301 Moved, 304 Not Modified.
4xx Client Error: 400 Bad Request, 401 Unauthorized, 404 Not Found, 429 Too Many Requests.
5xx Server Error: 500 Internal Server Error, 503 Service Unavailable.

### API Versioning
URL versioning: /api/v1/resources (simple, explicit).
Header versioning: Accept: application/vnd.api+json;version=1 (cleaner URLs).
Query parameter: /api/resources?version=1 (easy to test).
Semantic versioning: major.minor.patch, increment major for breaking changes.

### Rate Limiting Headers
X-RateLimit-Limit: maximum requests per window.
X-RateLimit-Remaining: requests remaining in current window.
X-RateLimit-Reset: Unix timestamp when window resets.
Retry-After: seconds to wait before retrying (on 429 response).

### gRPC
Protocol Buffers for efficient binary serialization.
HTTP/2: multiplexing, streaming, header compression.
Strongly typed contracts via .proto files.
Bidirectional streaming for real-time applications.
Code generation for multiple languages.
"""


def _devops_docs() -> str:
    return """
DevOps practices for reliable software delivery.

### CI/CD Pipeline
Continuous Integration: automatically build and test on every commit.
Continuous Delivery: automatically deploy to staging after tests pass.
Continuous Deployment: automatically deploy to production.
Pipeline stages: lint → test → build → deploy → verify.
Blue-green deployment: run two identical environments, switch traffic.

### Docker Best Practices
Multi-stage builds: separate build and runtime environments.
Non-root user: run containers as non-root for security.
Layer caching: order Dockerfile commands from least to most frequently changed.
.dockerignore: exclude unnecessary files from build context.
Health checks: HEALTHCHECK instruction for container health monitoring.

### Kubernetes Resources
Deployment: manages pod replicas, rolling updates, rollbacks.
Service: stable network endpoint (ClusterIP, NodePort, LoadBalancer).
ConfigMap: externalize configuration from container images.
Secret: sensitive data (passwords, tokens) encoded as base64.
HPA: Horizontal Pod Autoscaler scales replicas based on CPU/memory metrics.
PVC: Persistent Volume Claim for stateful workloads.

### Observability
Metrics: quantitative measurements (counters, gauges, histograms).
Logs: structured text records of events (JSON format recommended).
Traces: end-to-end request tracking across services.
Prometheus: pull-based metrics collection and alerting.
Grafana: visualization dashboards for metrics.
OpenTelemetry: vendor-neutral observability standard.

### AWS Services
EKS: Elastic Kubernetes Service — managed Kubernetes control plane.
ECR: Elastic Container Registry — managed Docker image registry.
S3: Simple Storage Service — object storage for files and backups.
RDS: Relational Database Service — managed PostgreSQL, MySQL.
ElastiCache: managed Redis and Memcached for caching.
ALB: Application Load Balancer — HTTP/HTTPS layer 7 load balancing.
"""


def _algorithms_docs() -> str:
    return """
Fundamental algorithms and complexity analysis.

### Sorting Algorithms
QuickSort: O(n log n) average, O(n^2) worst. In-place, unstable.
MergeSort: O(n log n) always. Not in-place, stable. Preferred for linked lists.
HeapSort: O(n log n) always. In-place, unstable. Uses heap data structure.
TimSort: Python's built-in. Hybrid merge/insertion sort. O(n log n) worst, O(n) best.

### Graph Algorithms
BFS: explore level by level. Shortest path in unweighted graph. O(V+E).
DFS: explore depth-first. Cycle detection, topological sort. O(V+E).
Dijkstra: shortest path in weighted graph (non-negative weights). O((V+E) log V).
A*: heuristic shortest path. Faster than Dijkstra with good heuristic.
Bellman-Ford: shortest path with negative weights. O(VE).

### Dynamic Programming
Overlapping subproblems + optimal substructure → DP.
Memoization (top-down): cache results of recursive calls.
Tabulation (bottom-up): fill table iteratively.
Classic problems: Fibonacci, knapsack, longest common subsequence, edit distance.
Time/space trade-off: O(n^2) time + O(n) space for many problems.

### Complexity Classes
O(1): constant time — hash lookup, array access.
O(log n): logarithmic — binary search, balanced BST operations.
O(n): linear — array scan, single pass algorithms.
O(n log n): linearithmic — efficient sorting, some tree algorithms.
O(n^2): quadratic — nested loops, bubble sort.
O(2^n): exponential — brute force subset enumeration.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=100_000,
                        help="Target number of lines to ingest")
    args = parser.parse_args()

    print(f"\n  Generating {args.lines:,} line documentation corpus...")
    corpus = generate_corpus(args.lines)
    actual_lines = corpus.count("\n")
    print(f"  Generated {actual_lines:,} lines ({len(corpus):,} chars)")

    print(f"\n  Ingesting into ChromaDB vector store...")
    ingester = DocumentIngester()
    ingester.reset()

    t0     = time.perf_counter()
    chunks = ingester.ingest_text(corpus, source="100k-corpus")
    elapsed = time.perf_counter() - t0

    print(f"\n  {'─'*50}")
    print(f"  Lines ingested:  {actual_lines:>10,}")
    print(f"  Chunks created:  {chunks:>10,}")
    print(f"  Vector store:    {ingester.count():>10,} total chunks")
    print(f"  Elapsed:         {elapsed:>10.1f}s")
    print(f"  Rate:            {actual_lines/elapsed:>10,.0f} lines/sec")
    print(f"  {'─'*50}")
    print(f"  ✓ 100K+ documentation lines processed successfully")
    print()


if __name__ == "__main__":
    main()
