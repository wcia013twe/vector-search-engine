# Vector Search Engine

A CLIP-based image retrieval system over ImageNet-1K, exploring the tradeoffs between exact nearest-neighbor search, inverted-file indexing, and product quantization.

## Project status

**Current:** Academic prototype and benchmark over 1,281,167 ImageNet training images.

**Next:** Package the pipeline into reusable modules, expose retrieval through an API, and add reproducible serving and concurrency benchmarks. The planned API and deployment components are not yet implemented.

## Goals

- Generate normalized 512-dimensional image embeddings with CLIP ViT-B/32.
- Compare exact and approximate FAISS indexes.
- Quantify retrieval-quality, latency, and serialized-index-size tradeoffs.
- Make index construction and evaluation reproducible.

## Current benchmark snapshot

The existing notebook records 300 sampled queries over the 1.28M-vector index.

| Index configuration | Median latency | P99 latency | Serialized index size | Precision@10 |
|---|---:|---:|---:|---:|
| Flat | 481.29 ms | 598.43 ms | 2,502 MiB | 0.613 |
| PQ | 40.79 ms | 50.25 ms | 39.6 MiB | 0.404 |
| IVFFlat, nprobe=16 | 4.34 ms | 5.70 ms | 2,514 MiB | 0.611 |
| IVFPQ, nprobe=16 | 0.92 ms | 1.26 ms | 51.4 MiB | 0.509 |

Precision@10 is based on matching ImageNet class labels. Approximate-index recall should be interpreted separately from class-label precision. Serialized index size is not total process RAM.

## Planned architecture

```text
ImageNet data
    -> batch preprocessing
    -> CLIP embedding extraction
    -> chunked embedding/label artifacts
    -> FAISS index builder
    -> index registry
    -> retrieval API
    -> evaluation and benchmark reports
```

## Repository layout

```text
configs/              Index and experiment configurations
docs/                 Architecture and benchmark notes
src/embeddings/       Embedding extraction interfaces
src/indexing/         FAISS index construction and persistence
src/retrieval/        Top-k search and result mapping
src/evaluation/       Precision, recall, latency, and memory metrics
src/api/              Planned FastAPI service
scripts/              Dataset and benchmark entry points
tests/                Unit and integration tests
benchmarks/           Raw results and generated summaries
```

## Planned milestones

1. Extract notebook logic into reusable embedding and indexing modules.
2. Add deterministic index configurations and artifact metadata.
3. Add exact-search ground truth and Recall@K evaluation.
4. Build a FastAPI endpoint for image-to-image retrieval.
5. Measure cold/warm latency, concurrency, throughput, and process memory.
6. Document failure cases and retrieval-quality tradeoffs.

## Reproducibility

Record the dataset revision, CLIP checkpoint, preprocessing, embedding dimension, normalization, FAISS version, index parameters, CPU/thread configuration, query IDs, and random seed for each run.

## Data and model

- Dataset: [ImageNet-1K 256x256](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256)
- Embedding model: [OpenAI CLIP ViT-B/32](https://github.com/openai/CLIP)
- ANN library: [FAISS](https://github.com/facebookresearch/faiss)

## License and data

Check the dataset and model licenses and usage terms before redistributing data or derived artifacts. This repository should contain code and metadata, not the ImageNet images or large embedding/index files.
