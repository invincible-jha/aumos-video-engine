# aumos-video-engine

Synthetic video generation engine for manufacturing QA, surveillance training, and autonomous
vehicle datasets. Provides temporally coherent synthetic video with per-frame privacy enforcement,
domain-specific scenario composition, and enterprise-grade multi-tenant isolation.

## Overview

`aumos-video-engine` is **Tier B (Open Core)** and part of **AumOS Data Factory** (Phase 1A).
It generates synthetic video sequences using Stable Video Diffusion and BlenderProc 3D scene
composition, with:

- **Temporal coherence**: Frame-to-frame motion consistency scoring and enforcement
- **Per-frame privacy**: Automatic face blur, license plate redaction, PII removal via privacy-engine
- **Domain scenarios**: Manufacturing assembly lines, security surveillance, traffic/AV intersections
- **Batch generation**: Large-scale parallel video generation for training datasets
- **Multi-tenant isolation**: Full RLS enforcement — tenants cannot access each other's videos

## Architecture

```
aumos-platform-core
    └── aumos-auth-gateway
           └── aumos-privacy-engine ──────┐
           └── aumos-image-engine ─────────┤
                   └── aumos-video-engine ◄─┘
                          ↘ aumos-event-bus (lifecycle events)
                          ↘ aumos-data-layer (job storage)
```

## Quickstart

```bash
# Setup
cp .env.example .env
make dev

# Start dev services
make docker-compose-up

# Run migrations
make migrate

# Start service
make run
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/v1/video/generate | Generate synthetic video from prompt |
| POST | /api/v1/video/compose-scene | Compose 3D scene and render video |
| POST | /api/v1/video/enforce-privacy | Apply per-frame privacy to existing video |
| GET | /api/v1/video/jobs/{id} | Poll generation job status |
| POST | /api/v1/video/batch | Submit batch generation job |
| GET | /api/v1/video/templates | List available scene templates |
| GET | /health | Health check |

## Domain Scenarios

### Manufacturing QA
Generate assembly line footage with configurable defect rates, lighting conditions,
and camera angles for training visual inspection models.

### Surveillance Training
Generate security camera footage with configurable crowd density, lighting, occlusion
patterns, and anomaly scenarios — all faces and plates privacy-enforced.

### Traffic / Autonomous Vehicles
Generate intersection and highway footage with configurable vehicle density, weather
conditions, and rare edge cases (pedestrian jaywalking, debris, etc.).

## Tech Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API |
| Stable Video Diffusion | 1.1+ | Frame generation |
| OpenCV | 4.9+ | Frame processing |
| PyAV (libav) | 12.0+ | Video encoding/decoding |
| PyTorch | 2.2+ | Model inference |
| SQLAlchemy | 2.0+ | Database ORM |
| Kafka | — | Event streaming |
| MinIO | — | Video artifact storage |

## Development

```bash
make lint         # Run ruff linter
make type-check   # Run mypy strict
make test-cov     # Run tests with 80% coverage gate
make all          # Full CI pipeline
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
