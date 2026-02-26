# CLAUDE.md — AumOS Video Engine

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-video-engine`) is part of **Tier B: Open Core**:
Phase 1A Data Factory synthesis engines.

**Release Tier:** B: Open Core
**Product Mapping:** Product 1 — Data Factory
**Phase:** 1A (Months 3-8)

## Repo Purpose

`aumos-video-engine` generates synthetic video sequences for manufacturing QA, surveillance
training, and autonomous vehicle datasets. It uses Stable Video Diffusion for frame generation
and BlenderProc for 3D scene composition, with temporal coherence enforcement and per-frame
privacy protection (face blur, license plate redaction) via the privacy-engine service.

## Architecture Position

```
aumos-platform-core → aumos-auth-gateway → aumos-privacy-engine ─┐
                                         → aumos-image-engine ────┤
                                                └── aumos-video-engine ◄─┘
                                                       ↘ aumos-event-bus
                                                       ↘ aumos-data-layer
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events
- `aumos-privacy-engine` — Per-frame face blur, plate redaction (HTTP client)
- `aumos-image-engine` — Reference frame generation for SVD conditioning (HTTP client)

**Downstream dependents (other repos IMPORT from this):**
- `aumos-fidelity-validator` — validates synthetic video statistical fidelity
- `aumos-data-pipeline` — orchestrates video generation within data pipelines

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| confluent-kafka | 2.3+ | Kafka producer/consumer |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |
| diffusers | 0.27+ | Stable Video Diffusion pipeline |
| torch | 2.2+ | Model inference (GPU optional) |
| opencv-python-headless | 4.9+ | Frame processing, face detection |
| av (PyAV) | 12.0+ | Video encoding/decoding (libav wrapper) |
| transformers | 4.38+ | HuggingFace model loading |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.** If aumos-common provides it, use it.
   ```python
   # CORRECT
   from aumos_common.auth import get_current_tenant, get_current_user
   from aumos_common.database import get_db_session, Base, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ErrorCode
   from aumos_common.config import AumOSSettings
   from aumos_common.health import create_health_router
   from aumos_common.pagination import PageRequest, PageResponse, paginate
   from aumos_common.app import create_app
   ```

2. **Type hints on EVERY function.** No exceptions.

3. **Pydantic models for ALL API inputs/outputs.** Never return raw dicts.

4. **RLS tenant isolation via aumos-common.** Never write raw SQL that bypasses RLS.

5. **Structured logging via structlog.** Never use print() or logging.getLogger().

6. **Publish domain events to Kafka after state changes.**

7. **Async by default.** All I/O operations must be async.

8. **Google-style docstrings** on all public classes and functions.

### Style Rules

- Max line length: **120 characters**
- Import order: stdlib → third-party → aumos-common → local
- Linter: `ruff` (select E, W, F, I, N, UP, ANN, B, A, COM, C4, PT, RUF)
- Type checker: `mypy` strict mode
- Formatter: `ruff format`

## File Structure Convention

```
src/aumos_video_engine/
├── __init__.py
├── main.py                    # FastAPI app entry point
├── settings.py                # Extends AumOSSettings
├── api/
│   ├── __init__.py
│   ├── router.py              # All endpoints
│   ├── schemas.py             # Request/response Pydantic models
│   └── dependencies.py        # FastAPI dependency injection
├── core/
│   ├── __init__.py
│   ├── models.py              # VideoGenerationJob, SceneTemplate ORM models
│   ├── interfaces.py          # Protocol definitions
│   └── services.py            # Business logic services
├── adapters/
│   ├── __init__.py
│   ├── repositories.py        # SQLAlchemy repositories
│   ├── kafka.py               # Kafka event publishing
│   ├── storage.py             # MinIO video artifact storage
│   ├── privacy_client.py      # Privacy Engine HTTP client
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── stable_video_diffusion.py  # SVD adapter
│   │   └── blenderproc_scene.py       # BlenderProc adapter
│   ├── temporal_engine.py     # Frame coherence enforcement
│   ├── privacy_enforcer.py    # Per-frame privacy (local fallback)
│   └── domain_specific.py     # Domain scenario generators
└── migrations/
    ├── env.py
    ├── alembic.ini
    └── versions/
tests/
├── __init__.py
├── conftest.py
├── test_api.py
├── test_services.py
└── test_repositories.py
```

## Database Conventions

- **Table prefix: `vid_`** (e.g., `vid_generation_jobs`, `vid_scene_templates`)
- ALL tenant-scoped tables extend `AumOSModel` (id, tenant_id, created_at, updated_at)
- RLS policy on every tenant table
- Migration naming: `{timestamp}_vid_{description}.py`

## Domain-Specific Notes

### Temporal Coherence
- Score range: 0.0 to 1.0 (1.0 = perfectly coherent)
- Minimum acceptable score: configurable via `TEMPORAL_COHERENCE_MIN_SCORE` (default 0.7)
- Uses optical flow (OpenCV) + feature matching to score frame transitions
- `TemporalCoherenceService` rejects or re-generates frames below threshold

### Privacy Enforcement
- Primary path: delegate to `aumos-privacy-engine` HTTP client (`PrivacyEnforcerProtocol`)
- Fallback path: local OpenCV face blur if privacy-engine is unavailable
- All faces MUST be blurred in surveillance/traffic domain videos
- License plates MUST be redacted in all domains
- `privacy_enforced` flag on VideoGenerationJob tracks whether enforcement was applied

### SVD (Stable Video Diffusion)
- Model: `stabilityai/stable-video-diffusion-img2vid-xt` (25 frames default)
- Conditioning on a reference image from aumos-image-engine
- GPU optional — falls back to CPU (slower but functional)
- Model weights cached at `SVD_CACHE_DIR` (default `/tmp/model-cache`)

### BlenderProc Scene Composition
- Used for structured 3D scene generation (manufacturing floor, intersections)
- Scenes defined as `SceneTemplate` records with `scene_config` JSONB
- Objects field contains 3D object library references

### Domain Scenarios
| Domain | Key Use Cases |
|--------|--------------|
| manufacturing | Assembly line QA, defect detection, robot arm training |
| surveillance | Crowd monitoring, intrusion detection, anomaly detection |
| traffic | AV intersection datasets, rare scenario generation |
| custom | Tenant-defined scenario configuration |

## API Conventions

- All endpoints under `/api/v1/` prefix
- Auth: Bearer JWT token (validated by aumos-common)
- Tenant: `X-Tenant-ID` header (set by auth middleware)
- Generation jobs are async — poll GET /video/jobs/{id} for status
- Batch jobs return a batch_id for polling

## What Claude Code Should NOT Do

1. **Do NOT reimplement anything in aumos-common.**
2. **Do NOT use print().** Use `get_logger(__name__)`.
3. **Do NOT return raw dicts from API endpoints.**
4. **Do NOT write raw SQL.** Use SQLAlchemy ORM with BaseRepository.
5. **Do NOT hardcode configuration.** Use Settings with env vars.
6. **Do NOT skip type hints.**
7. **Do NOT skip privacy enforcement.** Every video output MUST have `privacy_enforced=True` unless explicitly opted out by a SUPER_ADMIN tenant.
8. **Do NOT put business logic in API routes.**
9. **Do NOT create new exception classes** unless they map to a new ErrorCode in aumos-common.
10. **Do NOT bypass RLS.**
11. **Do NOT load model weights from user-supplied URLs** — only from trusted registries.
