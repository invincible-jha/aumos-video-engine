# Changelog — aumos-video-engine

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-02-26

### Added
- Initial scaffolding for aumos-video-engine
- Stable Video Diffusion (SVD) frame generation adapter
- BlenderProc 3D scene composition adapter
- Temporal coherence enforcement engine
- Per-frame privacy enforcement (face blur, license plate redaction, PII removal)
- Domain-specific scenario generators: manufacturing QA, surveillance, traffic/AV
- VideoGenerationJob and SceneTemplate SQLAlchemy models with tenant isolation
- REST API: POST /video/generate, /video/compose-scene, /video/enforce-privacy
- REST API: GET /video/jobs/{id}, POST /video/batch, GET /video/templates
- GenerationService, SceneCompositionService, TemporalCoherenceService
- PrivacyEnforcementService, BatchService
- Kafka event publishing for generation lifecycle events
- MinIO/S3 storage adapter for video artifacts
- Privacy Engine HTTP client adapter
- Image Engine HTTP client adapter
- Full hexagonal architecture (api/ + core/ + adapters/)
- Comprehensive test suite (unit + integration)
- Docker + docker-compose.dev.yml for local development
- GitHub Actions CI/CD pipeline
