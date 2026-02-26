# Contributing to aumos-video-engine

Thank you for contributing to the AumOS Video Engine. This document outlines the
development workflow and contribution standards.

## Development Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd aumos-video-engine

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
make dev

# Copy environment config
cp .env.example .env
# Edit .env with your local values

# Start development services
make docker-compose-up
```

## Development Workflow

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes following the coding standards in CLAUDE.md
3. Run the full CI pipeline locally: `make all`
4. Write or update tests for changed code
5. Commit with a conventional commit message
6. Open a pull request against `main`

## Coding Standards

See [CLAUDE.md](CLAUDE.md) for the complete coding standards. Key requirements:

- Type hints on every function signature
- Pydantic models for all API inputs/outputs
- Structured logging via `get_logger(__name__)` — no `print()`
- RLS tenant isolation must not be bypassed
- Async by default for all I/O

## Commit Messages

Follow Conventional Commits:

```
feat: add temporal coherence scoring
fix: correct per-frame privacy enforcement for edge cases
refactor: extract scene composition into separate service
test: add unit tests for stable video diffusion adapter
docs: document domain-specific scenario configurations
```

## Testing

```bash
# Unit tests
make test-unit

# All tests with coverage
make test-cov

# Integration tests (requires Docker)
make test-integration
```

Tests must maintain 80% coverage on core modules.

## Pull Request Process

1. Ensure `make all` passes (lint + type-check + test-cov)
2. Update CHANGELOG.md under `[Unreleased]`
3. Request a review from a team member
4. Squash-merge to keep history clean

## Security

If you find a security vulnerability, please follow the disclosure process in
[SECURITY.md](SECURITY.md). Do not open a public issue.
