# Security Policy — aumos-video-engine

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities privately to: security@aumos.ai

Include:
- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Affected versions
- Suggested fix (if available)

We will acknowledge receipt within 48 hours and provide a timeline for resolution.

## Responsible Disclosure

- We will work with you to understand and fix the issue
- We will credit reporters who follow responsible disclosure
- Public disclosure is coordinated after a patch is released

## Security Considerations for This Service

### Privacy Enforcement
This service processes video frames to detect and redact PII (faces, license plates).
Failures in privacy enforcement are treated as critical security issues.

### Tenant Isolation
All video generation jobs are tenant-scoped. Row-level security (RLS) is enforced
at the database layer. Cross-tenant data access without explicit authorization is
a critical vulnerability.

### Model Artifacts
Model weights are loaded from trusted registries. Never load model weights from
untrusted sources or user-supplied URLs.

### GPU Resources
GPU resources are shared across tenants. Resource exhaustion attacks (generating
very long videos in parallel) should be reported as DoS vulnerabilities.
