# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and this project does not yet publish versioned releases.

Entries are built from the repository's own `feat:` and `fix:` commit subjects; earlier commits that do not use those prefixes are not listed.

## [Unreleased]

### Added
- Agentic escalation tier with Pydantic output contracts and a production spec (`bca4f0c`).
- Tier-1 classifier confidence scoring (`a224197`).
- Pure Tier-1/Tier-2 escalation gate implementing SPEC section 5.1 (`b0da305`).
- Groq as the primary Tier-2 LLM provider (`26bca59`).
- `/triage_agent` rewired into a real two-tier gated pipeline (`1988409`).
- Escalation Triage tab in the live demo UI (`4fbc4cf`).
- Measured Tier-1 no-escalation rate in `reports/ESCALATION_GATE_BENCHMARK.md`, replacing an assumed 90% (`901177b`).

### Changed
- Escalation thresholds recalibrated against the real data distribution: confidence 0.67 and 185.0 predicted hours (`af6d5e0`).
- Documentation refresh: standard README, docs index, decisions log, changelog, MIT license and `.env.example` comments; corrected `docs/SPEC.md`, `docs/DEPLOY.md`, `docs/DEMO.md` and `data/DATA_SETUP.md` to match the code; moved the superseded `docs/PROJECT_SPEC.md` to an untracked `docs/archive/`; stopped tracking `docs/superpowers/`; extended `.gitignore`.

### Fixed
- Production hardening: repaired `docker-compose.yml`, sanitized error responses and added typed settings (`d9bc959`).
- Restored valid `PATH` and `PORT` variables in the Dockerfile (`cf41dcb`).
- Final review findings: held-out benchmark sample, spec drift, field semantics and logging (`153e549`).
- Escalation Triage tab now sends the `previous_tickets` and `channel` fields (`bd842bf`).
