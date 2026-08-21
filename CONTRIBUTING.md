# Contributing

Thanks for your interest in dsh-vision! Please follow the conventions below to keep collaboration smooth.

## Commit Message Convention (Conventional Commits)

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[body: explain the motivation and how it was verified]
```

Common `type` values:

| type | Meaning |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Refactor (no behavior change) |
| `test` | Tests |
| `docs` | Documentation |
| `ci` / `build` | CI / build config |
| `chore` | Misc (deps, formatting, etc.) |

`scope` is optional; use it when the change targets a specific module, e.g. `feat(video): support scene-change frame sampling`.

Examples:

```
feat(api): add /api/face/verify endpoint

- cosine similarity over registry embeddings
- add tests/test_face_verify.py (5 cases)
```

## Version Tags (SemVer)

Releases use semantic versioning in the form `v<major>.<minor>.<patch>`, e.g.:

```bash
git tag v0.1.0
git push origin v0.1.0
```

- Pushing a `v*` tag triggers GitHub Actions to build and push the Docker image (ghcr.io).
- Breaking API changes → `major`; backward-compatible features → `minor`; bug fixes → `patch`.
- During 0.x: treat `minor` as potentially breaking (0.1 → 0.2 may be incompatible).

## Release Checklist

- [ ] `ruff check app scripts tests examples` passes
- [ ] `ruff format app scripts tests examples --check` passes
- [ ] `python -m pytest tests` is green
- [ ] `scripts/download_models.py` can download each model (at least object detection)
- [ ] Docker image builds and `/health` returns 200
- [ ] The Model License List in README matches `models.json`
- [ ] Version bumped (`app/__init__.py` matches the tag)
- [ ] Tag: `git tag v0.1.0 && git push origin v0.1.0`

## Repository Description (GitHub)

One-liner for the repository settings:

> Open-source vision recognition service for DeepSeek function calling: object detection, captioning, scene, OCR and face recognition for images and videos — FastAPI + Python SDK + one-command Docker deployment.
