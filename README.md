# Manifold Index Calculator

> **See [SPEC.md](SPEC.md) for the full mathematical specification.**

A Python application that, given a 3-manifold name, computes:

1. Triangulation data via SnaPy (cusps, tetrahedra, gluing equations)
2. The 3D index `I(m⃗, e⃗)`
3. Non-closable cycles via Dehn filling
4. A phase space basis via easy edges
5. The refined index

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
manifold-index <manifold_name>
```

## Project Structure

```
src/manifold_index/
├── app/          # Entry point, CLI/UI layer
├── core/         # Mathematical pipeline modules
└── utils/        # Shared utilities
```

## Development

```bash
pytest tests/
```
