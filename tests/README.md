# Aether Testing

This directory contains tests for the Aether project.

## Directory Structure

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for end-to-end scenarios

## Running Tests

To run all tests:

```bash
pytest
```

To run only unit tests:

```bash
pytest tests/unit
```

To run only integration tests:

```bash
pytest tests/integration
```

To run tests with coverage:

```bash
pytest --cov=aether
```

## Test Categories

### Unit Tests

- `test_io.py` - Tests for mesh loading and I/O operations
- `test_extract.py` - Tests for scatterer detection
- `test_preprocess.py` - Tests for mesh preprocessing
- `test_ranking.py` - Tests for scatterer ranking functions
- `test_config.py` - Tests for configuration handling

### Integration Tests

- `test_plate.py` - End-to-end test with a simple plate
- `test_cli.py` - Tests for the command-line interface
