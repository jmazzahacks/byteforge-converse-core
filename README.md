# byteforge-converse-core

Business logic and service layer for ByteforgeConverse — LLM orchestration, conversation state, persistence.

Depends on [`byteforge-converse-models`](https://github.com/jmazzahacks/byteforge-converse-models). Consumed by `byteforge-converse-backend`.

## Installation

This is a public repo — no token required.

```bash
pip install git+https://github.com/jmazzahacks/byteforge-converse-core.git
```

### As a dependency in pyproject.toml
```toml
dependencies = [
    "byteforge-converse-core @ git+https://github.com/jmazzahacks/byteforge-converse-core.git",
]
```

### As a dependency in requirements.txt
```
byteforge-converse-core @ git+https://github.com/jmazzahacks/byteforge-converse-core.git
```

## Usage

```python
import byteforge_converse_core

# Add usage examples here
```

## Development

### Setup

```bash
# Create virtual environment
python -m venv .

# Activate virtual environment
source bin/activate  # On Windows: bin\Scripts\activate

# Install dependencies
pip install -r dev-requirements.txt
pip install -e .
```

The `byteforge-converse-models` dependency is pulled from its public GitHub repo — no token needed.

## License

O'Saasy License — see [LICENSE](LICENSE). Reserves commercial SaaS rights for the copyright holder. See https://osaasy.dev/ for details.

## Author

Jason Byteforge ([@jmazzahacks](https://github.com/jmazzahacks)) — jason@reallybadapps.com
