# langchain-learning

## Setup

1. Install [uv](https://docs.astral.sh/uv/)
2. Download dependencies: `uv sync`
3. If using HuggingFace, log in to huggingface with `huggingface-cli login`
4. Otherwise, run `export $(grep -v '^#' .env | xargs)`
5. `uv run <script>.py`