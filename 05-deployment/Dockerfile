# FROM python:3.13.9-slim-bookworm
FROM python:3.13.9-slim-bookworm

# Use non-interactive mode and install uv (the package manager)
# RUN pip install --no-cache-dir uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PATH=/app/.venv/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY .python-version pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Copy the application files
COPY predict.py model.bin ./

# Expose the app port
EXPOSE 9696

# Start the FastAPI app using uvicorn
# ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]