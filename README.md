## Running the Project with Docker

This project provides a Docker setup for running the main FastAPI application. The Docker configuration is tailored to the project's requirements and dependencies.

### Project-Specific Docker Details

- **Base Image:** `python:3.10-slim` (Python 3.10)
- **System Dependencies:** Installs build tools, `ffmpeg`, and libraries required for OpenCV and PyTorch.
- **Python Dependencies:** Installed from `requirements.txt` inside a virtual environment (`.venv`).
- **User:** Runs as a non-root user (`appuser`) for security.
- **Exposed Port:** `8000` (FastAPI default)
- **Entrypoint:** Runs the FastAPI app via Uvicorn: `main:app` on `0.0.0.0:8000`.

### Environment Variables

- No required environment variables are specified in the Dockerfile or compose file. If you need to add any, uncomment the `env_file` line in `docker-compose.yml` and provide a `.env` file.

### Build and Run Instructions

1. **Build and Start the Service:**

   ```sh
   docker compose up --build
   ```

   This will build the Docker image and start the FastAPI application, making it available at [http://localhost:8000](http://localhost:8000).

2. **Stopping the Service:**

   ```sh
   docker compose down
   ```

### Special Configuration Notes

- **No external services** (databases, caches, etc.) are required or configured.
- **No persistent volumes** are needed for this setup.
- If you need to add environment variables, create a `.env` file and uncomment the `env_file` line in `docker-compose.yml`.

### Ports

- **8000:** Exposed by the FastAPI application and mapped to the host.

---

_This section was updated to reflect the current Docker setup for the project. Please ensure your `requirements.txt` is up to date with all necessary dependencies for your application._
