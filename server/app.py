"""
FastAPI application for the DataCleanEnv Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from ..models import DataCleanAction, DataCleanObservation
    from .datacleanenv_environment import DataCleanEnvironment
except (ModuleNotFoundError, ImportError):
    from models import DataCleanAction, DataCleanObservation
    from server.datacleanenv_environment import DataCleanEnvironment


app = create_app(
    DataCleanEnvironment,
    DataCleanAction,
    DataCleanObservation,
    env_name="datacleanenv",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)


# Add root endpoint for health check
from fastapi.responses import JSONResponse

@app.get("/")
def root():
    return JSONResponse({"name": "DataCleanEnv", "status": "running", "tasks": ["easy", "medium", "hard"]})
