try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required.") from e

try:
    from ..models import DataCleanAction, DataCleanObservation
    from .datacleanenv_environment import DataCleanEnvironment
except (ModuleNotFoundError, ImportError):
    from models import DataCleanAction, DataCleanObservation
    from server.datacleanenv_environment import DataCleanEnvironment

from fastapi.responses import JSONResponse

app = create_app(
    DataCleanEnvironment,
    DataCleanAction,
    DataCleanObservation,
    env_name="datacleanenv",
    max_concurrent_envs=4,
)

@app.get("/")
def root():
    return JSONResponse({"name": "DataCleanEnv", "status": "running", "tasks": ["easy", "medium", "hard"]})

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
