from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel # Field is not used from Pydantic in the provided snippet
from typing import List, Optional # Dict is not used from typing in the provided snippet
from contextlib import asynccontextmanager # Not used in the provided snippet
import logging
import shutil
import os

# Setup logging
logger = logging.getLogger("api_logger")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Current Directory Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
# It's generally better not to change the current working directory globally
# Instead, construct absolute paths.
# os.chdir(script_dir) # Avoid if possible, use absolute paths instead
current_working_dir = script_dir # Use script_dir as the base for other paths
print(f"Base Directory: {current_working_dir}")

# --- BEGIN MODIFICATIONS ---
VIDEO_DIR = os.path.join(current_working_dir, "Videos")
PROCESSED_VIDEO_DIR = os.path.join(current_working_dir, "processed_videos") # Example if football_analytics saves output
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(PROCESSED_VIDEO_DIR, exist_ok=True) # Example

components_initialized_successfully = True
try:
    from Football_Analytics import football_analytics
    logger.info("Successfully imported from Football_Analytics.py")
except ImportError as e:
    logger.critical(f"CRITICAL: Could not import from Football_Analytics.py. Error: {e}")
    components_initialized_successfully = False
    def football_analytics(video_path_or_name: str): # Dummy for startup if import fails
        logger.error(f"Football_Analytics not loaded. Called with {video_path_or_name}")
        raise RuntimeError("Football_Analytics components are not available.")
# --- END MODIFICATIONS ---

# --- BEGIN MODIFICATIONS: Pydantic Models and Endpoints ---
class VideoNameRequest(BaseModel):
    video_name: str
    # You could add more parameters here if football_analytics needs them
    # e.g., output_folder: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not components_initialized_successfully:
        logger.error("FastAPI app starting, but Football_Analytics components FAILED to initialize.")
    else:
        logger.info("FastAPI app running, Football_Analytics components initialized successfully.")
        yield

app = FastAPI(lifespan= lifespan)

@app.post("/analyze-video/upload")
async def analyze_video_upload(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...)
):
    """
    Upload a video file and start the football analytics processing.
    The processing runs in the background. Video display windows will appear on the server.
    """
    if not components_initialized_successfully:
        raise HTTPException(status_code=503, detail="Football_Analytics components FAILED to initialize.")

    file_path = os.path.join(VIDEO_DIR, video_file.filename)

    # Save the uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        logger.info(f"Uploaded video '{video_file.filename}' saved to '{file_path}'.")
    except Exception as e:
        logger.error(f"Error saving uploaded file '{video_file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Could not save video file: {e}")
    finally:
        video_file.file.close()

    # Add the processing to background tasks
    #background_tasks.add_task(football_analytics, video_file.filename)

    return {
        "message": f"Video '{video_file.filename}' uploaded successfully. Analytics processing started in the background. Display windows will appear on the server if `football_analytics` is configured to show them.",
        "uploaded_video_path": file_path
    }

@app.post("/analyze-video/by-name")
async def analyze_video_by_name(
    request: VideoNameRequest,
    background_tasks: BackgroundTasks
):
    """
    Start football analytics processing for a video already existing on the server.
    The video should be located in the server's pre-configured VIDEO_DIR.
    Processing runs in the background. Video display windows will appear on the server.
    """
    if not components_initialized_successfully:
        raise HTTPException(status_code=503, detail="Football_Analytics components FAILED to initialize.")

    video_path = os.path.join(VIDEO_DIR, request.video_name)

    if not os.path.exists(video_path) or not os.path.isfile(video_path):
        logger.warning(f"Video '{request.video_name}' not found at path '{video_path}'.")
        raise HTTPException(status_code=404, detail=f"Video '{request.video_name}' not found in the server's video directory.")

    # Add the processing to background tasks
    background_tasks.add_task(football_analytics, video_path)

    return {
        "message": f"Analytics processing for video '{request.video_name}' started in the background. Display windows will appear on the server if `football_analytics` is configured to show them.",
        "video_path": video_path
    }

@app.get("/videos")
async def list_available_videos():
    """
    Lists video files available in the server's pre-configured VIDEO_DIR.
    """
    if not os.path.exists(VIDEO_DIR):
        logger.warning(f"Video directory '{VIDEO_DIR}' not found when trying to list videos.")
        return {"message": "Video directory configured on server not found.", "videos": []}
    try:
        video_files = [
            f for f in os.listdir(VIDEO_DIR)
            if os.path.isfile(os.path.join(VIDEO_DIR, f))
            and f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) # Add more extensions if needed
        ]
        return {"videos": video_files, "video_directory": VIDEO_DIR}
    except Exception as e:
        logger.error(f"Error listing videos from '{VIDEO_DIR}': {e}")
        raise HTTPException(status_code=500, detail=f"Could not list videos: {e}")

@app.get("/health")
async def health():
    if components_initialized_successfully:
        return {"status": "ok", "message": "FastAPI app running, Football_Analytics reported component init success."}
    else:
        # Use 503 Service Unavailable for health check failure
        raise HTTPException(status_code=503, detail="FastAPI app running, BUT Football_Analytics components FAILED to initialize.")

if __name__ == "__main__":
    import uvicorn
    # Use the app instance name defined (app)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) # Added "main:app" for uvicorn and reload=True for dev