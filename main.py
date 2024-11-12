from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import shutil
from pathlib import Path
import uuid
import subprocess
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO models
models = {
    "dog": YOLO("models/yolov8n.pt"),
    "pothole": YOLO("models/y8best.pt")
}

# Directory to save uploaded and processed files
UPLOAD_FOLDER = Path("static")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Store task statuses
task_statuses = {}

# HTML form for uploading files, selecting model, and displaying results
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "file_url": None})

# Background task for processing the image or video
def process_file(file_path: Path, output_path: Path, model, classes, conf, task_id, min_skip=2, max_skip=5):
    start_time = time.time()  # Start tracking time
    try:
        if file_path.suffix in [".jpg", ".jpeg", ".png"]:  # Image processing
            results = model.predict(source=str(file_path), classes=classes, conf=conf, stream=True)
            # Since stream=True returns a generator, we need to iterate to get results
            for result in results:
                annotated_frame = result.plot()  # Annotate the detection on the image
            cv2.imwrite(str(output_path), annotated_frame)
            end_time = time.time()  # Stop tracking time
            task_statuses[task_id] = {
                "status": "Completed",
                "file_url": f"/static/{output_path.name}",
                "processing_time": round(end_time - start_time, 2)  # Time in seconds
            }

        elif file_path.suffix in [".mp4", ".avi", ".mov"]:  # Video processing
            input_video = cv2.VideoCapture(str(file_path))
            if not input_video.isOpened():
                raise Exception("Error opening video file")

            frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(input_video.get(cv2.CAP_PROP_FPS))
            frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = frame_count / fps

            # Dynamically calculate skip_frames based on video duration
            skip_frames = max(min_skip, min(int(video_duration / 10), max_skip))

            # Temporary output path for intermediate video processing
            temp_output_path = UPLOAD_FOLDER / f"temp_{output_path.name}"
            video_writer = cv2.VideoWriter(
                str(temp_output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height)
            )

            if not video_writer.isOpened():
                raise Exception("Error initializing VideoWriter")

            # Perform detection on every nth frame to improve processing speed
            frame_index = 0
            while input_video.isOpened():
                ret, frame = input_video.read()
                if not ret:
                    break

                # Process every (skip_frames + 1)-th frame
                if frame_index % (skip_frames + 1) == 0:
                    # Using stream=True to process frames in real-time
                    results = model.predict(source=frame, classes=classes, conf=conf, stream=True)
                    for result in results:
                        annotated_frame = result.plot() if result else frame
                    video_writer.write(annotated_frame)
                else:
                    video_writer.write(frame)  # Write unprocessed frame to maintain video length

                frame_index += 1

            input_video.release()
            video_writer.release()

            # Re-encode with FFmpeg for browser compatibility
            subprocess.run([
                "ffmpeg", "-y", "-i", str(temp_output_path), "-vcodec", "libx264", "-acodec", "aac", "-strict", "-2", str(output_path)
            ], check=True)
            
            if temp_output_path.exists():
                temp_output_path.unlink()

            end_time = time.time()  # Stop tracking time
            task_statuses[task_id] = {
                "status": "Completed",
                "file_url": f"/static/{output_path.name}",
                "processing_time": round(end_time - start_time, 2)  # Time in seconds
            }

    except Exception as e:
        end_time = time.time()  # Stop tracking time on error
        print(f"Error processing file: {e}")
        task_statuses[task_id] = {
            "status": "Failed",
            "error": str(e),
            "processing_time": round(end_time - start_time, 2)  # Time in seconds
        }

# Endpoint to handle file uploads, model selection, and perform detection
@app.post("/detect")
async def detect(background_tasks: BackgroundTasks, file: UploadFile = File(...), model_type: str = Form(...)):
    file_path = UPLOAD_FOLDER / file.filename
    output_path = UPLOAD_FOLDER / f"detected_{file.filename}"
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    model = models.get(model_type)
    if not model:
        return JSONResponse({"error": "Invalid model selection"}, status_code=400)
    
    # Determine class and confidence threshold based on model type
    classes = [16] if model_type == "dog" else [7]
    conf = 0.4 if model_type == "dog" else 0.1
    task_id = str(uuid.uuid4())
    task_statuses[task_id] = {"status": "Processing", "file_url": None, "processing_time": None}

    # Add the background task with dynamic frame skipping and timing
    background_tasks.add_task(process_file, file_path, output_path, model, classes, conf, task_id)
    
    return JSONResponse({"task_id": task_id})

# Endpoint to check the status of a detection task
@app.get("/status/{task_id}")
async def check_status(task_id: str):
    status = task_statuses.get(task_id)
    if not status:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse(status)
