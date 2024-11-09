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

app = FastAPI()

# Load YOLO models
models = {
    "dog": YOLO("models/yolov8n.pt"),  # Path to the YOLO model trained on COCO for dogs
    "pothole": YOLO("models/y8best.pt")  # Path to the pothole detection model
}

# Directory to save uploaded and processed files
UPLOAD_FOLDER = Path("static")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Store the statuses of the tasks
task_statuses = {}

# HTML form for uploading files, selecting model, and displaying results
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "file_url": None})

# Background task for processing the image or video
def process_file(file_path: Path, output_path: Path, model, classes, conf, task_id):
    # Perform detection based on file type and model
    if file_path.suffix in [".jpg", ".jpeg", ".png"]:  # Image processing
        results = model.predict(source=str(file_path), classes=classes, conf=conf)
        annotated_frame = results[0].plot()  # Annotate the detection on the image
        cv2.imwrite(str(output_path), annotated_frame)

    elif file_path.suffix in [".mp4", ".avi", ".mov"]:  # Video processing
        input_video = cv2.VideoCapture(str(file_path))
        frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(input_video.get(cv2.CAP_PROP_FPS))

        # Set up VideoWriter for output video
        temp_output_path = UPLOAD_FOLDER / f"temp_{output_path.name}"
        video_writer = cv2.VideoWriter(str(temp_output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Perform detection on each frame with specified classes and confidence
        results = model.predict(source=str(file_path), stream=True, classes=classes, conf=conf)
        for r in results:
            annotated_frame = r.plot()  # Annotate the detection on the frame
            video_writer.write(annotated_frame)  # Write to output video

        # Release video resources
        input_video.release()
        video_writer.release()

        # Re-encode with FFmpeg for browser compatibility
        subprocess.run([
            "ffmpeg", "-y", "-i", str(temp_output_path), "-vcodec", "libx264", "-acodec", "aac", "-strict", "-2", str(output_path)
        ])
        temp_output_path.unlink()  # Delete the temporary file

    # Update the task status to 'Completed'
    task_statuses[task_id] = {"status": "Completed", "file_url": f"/static/{output_path.name}"}

# Endpoint to handle file uploads, model selection, and perform detection
@app.post("/detect")
async def detect(background_tasks: BackgroundTasks, file: UploadFile = File(...), model_type: str = Form(...)):
    # Define file paths
    file_path = UPLOAD_FOLDER / file.filename
    output_path = UPLOAD_FOLDER / f"detected_{file.filename}"
    
    # Save the uploaded file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Select the model and settings based on user choice
    model = models.get(model_type)
    if not model:
        return JSONResponse({"error": "Invalid model selection"}, status_code=400)
    
    # Define class filter and confidence based on model type
    classes = [16] if model_type == "dog" else [7]  # Use class 16 for dogs, class 7 for potholes
    conf = 0.4 if model_type == "dog" else 0.1       # Default 0.4 for dogs, 0.1 for potholes
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    task_statuses[task_id] = {"status": "Processing", "file_url": None}

    # Run the processing in the background
    background_tasks.add_task(process_file, file_path, output_path, model, classes, conf, task_id)

    # Return the task ID for status tracking
    return JSONResponse({"task_id": task_id})

# Endpoint to check the status of a detection task
@app.get("/status/{task_id}")
async def check_status(task_id: str):
    status = task_statuses.get(task_id)
    if not status:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse(status)
