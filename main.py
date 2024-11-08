from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import shutil
from pathlib import Path
import subprocess  # Optional for FFmpeg conversion

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

# HTML form for uploading files, selecting model, and displaying results
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "file_url": None})

# Endpoint to handle file uploads, model selection, and perform detection
@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...), model_type: str = Form(...)):
    # Define file paths
    file_path = UPLOAD_FOLDER / file.filename
    output_path = UPLOAD_FOLDER / f"detected_{file.filename}"
    
    # Save the uploaded file
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Select the model and settings based on user choice
    model = models.get(model_type)
    if not model:
        return {"error": "Invalid model selection"}
    
    # Define class filter and confidence based on model type
    classes = [16] if model_type == "dog" else [7]  # Use class 16 for dogs, class 7 for potholes
    conf = 0.4 if model_type == "dog" else 0.1       # Default 0.4 for dogs, 0.1 for potholes
    
    # Perform detection based on file type and model
    if file.content_type.startswith("image"):
        # Image processing
        results = model.predict(source=str(file_path), classes=classes, conf=conf)
        annotated_frame = results[0].plot()  # Annotate the detection on the image
        cv2.imwrite(str(output_path), annotated_frame)

    elif file.content_type.startswith("video"):
        # Video processing
        input_video = cv2.VideoCapture(str(file_path))
        frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(input_video.get(cv2.CAP_PROP_FPS))

        # Set up VideoWriter for output video
        video_writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Perform detection on each frame with specified classes and confidence
        results = model.predict(source=str(file_path), stream=True, classes=classes, conf=conf)
        for r in results:
            annotated_frame = r.plot()  # Annotate the detection on the frame
            video_writer.write(annotated_frame)  # Write to output video

        # Release video resources
        input_video.release()
        video_writer.release()

        # Optional: Ensure compatibility using FFmpeg re-encoding
        compatible_output_path = UPLOAD_FOLDER / f"compatible_{file.filename}"
        subprocess.run([
            "ffmpeg", "-i", str(output_path), "-vcodec", "libx264", "-acodec", "aac", str(compatible_output_path)
        ])
        output_path = compatible_output_path  # Update to compatible output path
    
    else:
        return {"error": "Unsupported file type"}

    # Pass file URL to render the updated HTML with the result
    file_url = f"/static/{output_path.name}"
    return templates.TemplateResponse("index.html", {"request": request, "file_url": file_url})
