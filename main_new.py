import os
import asyncio
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from tripo3d import TripoClient
from tripo3d.enums import TaskStatus

load_dotenv()

app = FastAPI(title="2D to 3D Converter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRIPO_API_KEY = os.getenv("TRIPO_API_KEY")


@app.get("/")
async def root():
    return {"message": "2D to 3D Converter API", "status": "running"}


@app.post("/upload-and-convert")
async def upload_and_convert(files: List[UploadFile] = File(...)):
    """Upload images and convert to 3D model using Tripo3D SDK."""
    
    if not TRIPO_API_KEY:
        raise HTTPException(status_code=500, detail="TRIPO_API_KEY not configured")
    
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one image is required")
    
    # Save uploaded files to temp directory
    temp_files = []
    try:
        for file in files:
            contents = await file.read()
            # Create temp file with proper extension
            suffix = os.path.splitext(file.filename)[1] or ".jpg"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(contents)
            temp_file.close()
            temp_files.append(temp_file.name)
        
        async with TripoClient(api_key=TRIPO_API_KEY) as client:
            # Create task based on number of images
            if len(temp_files) == 1:
                # Single image: use image_to_model
                task_id = await client.image_to_model(image=temp_files[0])
            else:
                # Multiple images: use multiview_to_model
                task_id = await client.multiview_to_model(images=temp_files)
            
            print(f"Task created: {task_id}")
            
            # Wait for task completion
            task = await client.wait_for_task(task_id, verbose=True)
            
            if task.status == TaskStatus.SUCCESS:
                model_url = task.output.model if task.output else None
                return JSONResponse({
                    "status": "success",
                    "task_id": task_id,
                    "model_url": model_url,
                    "image_count": len(temp_files),
                })
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Task failed with status: {task.status}"
                )
    
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a conversion task"""
    
    if not TRIPO_API_KEY:
        raise HTTPException(status_code=500, detail="TRIPO_API_KEY not configured")
    
    async with TripoClient(api_key=TRIPO_API_KEY) as client:
        task = await client.get_task(task_id)
        return {
            "task_id": task_id,
            "status": task.status,
            "output": task.output.__dict__ if task.output else None
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
