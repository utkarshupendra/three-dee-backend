import os
import asyncio
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, APIRouter, BackgroundTasks
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from tripo3d import TripoClient, TaskStatus
import database as db

# In-memory store for pending tasks (maps our job_id to tripo task_id and metadata)
pending_jobs: Dict[str, Dict[str, Any]] = {}

load_dotenv()

app = FastAPI(title="2D to 3D Converter")
api_router = APIRouter(prefix="/api")

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


@api_router.post("/upload-and-convert")
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
            # Get extension and ensure it's a supported format
            suffix = os.path.splitext(file.filename)[1].lower() or ".jpg"
            # Tripo supports: jpg, jpeg, png, webp
            if suffix not in ['.jpg', '.jpeg', '.png', '.webp']:
                suffix = '.jpg'  # Default to jpg for unsupported formats
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(contents)
            temp_file.close()
            temp_files.append(temp_file.name)
        
        async with TripoClient(api_key=TRIPO_API_KEY) as client:
            if len(temp_files) == 1:
                # Single image mode - minimal parameters
                task_id = await client.image_to_model(image=temp_files[0])
            else:
                # Multiview mode - images should be in order: front, back, left, right
                multiview_images = temp_files[:4]  # Max 4 images
                task_id = await client.multiview_to_model(images=multiview_images)
            
            print(f"Task created: {task_id}")
            
            # Wait for task completion
            task = await client.wait_for_task(task_id, verbose=True)
            
            if task.status == TaskStatus.SUCCESS:
                # Get model URL - prefer pbr_model, fallback to model or base_model
                model_url = None
                if task.output:
                    model_url = task.output.pbr_model or task.output.model or task.output.base_model
                print(f"Model URL: {model_url}")
                return JSONResponse({
                    "status": "success",
                    "task_id": task_id,
                    "model_url": model_url,
                    "image_count": len(temp_files),
                })
            else:
                # Get failure details
                print(f"Task failed: {task}")
                print(f"Task dict: {task.__dict__ if task else None}")
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


@api_router.get("/proxy-glb")
async def proxy_glb(url: str):
    """Proxy GLB file to avoid CORS issues"""
    import httpx
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        from fastapi.responses import Response
        return Response(
            content=response.content,
            media_type="model/gltf-binary",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Content-Disposition": "attachment; filename=model.glb"
            }
        )


@api_router.get("/task/{task_id}")
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


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


async def process_conversion_job(job_id: str, images: List[str], name: Optional[str], view_config: dict):
    """Background task to process the conversion."""
    try:
        pending_jobs[job_id]["status"] = "processing"
        pending_jobs[job_id]["progress"] = 10
        pending_jobs[job_id]["message"] = "Uploading images to Tripo3D..."
        
        async with TripoClient(api_key=TRIPO_API_KEY) as client:
            # Submit to Tripo
            if len(images) == 1:
                tripo_task_id = await client.image_to_model(image=images[0])
            else:
                tripo_task_id = await client.multiview_to_model(images=images)
            
            pending_jobs[job_id]["tripo_task_id"] = tripo_task_id
            pending_jobs[job_id]["progress"] = 20
            pending_jobs[job_id]["message"] = "Task submitted, generating 3D model..."
            print(f"Job {job_id}: Tripo task created: {tripo_task_id}")
            
            # Poll for completion
            while True:
                task = await client.get_task(tripo_task_id)
                
                if task.status == TaskStatus.SUCCESS:
                    model_url = None
                    thumbnail_url = None
                    if task.output:
                        model_url = task.output.pbr_model or task.output.model or task.output.base_model
                        thumbnail_url = task.output.rendered_image
                    
                    # Save to database
                    model_id = db.save_model(
                        task_id=tripo_task_id,
                        model_url=model_url,
                        name=name,
                        thumbnail_url=thumbnail_url,
                        view_config=view_config
                    )
                    
                    pending_jobs[job_id]["status"] = "completed"
                    pending_jobs[job_id]["progress"] = 100
                    pending_jobs[job_id]["message"] = "Model generation complete!"
                    pending_jobs[job_id]["result"] = {
                        "model_id": model_id,
                        "model_url": model_url,
                        "thumbnail_url": thumbnail_url,
                        "view_config": view_config
                    }
                    print(f"Job {job_id}: Completed successfully")
                    break
                    
                elif task.status == TaskStatus.FAILED:
                    pending_jobs[job_id]["status"] = "failed"
                    pending_jobs[job_id]["progress"] = 0
                    pending_jobs[job_id]["message"] = "Model generation failed"
                    pending_jobs[job_id]["error"] = str(task.status)
                    print(f"Job {job_id}: Failed")
                    break
                    
                else:
                    # Still processing - update progress based on status
                    progress = pending_jobs[job_id].get("progress", 20)
                    if progress < 90:
                        pending_jobs[job_id]["progress"] = min(progress + 5, 90)
                    
                    status_msg = str(task.status).replace("TaskStatus.", "")
                    pending_jobs[job_id]["message"] = f"Processing: {status_msg}"
                    
                await asyncio.sleep(3)  # Poll every 3 seconds
                
    except Exception as e:
        pending_jobs[job_id]["status"] = "failed"
        pending_jobs[job_id]["progress"] = 0
        pending_jobs[job_id]["message"] = f"Error: {str(e)}"
        pending_jobs[job_id]["error"] = str(e)
        print(f"Job {job_id}: Error - {e}")
    
    finally:
        # Clean up temp files
        for img_path in images:
            try:
                os.unlink(img_path)
            except:
                pass


@api_router.post("/convert-multiview")
async def convert_multiview(
    background_tasks: BackgroundTasks,
    front: Optional[UploadFile] = File(None),
    back: Optional[UploadFile] = File(None),
    left: Optional[UploadFile] = File(None),
    right: Optional[UploadFile] = File(None),
    name: Optional[str] = Form(None)
):
    """Submit images for async 3D model conversion. Returns job_id for polling."""
    
    if not TRIPO_API_KEY:
        raise HTTPException(status_code=500, detail="TRIPO_API_KEY not configured")
    
    if not front:
        raise HTTPException(status_code=400, detail="Front view image is required")
    
    view_config = {}
    images = []
    
    async def save_upload(upload: UploadFile, view_name: str) -> str:
        contents = await upload.read()
        suffix = os.path.splitext(upload.filename)[1].lower() or ".jpg"
        if suffix not in ['.jpg', '.jpeg', '.png', '.webp']:
            suffix = '.jpg'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(contents)
        temp_file.close()
        view_config[view_name] = upload.filename
        return temp_file.name
    
    # Save all uploads
    images.append(await save_upload(front, "front"))
    if back:
        images.append(await save_upload(back, "back"))
    if left:
        images.append(await save_upload(left, "left"))
    if right:
        images.append(await save_upload(right, "right"))
    
    # Create job
    job_id = str(uuid.uuid4())
    pending_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Job queued, starting...",
        "name": name,
        "view_count": len(images)
    }
    
    # Start background processing
    background_tasks.add_task(process_conversion_job, job_id, images, name, view_config)
    
    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "message": "Conversion job started"
    })


@api_router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a conversion job for polling."""
    if job_id not in pending_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = pending_jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
    }
    
    # Include result if completed
    if job.get("status") == "completed" and "result" in job:
        response["result"] = job["result"]
    
    # Include error if failed
    if job.get("status") == "failed" and "error" in job:
        response["error"] = job["error"]
    
    return response


@api_router.get("/models")
async def get_models():
    """Get all saved models."""
    models = db.get_all_models()
    return {"models": models}


@api_router.get("/models/{model_id}")
async def get_model(model_id: int):
    """Get a specific model by ID."""
    model = db.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@api_router.put("/models/{model_id}")
async def update_model(model_id: int, update: ModelUpdate):
    """Update model metadata."""
    success = db.update_model(model_id, name=update.name, description=update.description)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found or no changes made")
    return {"status": "updated", "model_id": model_id}


@api_router.delete("/models/{model_id}")
async def delete_model(model_id: int):
    """Delete a model."""
    success = db.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deleted", "model_id": model_id}


app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
