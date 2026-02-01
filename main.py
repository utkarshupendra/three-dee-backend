import os
import asyncio
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel
from tripo3d import TripoClient, TaskStatus
import database as db

load_dotenv()

app = FastAPI(title="2D to 3D Converter")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
allowed_origins = [FRONTEND_URL, "http://localhost:5173", "http://localhost:3000"]
if FRONTEND_URL and FRONTEND_URL not in allowed_origins:
    allowed_origins.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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


@app.get("/proxy-glb")
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


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@app.post("/convert-multiview")
async def convert_multiview(
    front: Optional[UploadFile] = File(None),
    back: Optional[UploadFile] = File(None),
    left: Optional[UploadFile] = File(None),
    right: Optional[UploadFile] = File(None),
    name: Optional[str] = Form(None)
):
    """Convert images to 3D model with specific view positions."""
    
    if not TRIPO_API_KEY:
        raise HTTPException(status_code=500, detail="TRIPO_API_KEY not configured")
    
    if not front:
        raise HTTPException(status_code=400, detail="Front view image is required")
    
    temp_files = []
    view_config = {}
    
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
    
    try:
        # Build images list in order: front, back, left, right
        images = []
        
        # Front is required
        images.append(await save_upload(front, "front"))
        temp_files.append(images[-1])
        
        # Optional views
        if back:
            images.append(await save_upload(back, "back"))
            temp_files.append(images[-1])
        if left:
            images.append(await save_upload(left, "left"))
            temp_files.append(images[-1])
        if right:
            images.append(await save_upload(right, "right"))
            temp_files.append(images[-1])
        
        async with TripoClient(api_key=TRIPO_API_KEY) as client:
            if len(images) == 1:
                task_id = await client.image_to_model(image=images[0])
            else:
                task_id = await client.multiview_to_model(images=images)
            
            print(f"Task created: {task_id}")
            task = await client.wait_for_task(task_id, verbose=True)
            
            if task.status == TaskStatus.SUCCESS:
                model_url = None
                thumbnail_url = None
                if task.output:
                    model_url = task.output.pbr_model or task.output.model or task.output.base_model
                    thumbnail_url = task.output.rendered_image
                
                # Save to database
                model_id = db.save_model(
                    task_id=task_id,
                    model_url=model_url,
                    name=name,
                    thumbnail_url=thumbnail_url,
                    view_config=view_config
                )
                
                return JSONResponse({
                    "status": "success",
                    "task_id": task_id,
                    "model_id": model_id,
                    "model_url": model_url,
                    "thumbnail_url": thumbnail_url,
                    "view_config": view_config
                })
            else:
                print(f"Task failed: {task}")
                raise HTTPException(status_code=500, detail=f"Task failed: {task.status}")
    
    finally:
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


@app.get("/models")
async def get_models():
    """Get all saved models."""
    models = db.get_all_models()
    return {"models": models}


@app.get("/models/{model_id}")
async def get_model(model_id: int):
    """Get a specific model by ID."""
    model = db.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@app.put("/models/{model_id}")
async def update_model(model_id: int, update: ModelUpdate):
    """Update model metadata."""
    success = db.update_model(model_id, name=update.name, description=update.description)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found or no changes made")
    return {"status": "updated", "model_id": model_id}


@app.delete("/models/{model_id}")
async def delete_model(model_id: int):
    """Delete a model."""
    success = db.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deleted", "model_id": model_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
