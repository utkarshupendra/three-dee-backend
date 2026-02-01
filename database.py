import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "models.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT UNIQUE NOT NULL,
            name TEXT,
            description TEXT,
            model_url TEXT,
            thumbnail_url TEXT,
            input_images TEXT,
            view_config TEXT,
            status TEXT DEFAULT 'completed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_model(
    task_id: str,
    model_url: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    thumbnail_url: Optional[str] = None,
    input_images: Optional[List[str]] = None,
    view_config: Optional[Dict[str, Any]] = None
) -> int:
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO models (task_id, name, description, model_url, thumbnail_url, input_images, view_config)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        task_id,
        name or f"Model {task_id[:8]}",
        description,
        model_url,
        thumbnail_url,
        json.dumps(input_images) if input_images else None,
        json.dumps(view_config) if view_config else None
    ))
    
    model_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return model_id


def get_all_models() -> List[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    models = []
    for row in rows:
        model = dict(row)
        if model.get('input_images'):
            model['input_images'] = json.loads(model['input_images'])
        if model.get('view_config'):
            model['view_config'] = json.loads(model['view_config'])
        models.append(model)
    return models


def get_model(model_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        model = dict(row)
        if model.get('input_images'):
            model['input_images'] = json.loads(model['input_images'])
        if model.get('view_config'):
            model['view_config'] = json.loads(model['view_config'])
        return model
    return None


def update_model(model_id: int, name: Optional[str] = None, description: Optional[str] = None) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    
    updates = []
    values = []
    
    if name is not None:
        updates.append("name = ?")
        values.append(name)
    if description is not None:
        updates.append("description = ?")
        values.append(description)
    
    if not updates:
        return False
    
    updates.append("updated_at = CURRENT_TIMESTAMP")
    values.append(model_id)
    
    cursor.execute(f"UPDATE models SET {', '.join(updates)} WHERE id = ?", values)
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def delete_model(model_id: int) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


# Initialize database on import
init_db()
