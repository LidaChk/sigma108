from pathlib import Path
from typing import Dict, Optional
import uuid
from config import UPLOADS_DIR, PROCESSED_DIR

processing_tasks: Dict[str, Dict] = {}

def create_task() -> str:
    """Создаёт новую задачу, возвращает task_id."""
    task_id = str(uuid.uuid4())

    input_file_path = UPLOADS_DIR / f"{task_id}_input.csv"
    output_file_path = PROCESSED_DIR / f"result_{task_id}.csv"
    processing_tasks[task_id] = {
        "status": "created",
        "input_file": str(input_file_path),
        "output_file": str(output_file_path)
    }
    return task_id

def get_task_info(task_id: str) -> Optional[Dict]:
    return processing_tasks.get(task_id)

def update_task_status(task_id: str, status: str):
    if task_id in processing_tasks:
        processing_tasks[task_id]["status"] = status

def check_and_update_status(task_id: str):
    task_info = processing_tasks.get(task_id)
    if task_info and task_info["status"] == "processing" and task_info["output_file"]:
        output_path = Path(task_info["output_file"])
        if output_path.exists():
            task_info["status"] = "completed"
            return True
    return False

def get_output_path(task_id: str) -> Optional[Path]:
    task_info = processing_tasks.get(task_id)
    if task_info and task_info["output_file"]:
        return Path(task_info["output_file"])
    return None

def get_input_path(task_id: str) -> Optional[Path]:
    task_info = processing_tasks.get(task_id)
    if task_info and task_info["input_file"]:
        return Path(task_info["input_file"])
    return None

# TODO: Реализовать очистку файлов
def cleanup_task_files(task_id: str):
    task_info = processing_tasks.get(task_id)
    if task_info:
        input_path = Path(task_info.get("input_file", ""))
        output_path = Path(task_info.get("output_file", ""))

        if input_path.exists():
            input_path.unlink()
            print(f"Удалён входной файл задачи {task_id}: {input_path}")

        if output_path.exists():
            output_path.unlink()
            print(f"Удалён выходной файл задачи {task_id}: {output_path}")

        del processing_tasks[task_id]
