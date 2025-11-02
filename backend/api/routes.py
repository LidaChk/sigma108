from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from services.file_processor import process_csv_with_model
from services.task_manager import create_task, get_task_info, update_task_status, check_and_update_status, get_output_path, cleanup_task_files, processing_tasks, set_error_status
from logger.log_config import logger
from config import TASK_STATUS
router = APIRouter()

def process_task_with_error_handling(task_id: str, input_path: Path, output_path: Path):
    """
    Функция-обертка для обработки файла с обработкой ошибок.
    """
    try:
        process_csv_with_model(input_path, output_path)
    except Exception as e:
        error_msg = f"Ошибка при обработке файла: {str(e)}"
        logger.error(f"Задача {task_id}: {error_msg}")
        set_error_status(task_id, error_msg)
        logger.info(f"Задача {task_id}: установлен статус 'error'")
        # Очищаем временные файлы в случае ошибки
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

@router.post("/upload/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Эндпоинт для загрузки CSV-файла.
    Запускает обработку в фоне.
    Возвращает ID задачи.
    """
    if not file.filename.endswith(('.csv', '.CSV')):
        raise HTTPException(status_code=400, detail="Файл должен быть CSV")

    task_id = create_task()
    print(f"Получена загрузка для задачи {task_id}. Файл: {file.filename}")

    task_info = get_task_info(task_id)
    if not task_info:
        raise HTTPException(status_code=500, detail="Ошибка при создании задачи.")

    input_path = Path(task_info["input_file"])
    output_path = Path(task_info["output_file"])

    try:
        with open(input_path, 'wb') as f:
            content = await file.read()
            f.write(content)
    except Exception as e:

        input_path.unlink(missing_ok=True)

        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении файла: {e}")


    update_task_status(task_id, TASK_STATUS["PROCESSING"])

    background_tasks.add_task(process_task_with_error_handling, task_id, input_path, output_path)

    return {"task_id": task_id, "status": "File uploaded, processing started"}

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Эндпоинт для проверки статуса обработки.
    """
    task_info = get_task_info(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task ID not found")

    check_and_update_status(task_id)

    if task_info["status"] == "error":
        error_message = task_info.get("error_message", "Неизвестная ошибка при обработке файла")
        raise HTTPException(status_code=500, detail=error_message)

    return {"task_id": task_id, "status": task_info["status"]}

@router.get("/download/{task_id}")
async def download_file(task_id: str):
    """
    Эндпоинт для скачивания обработанного CSV-файла.
    """
    task_info = get_task_info(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task ID not found")

    output_path = get_output_path(task_id)

    # Проверяем статус задачи
    if task_info["status"] == "error":
        error_message = task_info.get("error_message", "Неизвестная ошибка при обработке файла")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {error_message}")

    if not output_path or not output_path.exists():
        if task_info["status"] != TASK_STATUS["COMPLETED"]:
             raise HTTPException(status_code=400, detail="File processing is not finished yet. Check status first.")
        else:
             raise HTTPException(status_code=500, detail="File processing failed, result not found.")

    response = FileResponse(
        path=output_path,
        filename=f"result_{task_id}.csv",
        media_type='text/csv'
    )

    return response

@router.post("/cleanup/{task_id}")
async def cleanup_files(task_id: str):
    """
    Эндпоинт для ручной очистки файлов задачи.
    """
    task_info = get_task_info(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task ID not found")

    cleanup_task_files(task_id)

    del processing_tasks[task_id]
    return {"message": f"Files for task {task_id} cleaned up."}
