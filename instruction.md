# Инструкция по запуску и использованию веб-приложения Sigma108

**Важно:** Из-за смешанного содержимого (фронтенд по HTTPS, бэкенд по HTTP) для корректной работы в браузере Google Chrome необходимо использовать специальные флаги безопасности при запуске. Это позволяет браузеру выполнять небезопасные HTTP-запросы к бэкенду.

## Шаг 1: Запуск Google Chrome с необходимыми флагами

Для корректной работы приложения **запускайте Google Chrome по одной из следующих команд** (в зависимости от вашей операционной системы):

- **Windows (в командной строке `cmd` или PowerShell):**

  ```bash
  "C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-web-security --allow-running-insecure-content --disable-features=VizDisplayCompositor --ignore-certificate-errors --new-window "https://lidachk.github.io/sigma108/"
  ```

  _Примечание:_ Убедитесь, что путь к `chrome.exe` указан верно (может отличаться, если Chrome установлен в другую папку).

- **macOS (в терминале):**

  ```bash
  open -a "Google Chrome" --args --disable-web-security --allow-running-insecure-content --disable-features=VizDisplayCompositor --ignore-certificate-errors --new-window "https://lidachk.github.io/sigma108/"
  ```

Эти флаги отключают защиту от смешанного содержимого и позволяют фронтенду общаться с HTTP-бэкендом.

## Шаг 2: Использование веб-интерфейса

1. **Откройте** браузер Chrome, запущенный с флагами из Шага 1.
2. Перейдите по адресу: `https://lidachk.github.io/sigma108/`.
3. Вы увидите интерфейс загрузки файла.
4. **Нажмите** на область загрузки или кнопку "Выберите файл" и выберите ваш CSV-файл.
5. **Нажмите** кнопку "Отправить на обработку".
6. Статус обработки и идентификатор задачи `taskId` будет отображаться на странице.
7. После завершения обработки появится кнопка "Скачать результат". **Нажмите** её, чтобы сохранить обработанный CSV-файл на ваш компьютер.

---

## Альтернативное использование: API через `curl`

Для пользователей, знакомых с командной строкой, или для автоматизации, можно напрямую взаимодействовать с API бэкенда с помощью `curl`. Это позволяет обойти ограничения браузера.

**Базовый URL API:** `http://185.135.83.219/api`

### 1. Загрузка файла

Отправляет CSV-файл на сервер для обработки.

```bash
curl -X POST "http://185.135.83.219/api/upload/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/file.csv;type=text/csv"
```

**Примеры путей к файлам:**

- Windows:

  ```bash
  curl -X POST "http://185.135.83.219/api/upload/" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@D:\data\my_test_file.csv;type=text/csv"
  ```

- Linux/macOS:
  ```bash
  curl -X POST "http://185.135.83.219/api/upload/" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/home/user/data/my_test_file.csv;type=text/csv"
  ```

**Ответ (если успешно):**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
  "status": "File uploaded, processing started"
}
```

Сохраните `task_id` для следующих шагов.

### 2. Проверка статуса обработки

Проверяет, завершена ли обработка файла.

```bash
curl -X GET "http://185.135.83.219/api/status/a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8" \
  -H "accept: application/json"
```

**Возможные ответы:**

**В обработке:**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
  "status": "processing"
}
```

**Завершено:**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
  "status": "completed"
}
```

**Ошибка (если произошла):**

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
  "status": "failed"
}
```

### 3. Скачивание результата

Скачивает обработанный CSV-файл.

```bash
curl -X GET "http://185.135.83.219/api/download/a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8" \
  -H "accept: text/csv" \
  --output result_a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8.csv
```

**Примечание:**

- Замените `a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8` на ваш `task_id`.
- Замените `result_a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8.csv` на желаемое имя файла для сохранения.
- Команда скачает файл и сохранит его в текущую директорию терминала с указанным именем.
