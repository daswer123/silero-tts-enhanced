# test_silero_tts.py

import os
import pytest
from silero_tts.silero_tts import SileroTTS


# conftest.py

import os
import sys
import shutil
import pytest
import subprocess

# Путь к временной папке для тестов
tests_temp_path = os.path.abspath("tests/tests_temp")

# Путь к виртуальному окружению
venv_path = os.path.join(tests_temp_path, "venv")

# Функция для создания временной папки и виртуального окружения
def create_test_env():
    os.makedirs(tests_temp_path, exist_ok=True)
    if not os.path.exists(venv_path):
        subprocess.check_call([sys.executable, "-m", "venv", venv_path])

# Функция для установки silero-tts в виртуальное окружение
def install_silero_tts():
    # Укажи путь к локальному wheel-файлу или имя пакета на PyPI
    package = "C:\\WEB\\photo-session\\silero_cli\\dist\silero_tts-0.0.2-py3-none-any.whl"  # или 'silero-tts'
    subprocess.check_call([f"{venv_path}/Scripts/pip", "install", package])

# Фикстура для создания и активации тестового окружения
@pytest.fixture(scope="session", autouse=True)
def test_env():
    create_test_env()
    install_silero_tts()

    # Добавляем путь к виртуальному окружению в sys.path
    sys.path.insert(0, os.path.join(venv_path, "lib", "python3.x", "site-packages"))

    yield

    # Удаляем временную папку после завершения тестов
    shutil.rmtree(tests_temp_path)


# Функция для проверки, что файл существует и не пустой
def check_audio_file(file_path):
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0

def test_console_text_to_speech():
    input_text = os.path.join(tests_temp_path, "test_input.txt")
    with open(input_text, "w") as f:
        f.write("Привет, мир!")
    # 
    os.system(f"{tests_temp_path}/venv/Scripts/python -m silero_tts --language ru  --text {input_text} --output-file {tests_temp_path}/test_output.wav")

    check_audio_file(os.path.join(tests_temp_path, "test_output.wav"))

def test_api_text_to_speech():
    tts = SileroTTS(model_id="v4_ru", language="ru")

    input_text = os.path.join(tests_temp_path, "test_input.txt")
    with open(input_text, "w") as f:
        f.write("Проверка синтеза")

    tts.from_file(input_text, os.path.join(tests_temp_path, "test_output.wav"))

    check_audio_file(os.path.join(tests_temp_path, "test_output.wav"))

def test_batch_processing():
    texts_dir = os.path.join(tests_temp_path, "test_texts")
    os.makedirs(texts_dir, exist_ok=True)

    for i in range(3):
        with open(os.path.join(texts_dir, f"text_{i}.txt"), "w") as f:
            f.write(f"Тестовый текст {i}")

    os.system(f"{tests_temp_path}/venv/Scripts/python -m silero_tts --language ru --input-dir {texts_dir} --output-dir {tests_temp_path}/test_audio")

    for i in range(3):
        check_audio_file(os.path.join(tests_temp_path, "test_audio", f"text_{i}.wav"))

def test_transliteration_en():
    tts = SileroTTS(model_id="v3_en", language="en")

    input_text = os.path.join(tests_temp_path, "test_input.txt")
    with open(input_text, "w") as f:
        f.write("Проверка транслитерации")

    tts.from_file(input_text, os.path.join(tests_temp_path, "test_output.wav"))

    check_audio_file(os.path.join(tests_temp_path, "test_output.wav"))

def test_transliteration_ru():
    tts = SileroTTS(model_id="v4_ru", language="ru")

    input_text = os.path.join(tests_temp_path, "test_input.txt")
    with open(input_text, "w") as f:
        f.write("Proverka transliteratsii")

    tts.from_file(input_text, os.path.join(tests_temp_path, "test_output.wav"))

    check_audio_file(os.path.join(tests_temp_path, "test_output.wav"))
