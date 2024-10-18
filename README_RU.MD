# Silero TTS Enhanced

**README доступен на следующих языках:**

[![EN](https://img.shields.io/badge/EN-blue.svg)](https://github.com/daswer123/silero-tts-enhanced)
[![RU](https://img.shields.io/badge/RU-red.svg)](https://github.com/daswer123/silero-tts-enhanced/blob/main/README_RU.MD)

Silero TTS Enhanced - это библиотека Python, которая предоставляет удобный способ синтеза речи из текста с использованием различных моделей Silero TTS, языков и голосов. Она может использоваться как автономный скрипт или интегрироваться в ваши собственные проекты на Python.

## Особенности

- Поддержка нескольких языков и моделей
- Автоматическая загрузка последнего файла конфигурации модели
- Предварительная обработка текста и транслитерация
- Пакетная обработка текстовых файлов
- Подробное логирование с помощью loguru
- Отслеживание прогресса с помощью tqdm
- Настраиваемые параметры частоты дискретизации, устройства и многое другое
- Может использоваться как автономный скрипт или интегрироваться в код Python

## Установка

### Автоматически (рекомендуется)

   ```
   pip install silero-tts
   ```

### Вручную
1. Клонируйте репозиторий:
   ```
   git clone https://github.com/daswer123/silero-tts-enhanced.git
   ```

2. Установите необходимые зависимости:
   ```
   pip install -r requirements.txt
   ```

## Использование

### В качестве автономного скрипта

Вы можете использовать Silero TTS в качестве автономного скрипта для синтеза речи из текстовых файлов или каталогов, содержащих текстовые файлы.

```
python -m silero_tts [параметры]
```

#### Параметры

- `--list-models`: Список доступных моделей
- `--list-speakers`: Список доступных голосов для модели
- `--language LANGUAGE`: Укажите код языка (обязательно)
- `--model MODEL`: Укажите идентификатор модели (по умолчанию: последняя версия для языка)
- `--speaker SPEAKER`: Укажите имя голоса (по умолчанию: первый доступный голос для модели)
- `--sample-rate SAMPLE_RATE`: Укажите частоту дискретизации (по умолчанию: 48000)
- `--device DEVICE`: Укажите устройство для использования (по умолчанию: cpu)
- `--text TEXT`: Укажите текст для синтеза
- `--input-file INPUT_FILE`: Укажите входной текстовый файл для синтеза
- `--input-dir INPUT_DIR`: Укажите входной каталог с текстовыми файлами для синтеза
- `--output-file OUTPUT_FILE`: Укажите выходной аудиофайл (по умолчанию: output.wav)
- `--output-dir OUTPUT_DIR`: Укажите выходной каталог для синтезированных аудиофайлов (по умолчанию: output)
- `--log-level INFO` : Укажите уровень журнала, вы можете отключить его, используя значение NONE. (по умолчанию: INFO)

#### Примеры

1. Синтез речи из текста:
   ```
   python silero_tts.py --language ru --text "Привет, мир!"
   ```

2. Синтез речи из текстового файла:
   ```
   python silero_tts.py --language en --input-file input.txt --output-file output.wav
   ```

3. Синтез речи из нескольких текстовых файлов в каталоге:
   ```
   python silero_tts.py --language es --input-dir texts --output-dir audio
   ```

### В качестве библиотеки Python

Вы также можете интегрировать Silero TTS в свои собственные проекты на Python, импортируя класс `SileroTTS` и используя его методы.

```python
from silero_tts.silero_tts import SileroTTS

# Get available models
models = SileroTTS.get_available_models()
print("Available models:", models)

# Get available languages
languages = SileroTTS.get_available_languages()
print("Available languages:", languages)

# Get the latest model for a specific language
latest_model = SileroTTS.get_latest_model('ru')
print("Latest model for Russian:", latest_model)

# Get available sample rates for a specific model and language
sample_rates = SileroTTS.get_available_sample_rates_static('ru', latest_model)
print("Available sample rates for the latest Russian model:", sample_rates)

# Initialize the TTS object
tts = SileroTTS(model_id='v3_en', language='en', speaker='en_2', sample_rate=48000, device='cpu')

# Synthesize speech from text
text = "Hello world!"
tts.tts(text, 'output.wav')

# Synthesize speech from a text file
# tts.from_file('input.txt', 'output.wav')

# Get available speakers for the current model
speakers = tts.get_available_speakers()
print("Available speakers for the current model:", speakers)

# Change the language
tts.change_language('en')
print("Language changed to:", tts.language)
print("New model ID:", tts.model_id)
print("New available speakers:", tts.get_available_speakers())

# Change the model
tts.change_model('v3_en')
print("Model changed to:", tts.model_id)
print("New available speakers:", tts.get_available_speakers())

# Change the speaker
tts.change_speaker('en_0')
print("Speaker changed to:", tts.speaker)

# Change the sample rate
tts.change_sample_rate(24000)
print("Sample rate changed to:", tts.sample_rate)
```

## Особенности CLI

CLI Silero TTS предоставляет следующие возможности:

-**Поддержка языков**: Укажите код языка с помощью флага `--language`, чтобы синтезировать речь на нужном языке.
-**Выбор модели**: Выберите конкретную модель с помощью флага `--model` или позвольте CLI автоматически выбрать последнюю модель для указанного языка.
-**Выбор голоса**: Выберите голос с помощью флага `--speaker` или используйте голос по умолчанию для выбранной модели.
-**Частота дискретизации**: Настройте частоту дискретизации синтезированной речи с помощью флага `--sample-rate`.
-**Устройство**: Укажите устройство (CPU или GPU) для использования при синтезе с помощью флага `--device`.
-**Ввод текста**: Предоставьте текст для синтеза непосредственно с помощью флага `--text` или укажите входной текстовый файл с помощью флага `--input-file`.
-**Пакетная обработка**: Обработка нескольких текстовых файлов в каталоге с помощью флага `--input-dir`.
-**Вывод**: Укажите выходной аудиофайл с помощью флага `--output-file` или выходной каталог для пакетной обработки с помощью флага `--output-dir`.
-**Список моделей**: Список всех доступных моделей с помощью флага `--list-models`.
-**Список голосов**: Список всех доступных голосов для конкретной модели с помощью флага `--list-speakers`.

## Поддерживаемые языки

- Русский (ru)
- Английский (en)
- Немецкий (de)
- Испанский (es)
- Французский (fr)
- Башкирский (ba)
- Калмыцкий (xal)
- Татарский (tt)
- Узбекский (uz)
- Украинский (ua)
- Индийский (indic)
- Кириллический (cyrillic)

## Лицензия

Этот проект лицензирован под [MIT License](LICENSE).

## Благодарности

- [Silero Models](https://github.com/snakers4/silero-models) за предоставление моделей TTS
- [silero_tts_standalone](https://github.com/S-trace/silero_tts_standalone) эта библиотека вдохновила меня на создание этого проекта.
