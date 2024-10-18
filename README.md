# Silero TTS

**README is available in the following languages:**

[![EN](https://img.shields.io/badge/EN-blue.svg)](https://github.com/daswer123/silero-tts-enhanced)
[![RU](https://img.shields.io/badge/RU-red.svg)](https://github.com/daswer123/silero-tts-enhanced/blob/main/README_RU.MD)

Silero TTS is a Python library that provides an easy way to synthesize speech from text using various Silero TTS models, languages, and speakers. It can be used as a standalone script or integrated into your own Python projects.

## Features

- Support for multiple languages and models
- Automatic downloading of the latest model configuration file
- Text preprocessing and transliteration
- Batch processing of text files
- Detailed logging with loguru
- Progress tracking with tqdm
- Customizable options for sample rate, device, and more
- Can be used as a standalone script or integrated into Python code

## Installation

### Auto ( Recomended )
 
   ```
   pip install silero-tts
   ```

### Manualy
1. Clone the repository:
   ```
   git clone https://github.com/daswer123/silero-tts-enhanced
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### As a Standalone Script

You can use Silero TTS as a standalone script to synthesize speech from text files or directories containing text files.

```
python -m silero_tts [options]
```

#### Options

- `--list-models`: List available models
- `--list-speakers`: List available speakers for a model
- `--language LANGUAGE`: Specify the language code (required)
- `--model MODEL`: Specify the model ID (default: latest version for the language)
- `--speaker SPEAKER`: Specify the speaker name (default: first available speaker for the model)
- `--sample-rate SAMPLE_RATE`: Specify the sample rate (default: 48000)
- `--device DEVICE`: Specify the device to use (default: cpu)
- `--text TEXT`: Specify the text to synthesize
- `--input-file INPUT_FILE`: Specify the input text file to synthesize
- `--input-dir INPUT_DIR`: Specify the input directory with text files to synthesize
- `--output-file OUTPUT_FILE`: Specify the output audio file (default: output.wav)
- `--output-dir OUTPUT_DIR`: Specify the output directory for synthesized audio files (default: output)
- `--log-level INFO` : Specify log-level, you can turn off use NONE value (default: INFO)

#### Examples

1. Synthesize speech from a text:
   ```
   python silero_tts.py --language ru --text "Привет, мир!"
   ```

2. Synthesize speech from a text file:
   ```
   python silero_tts.py --language en --input-file input.txt --output-file output.wav
   ```

3. Synthesize speech from multiple text files in a directory:
   ```
   python silero_tts.py --language es --input-dir texts --output-dir audio
   ```

### As a Python Library

You can also integrate Silero TTS into your own Python projects by importing the `SileroTTS` class and using its methods.

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

## CLI Features

The Silero TTS CLI provides the following features:

- **Language Support**: Specify the language code using the `--language` flag to synthesize speech in the desired language.
- **Model Selection**: Choose a specific model using the `--model` flag or let the CLI automatically select the latest model for the specified language.
- **Speaker Selection**: Select a speaker using the `--speaker` flag or use the default speaker for the chosen model.
- **Sample Rate**: Customize the sample rate of the synthesized speech using the `--sample-rate` flag.
- **Device**: Specify the device (CPU or GPU) to use for synthesis using the `--device` flag.
- **Text Input**: Provide the text to synthesize directly using the `--text` flag or specify an input text file using the `--input-file` flag.
- **Batch Processing**: Process multiple text files in a directory using the `--input-dir` flag.
- **Output**: Specify the output audio file using the `--output-file` flag or the output directory for batch processing using the `--output-dir` flag.
- **Model Listing**: List all available models using the `--list-models` flag.
- **Speaker Listing**: List all available speakers for a specific model using the `--list-speakers` flag.

## Supported Languages

- Russian (ru)
- English (en)
- German (de)
- Spanish (es)
- French (fr)
- Bashkir (ba)
- Kalmyk (xal)
- Tatar (tt)
- Uzbek (uz)
- Ukrainian (ua)
- Indic (indic)
- Cyrillic (cyrillic)

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Silero Models](https://github.com/snakers4/silero-models) for providing the TTS models
- [silero_tts_standalone](https://github.com/S-trace/silero_tts_standalone) this library inspired me to create this project.
