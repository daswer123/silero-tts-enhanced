import os
import re
import timeit
import torch
import wave
import yaml
import requests
from loguru import logger
from number2text.number2text import NumberToText
from silero_tts.lang_data import is_cyrillic, is_latin, lang_data
from silero_tts.transliterate import reverse_transliterate, transliterate

class SileroTTS:
    def __init__(self, model_id: str, language: str, speaker: str = None, sample_rate: int = 48000, device: str = 'cpu',
                 put_accent=True, put_yo=True, num_threads=6):
        self.model_id = model_id
        self.language = language
        self.sample_rate = sample_rate
        self.device = device
        self.put_accent = put_accent
        self.put_yo = put_yo
        self.num_threads = num_threads

        self.models_config = self.load_models_config()
        self.tts_model = self.init_model()

        if speaker is None:
            self.speaker = self.tts_model.speakers[0]
        else:
            self.speaker = speaker

        self.validate_model()

        self.converter = NumberToText(self.language)
        self.wave_channels = 1  # Mono
        self.wave_header_size = 44  # Bytes
        self.wave_sample_width = int(16 / 8)  # 16 bits == 2 bytes

    def load_models_config(self):
        models_file = os.path.join(os.path.dirname(__file__), 'latest_silero_models.yml')

        if not os.path.exists(models_file):
            logger.warning(f"Models config file not found: {models_file}. Downloading...")
            self.download_models_config(models_file)

        with open(models_file, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)
        logger.success(f"Models config loaded from: {models_file}")
        return models_config

    def download_models_config(self, models_file=None):
        url = "https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml"
        response = requests.get(url)
        
        if models_file is None:
            models_file = os.path.join(os.path.dirname(__file__), 'models.yml')

        if response.status_code == 200:
            with open(models_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logger.success(f"Models config file downloaded: {models_file}")
        else:
            logger.error(f"Failed to download models config file. Status code: {response.status_code}")
            raise Exception(f"Failed to download models config file. Status code: {response.status_code}")

    def get_available_speakers(self):
        return self.tts_model.speakers
    
    def get_available_sample_rates(self):
        model_config = self.models_config['tts_models'][self.language][self.model_id]['latest']
        sample_rates = model_config.get('sample_rate', [])

        if not isinstance(sample_rates, list):
            sample_rates = [sample_rates]

        return sample_rates

    def validate_model(self):
        model_config = self.models_config['tts_models'][self.language][self.model_id]['latest']

        if self.sample_rate not in model_config['sample_rate']:
            logger.error(f"Sample rate {self.sample_rate} is not supported for model '{self.model_id}'. Supported sample rates: {model_config['sample_rate']}")
            raise ValueError(f"Sample rate {self.sample_rate} is not supported for model '{self.model_id}'. Supported sample rates: {model_config['sample_rate']}")

        if self.speaker and self.speaker not in self.tts_model.speakers:
            logger.error(f"Speaker '{self.speaker}' is not supported for model '{self.model_id}'. Supported speakers: {self.tts_model.speakers}")
            raise ValueError(f"Speaker '{self.speaker}' is not supported for model '{self.model_id}'. Supported speakers: {self.tts_model.speakers}")

    
    def change_language(self, language):
        if language not in self.get_available_languages():
            logger.error(f"Language '{language}' is not supported.")
            logger.info(f"Available languages: {', '.join(self.get_available_languages())}")
            return

        self.language = language
        self.model_id = self.get_latest_model(language)
        available_sample_rates = self.get_available_sample_rates()
        self.sample_rate = max(available_sample_rates)
        self.tts_model,_= self.init_model()
        self.speaker = self.tts_model.speakers[0]
        self.validate_model()

        logger.success(f"Language changed to: {language}. Using the latest model: {self.model_id}")
        logger.info(f"Available speakers for the new model: {', '.join(self.get_available_speakers())}")
        logger.info(f"Sample rate set to the highest available: {self.sample_rate}")

    def change_model(self, model_id):
        if model_id not in self.get_available_models()[self.language]:
            logger.error(f"Model '{model_id}' is not available for language '{self.language}'.")
            logger.info(f"Available models for {self.language}: {', '.join(self.get_available_models()[self.language])}")
            return

        self.model_id = model_id
        available_sample_rates = self.get_available_sample_rates()
        self.sample_rate = max(available_sample_rates)
        self.tts_model,_ = self.init_model()
        self.speaker = self.tts_model.speakers[0]
        self.validate_model()

        logger.success(f"Model changed to: {model_id}")
        logger.info(f"Available speakers for the new model: {', '.join(self.get_available_speakers())}")
        logger.info(f"Sample rate set to the highest available: {self.sample_rate}")

    
    def change_speaker(self, speaker):
        if speaker not in self.get_available_speakers():
            logger.error(f"Speaker '{speaker}' is not supported for the current model '{self.model_id}'.")
            logger.info(f"Available speakers for this model: {', '.join(self.get_available_speakers())}")
            return

        self.speaker = speaker
        logger.success(f"Speaker changed to: {speaker}")


    def change_sample_rate(self, sample_rate):
        available_sample_rates = self.get_available_sample_rates()

        if sample_rate not in available_sample_rates:
            logger.error(f"Sample rate {sample_rate} is not supported for the current model '{self.model_id}'.")
            logger.info(f"Available sample rates for this model: {', '.join(map(str, available_sample_rates))}")
            return

        self.sample_rate = sample_rate
        logger.success(f"Sample rate changed to: {sample_rate}")

    def init_model(self):
        logger.info("Initializing model")
        t0 = timeit.default_timer()

        if not torch.cuda.is_available() and self.device == "auto":
            self.device = 'cpu'
        if torch.cuda.is_available() and (self.device == "auto" or self.device == "cuda"):
            torch_dev = torch.device("cuda", 0)
            gpus_count = torch.cuda.device_count()
            logger.info(f"Using {gpus_count} GPU(s)...")
        else:
            torch_dev = torch.device(self.device)
        torch.set_num_threads(self.num_threads)

        # Create silero_models directory
        silero_models_dir = os.path.join(os.path.dirname(__file__), 'silero_models')
        if not os.path.exists(silero_models_dir):
            os.makedirs(silero_models_dir)

        # Get package URL from models config
        package_url = self.models_config['tts_models'][self.language][self.model_id]['latest']['package']

        # Define model file path
        model_file_name = f"{self.model_id}_{self.language}.pt"
        model_file_path = os.path.join(silero_models_dir, model_file_name)

        # Download model file if not exists
        if not os.path.exists(model_file_path):
            logger.info(f"Downloading model from {package_url} to {model_file_path}")
            response = requests.get(package_url, stream=True)
            if response.status_code == 200:
                with open(model_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.success(f"Model downloaded successfully.")
            else:
                logger.error(f"Failed to download model file. Status code: {response.status_code}")
                raise Exception(f"Failed to download model file. Status code: {response.status_code}")

        # Load model from local file
        logger.info("Loading model")
        t1 = timeit.default_timer()
        from torch.package import PackageImporter
        model = PackageImporter(model_file_path).load_pickle("tts_models", "model")
        model.to(torch_dev)
        logger.info(f"Model to device takes {timeit.default_timer() - t1:.2f} seconds")

        if torch.cuda.is_available() and (self.device == "auto" or self.device == "cuda"):
            logger.info("Synchronizing CUDA")
            t2 = timeit.default_timer()
            torch.cuda.synchronize()
            logger.info(f"Cuda Synch takes {timeit.default_timer() - t2:.2f} seconds")
        logger.success("Model is loaded")

        return model

    def find_char_positions(self, string: str, char: str) -> list:
        pos = []  # list to store positions for each 'char' in 'string'
        for n in range(len(string)):
            if string[n] == char:
                pos.append(n)
        return pos

    def find_max_char_position(self, positions: list, limit: int) -> int:
        max_position = 0
        for pos in positions:
            if pos < limit:
                max_position = pos
            else:
                break
        return max_position

    def find_split_position(self, line: str, old_position: int, char: str, limit: int) -> int:
        positions = self.find_char_positions(line, char)
        new_position = self.find_max_char_position(positions, limit)
        position = max(new_position, old_position)
        return position

    def spell_digits(self, line) -> str:
        digits = re.findall(r'\d+', line)
        # Sort digits from largest to smallest - else "1 11" will be "один один один" but not "один одиннадцать"
        digits = sorted(digits, key=len, reverse=True)
        for digit in digits:
            line = line.replace(digit, self.converter.convert(int(digit[:12])))
        return line

    def preprocess_text(self, text):
        logger.info("Preprocessing text")

        if lang_data[self.language]['script'] == 'cyrillic' and is_latin(text):
            text = reverse_transliterate(text, self.language)
        elif lang_data[self.language]['script'] == 'latin' and is_cyrillic(text):
            if self.language in ["en", "fr", "es", "de"]:
                text = reverse_transliterate(text, self.language)
            else:
                text = transliterate(text, self.language)

        lines = text.split('\n')
        preprocessed_lines = []
        for line in lines:
            line = line.strip()  # Remove leading/trailing spaces
            if line == '':
                continue

            # Replace chars not supported by model
            for replacement in lang_data[self.language]['replacements']:
                line = line.replace(replacement[0], replacement[1])

            for pattern in lang_data[self.language]['patterns']:
                line = re.sub(pattern[0], pattern[1], line)

            line = self.spell_digits(line)

            preprocessed_lines.append(line)

        return preprocessed_lines

    def tts(self, text, output_file):
        # Основной метод для генерации речи
        preprocessed_lines = self.preprocess_text(text)

        # Инициализируем wav-файл
        wf = self.init_wave_file(output_file)

        logger.info("Starting TTS")
        # Синтезируем речь и пишем в файл
        for i, line in enumerate(preprocessed_lines):
            logger.info(f'Processing line {i+1}/{len(preprocessed_lines)}: {line}')
            try:
                audio = self.tts_model.apply_tts(text=line,
                                                 speaker=self.speaker,
                                                 sample_rate=self.sample_rate,
                                                 put_accent=self.put_accent,
                                                 put_yo=self.put_yo)
                wf.writeframes((audio * 32767).numpy().astype('int16'))
            except ValueError as e:
                logger.warning(f'TTS failed for line: {line}. Error: {str(e)}. Skipping...')

        wf.close()
        logger.success(f'Speech saved to {output_file}')

    def init_wave_file(self, path):
        logger.info(f'Initializing wave file: {path}')
        wf = wave.open(path, 'wb')
        wf.setnchannels(self.wave_channels)
        wf.setsampwidth(self.wave_sample_width)
        wf.setframerate(self.sample_rate)
        return wf

    def from_file(self, text_path, output_path):
        # Метод для генерации речи из текстового файла
        logger.info(f'Generating speech from file: {text_path}')
        with open(text_path, 'r') as f:
            text = f.read()

        self.tts(text, output_path)

    @staticmethod
    def get_available_models():
        models_file = os.path.join(os.path.dirname(__file__), 'latest_silero_models.yml')

        if not os.path.exists(models_file):
            logger.warning(f"Models config file not found: {models_file}. Downloading...")
            SileroTTS.download_models_config_static(models_file)

        with open(models_file, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)

        models_dict = {}
        for lang, models in models_config['tts_models'].items():
            models_dict[lang] = list(models.keys())

        return models_dict

    @staticmethod
    def get_latest_model(language):
        models_file = os.path.join(os.path.dirname(__file__), 'latest_silero_models.yml')

        if not os.path.exists(models_file):
            logger.warning(f"Models config file not found: {models_file}. Downloading...")
            SileroTTS.download_models_config_static(models_file)

        with open(models_file, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)

        models = models_config['tts_models'][language]
        latest_model = sorted(models.keys(), reverse=True)[0]
        return latest_model

    @staticmethod
    def get_available_languages():
        models_file = os.path.join(os.path.dirname(__file__), 'latest_silero_models.yml')

        if not os.path.exists(models_file):
            logger.warning(f"Models config file not found: {models_file}. Downloading...")
            SileroTTS.download_models_config_static(models_file)

        with open(models_file, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)

        return list(models_config['tts_models'].keys())
            
    
    @staticmethod
    def download_models_config_static(models_file=None):
        url = "https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml"
        response = requests.get(url)
        
        if models_file is None:
            models_file = os.path.join(os.path.dirname(__file__), 'models.yml')

        if response.status_code == 200:
            with open(models_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logger.success(f"Models config file downloaded: {models_file}")
        else:
            logger.error(f"Failed to download models config file. Status code: {response.status_code}")
            raise Exception(f"Failed to download models config file. Status code: {response.status_code}")


    @staticmethod
    def get_available_sample_rates_static(language, model_id):
        models_file = os.path.join(os.path.dirname(__file__), 'latest_silero_models.yml')

        if not os.path.exists(models_file):
            logger.warning(f"Models config file not found: {models_file}. Downloading...")
            SileroTTS.download_models_config_static(models_file)

        with open(models_file, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)

        model_config = models_config['tts_models'][language][model_id]['latest']
        sample_rates = model_config.get('sample_rate', [])

        if not isinstance(sample_rates, list):
            sample_rates = [sample_rates]

        return sample_rates


if __name__== '__main__':
    tts = SileroTTS(model_id='v4_ru',
                    language='ru',
                    speaker='aidar',
                    sample_rate=48000,
                    device='cpu')

    # Speech generation from text
    # text = "Проверка Silero"
    # tts.tts(text, 'output.wav')

    logger.info(f"Available speakers for model {tts.model_id}: {tts.get_available_speakers()}")

    # Generating speech from a file
    # tts.from_file('input.txt', 'output.wav')

    logger.info(f"Available models: {SileroTTS.get_available_models()}")
