import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from peft import PeftModel
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIModel:
    def __init__(self, model_name_or_path, model_type=None, device=None, **kwargs):
        self.raw_model_name = model_name_or_path
        self.model_type = model_type or self._infer_model_type(model_name_or_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None
        self.pipe = None

        # Нормализуем путь сразу для всех типов
        self.model_path = self._normalize_path(model_name_or_path)
        self._load()

    def _normalize_path(self, path):
        """Преобразует относительный путь в абсолютный и нормализует разделители."""
        path_obj = Path(path).expanduser().resolve()
        # Возвращаем как строку с прямыми слешами (совместимо с Windows)
        return path_obj.as_posix()

    def _infer_model_type(self, name):
        name_lower = str(name).lower()
        if any(x in name_lower for x in ['stable-diffusion', 'sdxl', 'diffusion', 'novaanime']):
            return 'image'
        elif any(x in name_lower for x in ['gpt', 'llama', 'qwen', 'mistral', 'deepseek']):
            return 'text'
        else:
            # Если имя заканчивается на .safetensors, скорее всего это модель изображения
            if name_lower.endswith('.safetensors'):
                return 'image'
            return 'text'

    def _load(self):
        logger.info(f"Загрузка модели {self.raw_model_name} типа {self.model_type} на {self.device}")
        logger.info(f"Нормализованный путь: {self.model_path}")

        if self.model_type == 'text':
            self._load_text_model()
        elif self.model_type == 'image':
            self._load_image_model()
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.model_type}")

        logger.info("Модель успешно загружена")

    def _load_text_model(self):
        quantization_config = None
        if self.kwargs.get('load_in_4bit', False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

        # Проверяем, существует ли локальная папка
        local_path = Path(self.model_path)
        if local_path.exists() and local_path.is_dir():
            model_identifier = self.model_path  # используем нормализованный путь
            logger.info(f"Загружаем из локальной папки: {model_identifier}")
        else:
            model_identifier = self.raw_model_name  # используем оригинальное имя (repo_id)
            logger.info(f"Загружаем с Hugging Face Hub: {model_identifier}")

        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_identifier,
            trust_remote_code=True,
            use_fast=True,
            # local_files_only=local_path.exists()  # можно добавить при необходимости
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Загрузка модели
        self.model = AutoModelForCausalLM.from_pretrained(
            model_identifier,
            quantization_config=quantization_config,
            device_map='auto' if self.device == 'cuda' else None,
            dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            trust_remote_code=True
        )

        # Загрузка LoRA
        if 'lora_adapter_path' in self.kwargs:
            lora_path = self.kwargs['lora_adapter_path']
            logger.info(f"Загрузка LoRA адаптера из {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        if self.device == 'cpu' and not hasattr(self.model, 'hf_device_map'):
            self.model = self.model.to(self.device)

        self.pipe = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device if self.device == 'cuda' else -1,
            **self.kwargs.get('pipeline_kwargs', {})
        )

    def _load_image_model(self):
        local_path = Path(self.model_path)

        # Если это одиночный файл .safetensors
        if str(self.model_path).endswith('.safetensors'):
            if not local_path.exists():
                raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
            logger.info(f"Загружаем из файла: {self.model_path}")
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                self.model_path,  # используем нормализованный путь
                dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                use_safetensors=True,
                **self.kwargs.get('pipeline_kwargs', {})
            )
        else:
            # Папка или repo_id
            if local_path.exists() and local_path.is_dir():
                model_identifier = self.model_path
                logger.info(f"Загружаем из локальной папки: {model_identifier}")
            else:
                model_identifier = self.raw_model_name
                logger.info(f"Загружаем с Hugging Face Hub: {model_identifier}")

            self.pipe = DiffusionPipeline.from_pretrained(
                model_identifier,
                dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                **self.kwargs.get('pipeline_kwargs', {})
            )

        if self.device == 'cuda':
            self.pipe = self.pipe.to('cuda')
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()
        else:
            self.pipe = self.pipe.to('cpu')

    def generate(self, prompt, **gen_kwargs):
        if self.model_type == 'text':
            if self.pipe:
                outputs = self.pipe(prompt, **gen_kwargs)
                return outputs[0]['generated_text']
            else:
                inputs = self.tokenizer(prompt, return_tensors='pt')
                if self.device == 'cuda':
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elif self.model_type == 'image':
            default_kwargs = {
                'width': 512,
                'height': 512,
                'num_inference_steps': 25,
                'guidance_scale': 5,
            }
            default_kwargs.update(gen_kwargs)
            images = self.pipe(prompt, **default_kwargs).images
            return images[0]

        else:
            raise RuntimeError(f"Неизвестный тип модели для генерации: {self.model_type}")

    def to(self, device):
        self.device = device
        if self.model_type == 'text':
            if hasattr(self.model, 'to'):
                self.model = self.model.to(device)
        elif self.model_type == 'image':
            self.pipe = self.pipe.to(device)
        return self

    def save_pretrained(self, save_path):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        if self.model_type == 'text':
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(save_path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(save_path)
        elif self.model_type == 'image':
            self.pipe.save_pretrained(save_path)
        logger.info(f"Модель сохранена в {save_path}")

    def __repr__(self):
        return f"<AIModel type={self.model_type} name={self.raw_model_name} device={self.device}>"
    
if __name__ == "__main__":
    #1. Текстовая модель (Qwen 2.5 7B) с 4-битной загрузкой
    text_model = AIModel(
        "F:/AI/Models/Qwen2.5-7B-Instruct",
        model_type="text",
        load_in_4bit=True,  # экономия памяти
    )
    response = text_model.generate("Какой билд собирать на нахиду из геншин импакт?", max_new_tokens=100)
    print("Ответ модели:", response)
    response = text_model.generate("Какой билд собирать на райден из геншин импакт?", max_new_tokens=100)
    print("Ответ модели:", response)
    response = text_model.generate("Какой билд собирать на варку из геншин импакт?", max_new_tokens=100)
    print("Ответ модели:", response)
    # 2. Модель для генерации изображений (ваша Nova Anime XL)
    # image_model = AIModel(
    #     "./Models/NovaAnimeXL_ilV160/novaAnimeXL_ilV160.safetensors",  # локальный файл
    #     model_type="image",
    # )
    # img = image_model.generate(
    #     "masterpiece, best quality, girl",
    #     negative_prompt="worst quality, bad anatomy, watermark",
    #     width=512,
    #     height=512,
    #     num_inference_steps=20
    # )
    # img.save("my_bear.png")
    # print("Изображение сохранено")

    # 3. Пример с LoRA-адаптером (если у вас есть обученный адаптер)
    # text_model_with_lora = AIModel(
    #     "Qwen/Qwen2.5-7B-Instruct",
    #     lora_adapter_path="./my_lora_adapter",
    #     load_in_4bit=True
    # )