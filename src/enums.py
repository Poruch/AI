from enum import Enum

class AiModelType(Enum):
    QWEN_7B = "qwen2.5:7b"
    LLAMA_8B = "llama3.1:8b"
    MISTRAL_7B = "mistral:7b"

class VoiceCommand(Enum):
    START = "start"
    STOP = "stop"
    GENERATE_IMAGE = "generate_image"
    CHANGE_MOTION = "change_motion"

class MotionName(Enum):
    IDLE = "Idle"
    TALKING = "Talking"
    WAVE = "Wave"
    THINKING = "Thinking"