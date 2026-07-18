"""
core/language_config.py
Central config: which languages use Sarvam vs Whisper/edge-tts.
"""

# Indian languages -> Sarvam (STT: saaras:v3, TTS: bulbul:v3)
INDIAN_LANGUAGES = {
    "Hindi":      {"code": "hi-IN", "speaker": "meera"},
    "Tamil":      {"code": "ta-IN", "speaker": "meera"},
    "Telugu":     {"code": "te-IN", "speaker": "meera"},
    "Bengali":    {"code": "bn-IN", "speaker": "meera"},
    "Marathi":    {"code": "mr-IN", "speaker": "meera"},
    "Gujarati":   {"code": "gu-IN", "speaker": "meera"},
    "Punjabi":    {"code": "pa-IN", "speaker": "meera"},
    "Kannada":    {"code": "kn-IN", "speaker": "meera"},
    "Malayalam":  {"code": "ml-IN", "speaker": "meera"},
    "Odia":       {"code": "od-IN", "speaker": "meera"},
}

# Foreign languages -> use Whisper + edge-tts
FOREIGN_LANGUAGES = {
    "Spanish":    "es-ES",
    "French":     "fr-FR",
    "German":     "de-DE",
    "Japanese":   "ja-JP",
    "Chinese":    "zh-CN",
    "Arabic":     "ar-SA",
    "Portuguese": "pt-BR",
    "Russian":    "ru-RU",
}

ALL_LANGUAGES = {**INDIAN_LANGUAGES, **FOREIGN_LANGUAGES}


def is_indian_language(language_name: str) -> bool:
    return language_name in INDIAN_LANGUAGES


def get_sarvam_code(language_name: str) -> str:
    return INDIAN_LANGUAGES.get(language_name, {}).get("code", "hi-IN")