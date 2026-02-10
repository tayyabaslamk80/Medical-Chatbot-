# rag/voice_module.py
"""
LangChain Voice Module for Medical Chatbot
Uses OpenAI Whisper API for voice transcription
"""

import os
import openai
from typing import Optional, Dict, Any
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class VoiceProcessor:
    """
    Voice processing module integrated with LangChain
    Uses OpenAI Whisper API for production-ready voice transcription
    """
    
    def __init__(self):
        """Initialize voice processor with OpenAI API"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.voice_enabled = bool(self.openai_api_key)
        
        if self.voice_enabled:
            openai.api_key = self.openai_api_key
            logger.info("✅ VoiceProcessor initialized with OpenAI Whisper")
        else:
            logger.warning("⚠️ VoiceProcessor disabled - OPENAI_API_KEY not found")
    
    def transcribe(self, audio_bytes: bytes, language: str = "hi") -> str:
        """
        Transcribe audio to text using OpenAI Whisper API
        
        Args:
            audio_bytes: Raw audio bytes
            language: Language code (hi, en, etc.)
            
        Returns:
            Transcribed text
        """
        if not self.voice_enabled:
            return ""
        
        try:
            # Create in-memory audio file
            audio_file = BytesIO(audio_bytes)
            audio_file.name = "audio.mp3"
            
            # Call OpenAI Whisper API
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="text"
            )
            
            logger.info(f"✅ Voice transcription successful: {len(transcript)} chars")
            return transcript
            
        except openai.RateLimitError:
            logger.error("❌ OpenAI rate limit exceeded")
            return ""
        except openai.APIError as e:
            logger.error(f"❌ OpenAI API error: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"❌ Voice transcription error: {str(e)}")
            return ""
    
    def process_voice_query(self, audio_bytes: bytes, language: str = "hi") -> Dict[str, Any]:
        """
        Process voice query and return structured response
        
        Args:
            audio_bytes: Raw audio bytes
            language: Language code
            
        Returns:
            Dictionary with transcription results
        """
        if not self.voice_enabled:
            return {
                "success": False,
                "error": "Voice not enabled",
                "transcribed_text": "",
                "language": language
            }
        
        # Check file size (10MB max)
        if len(audio_bytes) > 10 * 1024 * 1024:
            return {
                "success": False,
                "error": "Audio file too large (max 10MB)",
                "transcribed_text": "",
                "language": language
            }
        
        # Transcribe audio
        transcribed_text = self.transcribe(audio_bytes, language)
        
        if not transcribed_text or len(transcribed_text.strip()) < 2:
            return {
                "success": False,
                "error": "Could not transcribe audio",
                "transcribed_text": transcribed_text or "",
                "language": language
            }
        
        return {
            "success": True,
            "transcribed_text": transcribed_text,
            "language": language,
            "characters": len(transcribed_text),
            "words": len(transcribed_text.split()),
            "processor": "openai_whisper"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get voice processor status"""
        return {
            "enabled": self.voice_enabled,
            "model": "whisper-1",
            "max_file_size_mb": 10,
            "supported_languages": ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"]
        }

# Create global instance
voice_processor = VoiceProcessor()