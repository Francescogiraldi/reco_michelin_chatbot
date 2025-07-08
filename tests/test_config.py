"""Tests for configuration module."""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings


class TestSettings:
    """Test Settings configuration."""
    
    def test_default_settings(self):
        """Test default settings values."""
        # Mock environment to avoid requiring real API key
        with pytest.MonkeyPatch().context() as m:
            m.setenv("OPENAI_API_KEY", "sk-test123456789")
            settings = Settings()
            
            assert settings.embedding_model == "text-embedding-3-large"
            assert settings.chat_model == "gpt-4o-mini"
            assert settings.temperature == 0.2
            assert settings.top_k == 4
            assert settings.default_language == "fr"
    
    def test_environment_override(self):
        """Test environment variable override."""
        with pytest.MonkeyPatch().context() as m:
            m.setenv("OPENAI_API_KEY", "sk-test123456789")
            m.setenv("CHAT_MODEL", "gpt-4")
            m.setenv("TEMPERATURE", "0.5")
            m.setenv("TOP_K", "8")
            
            settings = Settings()
            
            assert settings.chat_model == "gpt-4"
            assert settings.temperature == 0.5
            assert settings.top_k == 8
    
    def test_invalid_api_key(self):
        """Test validation of invalid API key."""
        with pytest.MonkeyPatch().context() as m:
            m.setenv("OPENAI_API_KEY", "invalid-key")
            
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_temperature(self):
        """Test validation of invalid temperature."""
        with pytest.MonkeyPatch().context() as m:
            m.setenv("OPENAI_API_KEY", "sk-test123456789")
            m.setenv("TEMPERATURE", "3.0")  # Too high
            
            with pytest.raises(ValidationError):
                Settings()
    
    def test_invalid_top_k(self):
        """Test validation of invalid top_k."""
        with pytest.MonkeyPatch().context() as m:
            m.setenv("OPENAI_API_KEY", "sk-test123456789")
            m.setenv("TOP_K", "0")  # Too low
            
            with pytest.raises(ValidationError):
                Settings()
    
    def test_path_settings(self):
        """Test path-related settings."""
        with pytest.MonkeyPatch().context() as m:
            m.setenv("OPENAI_API_KEY", "sk-test123456789")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                m.setenv("CATALOG_PATH", str(temp_path / "test_catalog.csv"))
                m.setenv("INDEX_DIR", str(temp_path / "test_index"))
                
                settings = Settings()
                
                assert settings.catalog_path == temp_path / "test_catalog.csv"
                assert settings.index_dir == temp_path / "test_index"
    
    def test_language_settings(self):
        """Test language-related settings."""
        with pytest.MonkeyPatch().context() as m:
            m.setenv("OPENAI_API_KEY", "sk-test123456789")
            m.setenv("DEFAULT_LANGUAGE", "en")
            
            settings = Settings()
            
            assert settings.default_language == "en"
            assert "en" in settings.supported_languages
            assert "fr" in settings.supported_languages
    
    def test_security_settings(self):
        """Test security-related settings."""
        with pytest.MonkeyPatch().context() as m:
            m.setenv("OPENAI_API_KEY", "sk-test123456789")
            m.setenv("ENABLE_RATE_LIMITING", "false")
            m.setenv("MAX_REQUESTS_PER_MINUTE", "60")
            
            settings = Settings()
            
            assert settings.enable_rate_limiting is False
            assert settings.max_requests_per_minute == 60
    
    def test_logging_settings(self):
        """Test logging-related settings."""
        with pytest.MonkeyPatch().context() as m:
            m.setenv("OPENAI_API_KEY", "sk-test123456789")
            m.setenv("LOG_LEVEL", "DEBUG")
            
            settings = Settings()
            
            assert settings.log_level == "DEBUG"
            assert "{time:" in settings.log_format


class TestGetSettings:
    """Test get_settings function."""
    
    def test_get_settings_returns_same_instance(self):
        """Test that get_settings returns the same instance."""
        with pytest.MonkeyPatch().context() as m:
            m.setenv("OPENAI_API_KEY", "sk-test123456789")
            
            settings1 = get_settings()
            settings2 = get_settings()
            
            assert settings1 is settings2
    
    def test_get_settings_with_env_file(self):
        """Test get_settings with .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OPENAI_API_KEY=sk-test123456789\n"
                "CHAT_MODEL=gpt-3.5-turbo\n"
                "TEMPERATURE=0.3\n"
            )
            
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Import here to pick up the .env file
                from src.config import Settings
                settings = Settings()
                
                assert settings.openai_api_key == "sk-test123456789"
                assert settings.chat_model == "gpt-3.5-turbo"
                assert settings.temperature == 0.3
                
            finally:
                os.chdir(original_cwd)