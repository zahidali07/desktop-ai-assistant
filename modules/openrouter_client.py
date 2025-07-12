#!/usr/bin/env python3
"""
OpenRouter API Client for Desktop AI Assistant

This module provides integration with OpenRouter API to access various AI models
including DeepSeek, Gemini, GPT, Claude, and other language models.

Features:
- Multiple AI model support
- Chat completion API
- Streaming responses
- Error handling and retries
- Token usage tracking
- Model selection and switching

Author: Desktop AI Assistant Project
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

try:
    import aiohttp
    import requests
except ImportError as e:
    logging.error(f"Required HTTP libraries not installed: {e}")
    logging.error("Please install: pip install aiohttp requests")

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported AI model providers"""
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    META = "meta"
    MISTRAL = "mistral"
    COHERE = "cohere"


@dataclass
class ModelInfo:
    """Information about an AI model"""
    id: str
    name: str
    provider: ModelProvider
    context_length: int
    pricing: Dict[str, float]
    description: str = ""


@dataclass
class ChatMessage:
    """Chat message structure"""
    role: str  # 'system', 'user', 'assistant'
    content: str
    name: Optional[str] = None


@dataclass
class ChatResponse:
    """Chat completion response"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    response_time: float


class OpenRouterClient:
    """
    OpenRouter API client for accessing multiple AI models
    """
    
    # Popular models available through OpenRouter
    POPULAR_MODELS = {
        # DeepSeek models
        "deepseek/deepseek-chat": ModelInfo(
            id="deepseek/deepseek-chat",
            name="DeepSeek Chat",
            provider=ModelProvider.DEEPSEEK,
            context_length=32768,
            pricing={"prompt": 0.14, "completion": 0.28}
        ),
        "deepseek/deepseek-coder": ModelInfo(
            id="deepseek/deepseek-coder",
            name="DeepSeek Coder",
            provider=ModelProvider.DEEPSEEK,
            context_length=16384,
            pricing={"prompt": 0.14, "completion": 0.28}
        ),
        
        # Google models
        "google/gemini-pro": ModelInfo(
            id="google/gemini-pro",
            name="Gemini Pro",
            provider=ModelProvider.GOOGLE,
            context_length=32768,
            pricing={"prompt": 0.5, "completion": 1.5}
        ),
        "google/gemini-pro-vision": ModelInfo(
            id="google/gemini-pro-vision",
            name="Gemini Pro Vision",
            provider=ModelProvider.GOOGLE,
            context_length=16384,
            pricing={"prompt": 0.5, "completion": 1.5}
        ),
        
        # OpenAI models
        "openai/gpt-4-turbo": ModelInfo(
            id="openai/gpt-4-turbo",
            name="GPT-4 Turbo",
            provider=ModelProvider.OPENAI,
            context_length=128000,
            pricing={"prompt": 10.0, "completion": 30.0}
        ),
        "openai/gpt-3.5-turbo": ModelInfo(
            id="openai/gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider=ModelProvider.OPENAI,
            context_length=16384,
            pricing={"prompt": 0.5, "completion": 1.5}
        ),
        
        # Anthropic models
        "anthropic/claude-3-opus": ModelInfo(
            id="anthropic/claude-3-opus",
            name="Claude 3 Opus",
            provider=ModelProvider.ANTHROPIC,
            context_length=200000,
            pricing={"prompt": 15.0, "completion": 75.0}
        ),
        "anthropic/claude-3-sonnet": ModelInfo(
            id="anthropic/claude-3-sonnet",
            name="Claude 3 Sonnet",
            provider=ModelProvider.ANTHROPIC,
            context_length=200000,
            pricing={"prompt": 3.0, "completion": 15.0}
        ),
    }
    
    def __init__(self, config):
        self.config = config
        self.api_key = getattr(config, 'OPENROUTER_API_KEY', None)
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_model = getattr(config, 'DEFAULT_MODEL', "deepseek/deepseek-chat")
        self.current_model = self.default_model
        self.session = None
        self.conversation_history = []
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found in config. Please set OPENROUTER_API_KEY.")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for OpenRouter API"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/desktop-ai-assistant",
            "X-Title": "Desktop AI Assistant"
        }
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(
                f"{self.base_url}/models",
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('data', [])
                    logger.info(f"Retrieved {len(models)} available models")
                    return models
                else:
                    logger.error(f"Failed to get models: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def set_model(self, model_id: str):
        """Set the current model to use"""
        if model_id in self.POPULAR_MODELS:
            self.current_model = model_id
            logger.info(f"Model set to: {self.POPULAR_MODELS[model_id].name}")
        else:
            # Allow setting any model ID, even if not in our predefined list
            self.current_model = model_id
            logger.info(f"Model set to: {model_id}")
    
    def get_current_model_info(self) -> Optional[ModelInfo]:
        """Get information about the current model"""
        return self.POPULAR_MODELS.get(self.current_model)
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> ChatResponse:
        """Send a chat completion request"""
        start_time = time.time()
        
        try:
            if not self.api_key:
                raise ValueError("OpenRouter API key not configured")
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Use provided model or current model
            model_to_use = model or self.current_model
            
            # Convert ChatMessage objects to dict format
            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Prepare request payload
            payload = {
                "model": model_to_use,
                "messages": formatted_messages,
                "temperature": temperature,
                "stream": stream
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            logger.debug(f"Sending chat completion request to {model_to_use}")
            
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract response data
                    choice = data['choices'][0]
                    content = choice['message']['content']
                    finish_reason = choice.get('finish_reason', 'stop')
                    usage = data.get('usage', {})
                    
                    response_time = time.time() - start_time
                    
                    # Update usage tracking
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
                    
                    self.total_tokens_used += total_tokens
                    
                    # Calculate cost if model info is available
                    model_info = self.POPULAR_MODELS.get(model_to_use)
                    if model_info:
                        cost = (
                            (prompt_tokens / 1000) * model_info.pricing['prompt'] +
                            (completion_tokens / 1000) * model_info.pricing['completion']
                        ) / 1000  # Convert from per-million to per-token
                        self.total_cost += cost
                    
                    logger.info(f"Chat completion successful. Tokens: {total_tokens}, Time: {response_time:.2f}s")
                    
                    return ChatResponse(
                        content=content,
                        model=model_to_use,
                        usage=usage,
                        finish_reason=finish_reason,
                        response_time=response_time
                    )
                
                else:
                    error_text = await response.text()
                    logger.error(f"Chat completion failed: {response.status} - {error_text}")
                    raise Exception(f"API request failed: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
    
    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion response"""
        try:
            if not self.api_key:
                raise ValueError("OpenRouter API key not configured")
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            model_to_use = model or self.current_model
            
            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            payload = {
                "model": model_to_use,
                "messages": formatted_messages,
                "temperature": temperature,
                "stream": True
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload
            ) as response:
                
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix
                            
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await response.text()
                    logger.error(f"Streaming failed: {response.status} - {error_text}")
                    raise Exception(f"Streaming request failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {e}")
            raise
    
    async def simple_chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Simple chat interface for single messages"""
        messages = []
        
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        
        messages.append(ChatMessage(role="user", content=user_message))
        
        response = await self.chat_completion(messages)
        return response.content
    
    def add_to_conversation(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation_history.append(ChatMessage(role=role, content=content))
        
        # Keep conversation history manageable (last 20 messages)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    async def continue_conversation(self, user_message: str) -> str:
        """Continue an ongoing conversation"""
        # Add user message to history
        self.add_to_conversation("user", user_message)
        
        # Get response
        response = await self.chat_completion(self.conversation_history)
        
        # Add assistant response to history
        self.add_to_conversation("assistant", response.content)
        
        return response.content
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "current_model": self.current_model,
            "conversation_length": len(self.conversation_history)
        }
    
    async def test_connection(self) -> bool:
        """Test the connection to OpenRouter API"""
        try:
            models = await self.get_available_models()
            if models:
                logger.info("OpenRouter connection test successful")
                return True
            else:
                logger.error("OpenRouter connection test failed - no models returned")
                return False
        except Exception as e:
            logger.error(f"OpenRouter connection test failed: {e}")
            return False
    
    def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            asyncio.create_task(self.session.close())


# Utility functions
def create_system_prompt(assistant_name: str = "Desktop Assistant") -> str:
    """Create a system prompt for the desktop assistant"""
    return f"""You are {assistant_name}, a helpful desktop AI assistant that can:
    
    1. Control system functions (open apps, manage files, system settings)
    2. Answer questions and provide information
    3. Help with productivity tasks
    4. Provide contextual assistance based on user activity
    5. Execute voice commands for desktop automation
    
    You should be concise, helpful, and proactive. When users ask you to perform system actions,
    acknowledge the request and indicate what action you're taking.
    
    Always respond in a natural, conversational tone as if you're a helpful assistant.
    """


def create_context_prompt(context_data: Dict[str, Any]) -> str:
    """Create a context-aware prompt based on current system state"""
    context_parts = []
    
    if context_data.get('current_app'):
        context_parts.append(f"Current application: {context_data['current_app']}")
    
    if context_data.get('time_of_day'):
        context_parts.append(f"Time: {context_data['time_of_day']}")
    
    if context_data.get('battery_level'):
        context_parts.append(f"Battery: {context_data['battery_level']}%")
    
    if context_data.get('location'):
        context_parts.append(f"Location: {context_data['location']}")
    
    if context_parts:
        return f"Current context: {', '.join(context_parts)}"
    
    return ""


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from config.settings import Config
    
    async def test_openrouter_client():
        """Test the OpenRouter client"""
        config = Config()
        
        async with OpenRouterClient(config) as client:
            # Test connection
            if await client.test_connection():
                print("✓ OpenRouter connection successful")
            else:
                print("✗ OpenRouter connection failed")
                return
            
            # Test simple chat
            try:
                response = await client.simple_chat(
                    "Hello! Can you help me with desktop automation?",
                    create_system_prompt()
                )
                print(f"Assistant: {response}")
                
                # Test conversation
                response2 = await client.continue_conversation(
                    "What can you do to help me be more productive?"
                )
                print(f"Assistant: {response2}")
                
                # Show usage stats
                stats = client.get_usage_stats()
                print(f"Usage stats: {stats}")
                
            except Exception as e:
                print(f"Error testing chat: {e}")
    
    # Run the test
    asyncio.run(test_openrouter_client())
