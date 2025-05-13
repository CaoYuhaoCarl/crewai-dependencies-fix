"""Crew AI based sample for A2A protocol.

Handles the agents and also presents the tools required.
"""

import base64
import logging
import os
import re

from collections.abc import AsyncIterable
from io import BytesIO
from typing import Any
from uuid import uuid4

# Simple in-memory cache replacement
class InMemoryCache:
    def __init__(self):
        self._cache = {}
    
    def get(self, key):
        return self._cache.get(key)
    
    def set(self, key, value):
        self._cache[key] = value

# Simple tool decorator replacement
def tool(name):
    def decorator(func):
        return func
    return decorator

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel


load_dotenv()

logger = logging.getLogger(__name__)


class Imagedata(BaseModel):
    """Represents image data.

    Attributes:
      id: Unique identifier for the image.
      name: Name of the image.
      mime_type: MIME type of the image.
      bytes: Base64 encoded image data.
      error: Error message if there was an issue with the image.
    """

    id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    bytes: str | None = None
    error: str | None = None


@tool('ImageGenerationTool')
def generate_image_tool(
    prompt: str, session_id: str, artifact_file_id: str = None
) -> str:
    """Image generation tool that generates images or modifies a given image based on a prompt."""
    if not prompt:
        raise ValueError('Prompt cannot be empty')

    client = genai.Client()
    cache = InMemoryCache()

    text_input = (
        prompt,
        'Ignore any input images if they do not match the request.',
    )

    ref_image = None
    logger.info(f'Session id {session_id}')
    print(f'Session id {session_id}')

    # TODO (rvelicheti) - Change convoluted memory handling logic to a better
    # version.
    # Get the image from the cache and send it back to the model.
    # Assuming the last version of the generated image is applicable.
    # Convert to PIL Image so the context sent to the LLM is not overloaded
    try:
        ref_image_data = None
        # image_id = session_cache[session_id][-1]
        session_image_data = cache.get(session_id)
        if artifact_file_id:
            try:
                ref_image_data = session_image_data[artifact_file_id]
                logger.info('Found reference image in prompt input')
            except Exception:
                ref_image_data = None
        if not ref_image_data:
            # Insertion order is maintained from python 3.7
            latest_image_key = list(session_image_data.keys())[-1]
            ref_image_data = session_image_data[latest_image_key]

        ref_bytes = base64.b64decode(ref_image_data.bytes)
        # We won't use PIL image here since it's not installed
        ref_image = BytesIO(ref_bytes)
    except Exception:
        ref_image = None

    if ref_image:
        contents = [text_input, ref_image]
    else:
        contents = text_input

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            ),
        )
    except Exception as e:
        logger.error(f'Error generating image {e}')
        print(f'Exception {e}')
        return -999999999

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            try:
                data = Imagedata(
                    bytes=base64.b64encode(part.inline_data.data).decode(
                        'utf-8'
                    ),
                    mime_type=part.inline_data.mime_type,
                    name='generated_image.png',
                    id=uuid4().hex,
                )
                session_data = cache.get(session_id)
                if session_data is None:
                    # Session doesn't exist, create it with the new item
                    cache.set(session_id, {data.id: data})
                else:
                    # Session exists, update the existing dictionary directly
                    session_data[data.id] = data

                return data.id
            except Exception as e:
                logger.error(f'Error unpacking image {e}')
                print(f'Exception {e}')
    return -999999999


class ImageGenerationAgent:
    """Agent that generates images based on user prompts."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'image/png']

    def __init__(self):
        # Simple implementation without crewai dependencies
        self.model = None
        self.image_creator_agent = None
        self.image_creation_task = None
        self.image_crew = None

    def extract_artifact_file_id(self, query):
        try:
            pattern = r'(?:id|artifact-file-id)\s+([0-9a-f]{32})'
            match = re.search(pattern, query)

            if match:
                return match.group(1)
            return None
        except Exception:
            return None

    def invoke(self, query, session_id) -> str:
        """Generate image and return the response without CrewAI."""
        artifact_file_id = self.extract_artifact_file_id(query)

        inputs = {
            'user_prompt': query,
            'session_id': session_id,
            'artifact_file_id': artifact_file_id,
        }
        logger.info(f'Inputs {inputs}')
        print(f'Inputs {inputs}')
        
        # Direct call to generate_image_tool instead of using crewai
        try:
            image_id = generate_image_tool(
                prompt=query, 
                session_id=session_id, 
                artifact_file_id=artifact_file_id
            )
            return image_id
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return "Error generating image"

    async def stream(self, query: str) -> AsyncIterable[dict[str, Any]]:
        """Streaming is not supported by CrewAI."""
        raise NotImplementedError('Streaming is not supported by CrewAI.')

    def get_image_data(self, session_id: str, image_key: str) -> Imagedata:
        """Return Imagedata given a key. This is a helper method from the agent."""
        cache = InMemoryCache()
        session_data = cache.get(session_id)
        try:
            cache.get(session_id)
            return session_data[image_key]
        except KeyError:
            logger.error('Error generating image')
            return Imagedata(error='Error generating image, please try again.')