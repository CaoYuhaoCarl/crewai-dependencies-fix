# CrewAI Dependencies Fix

This repository provides fixed implementations for the CrewAI image generation agent to work without requiring hard-to-install dependencies.

## The Problem

The original implementation had dependencies on:

1. PIL (Pillow) for image processing
2. crewai for agent orchestration
3. Common utilities like in-memory cache

These dependencies often cause installation issues due to version conflicts or complex build requirements.

## The Solution

This repository provides simplified alternatives:

1. A lightweight in-memory cache implementation
2. A simplified tool decorator
3. Direct function calls instead of relying on crewai orchestration
4. Using BytesIO instead of PIL for image handling

## Installation

```bash
# Clone this repository
git clone https://github.com/CaoYuhaoCarl/crewai-dependencies-fix.git
cd crewai-dependencies-fix

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from agent import ImageGenerationAgent

# Create an agent instance
agent = ImageGenerationAgent()

# Generate an image
session_id = "your-session-id"
query = "Generate an image of a mountain landscape"
result = agent.invoke(query, session_id)

print(f"Result: {result}")
```

## Test

Run the test script to check if everything is working:

```bash
python test_agent.py
```

## Environment Variables

Create a `.env` file with your Google API key:

```
GOOGLE_API_KEY=your_api_key_here
```

## License

MIT