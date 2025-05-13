#!/usr/bin/env python3
"""Test script for the agent."""

import logging
import uuid

from agent import ImageGenerationAgent

logging.basicConfig(level=logging.INFO)

def main():
    """Run a simple test with the agent."""
    agent = ImageGenerationAgent()
    session_id = str(uuid.uuid4())
    
    print(f"Testing with session_id: {session_id}")
    
    query = "Test query for the agent"
    try:
        response = agent.invoke(query, session_id)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()