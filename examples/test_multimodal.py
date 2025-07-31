#!/usr/bin/env python3
"""
Test multi-modal capabilities of AgentUp agent.

Usage:
    python test_multimodal.py <image_file> [prompt]

Example:
    python test_multimodal.py cat.jpg "What animal is in this image?"
"""

import requests
import base64
import json
import sys
from pathlib import Path
import os

# Enable debug mode with environment variable
DEBUG = os.getenv('DEBUG', '').lower() in ('1', 'true', 'yes')

def resolve_image_path(image_path):
    p = Path(image_path)
    if p.exists():
        return p

    # fall back to “same folder as script”
    script_dir = Path(__file__).resolve().parent
    p2 = script_dir / image_path
    if p2.exists():
        return p2

    return p  # original, so your existing “not found” message still applies

def test_multimodal(image_path, prompt="What do you see in this image? Please describe it in detail."):
    """Test multi-modal agent with an image."""

    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found")
        return

    # Read and encode image
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        print(f"✓ Encoded image: {image_path} ({len(image_data)} base64 chars)")
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    # Determine MIME type
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/jpeg')

    # Prepare request - A2A compliant format
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": "test-multimodal-1",
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": prompt
                    },
                    {
                        "kind": "file",
                        "file": {
                            "name": Path(image_path).name,
                            "mimeType": mime_type,
                            "bytes": image_data
                        }
                    }
                ]
            }
        },
        "id": "test-multimodal-1"
    }

    # Your API key from agent_config.yaml
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "admin-key-123"
    }

    # Send request
    print("→ Sending request to http://localhost:8000")
    print(f"  Prompt: {prompt}")
    print(f"  Image: {mime_type}")

    if DEBUG:
        print("\nDEBUG: Request payload:")
        print(json.dumps(payload, indent=2)[:500] + "..." if len(json.dumps(payload)) > 500 else json.dumps(payload, indent=2))

    try:
        response = requests.post(
            "http://localhost:8000",
            json=payload,
            headers=headers,
            timeout=30  # Multi-modal can take time
        )

        # Check response
        if response.status_code == 200:
            result = response.json()

            if DEBUG:
                print("\nDEBUG: Raw response:")
                print(json.dumps(result, indent=2))

            print("\n✓ Response received:")
            print("-" * 50)

            if "result" in result:
                # Extract the response text from A2A format
                if isinstance(result["result"], dict) and "artifacts" in result["result"]:
                    for artifact in result["result"]["artifacts"]:
                        for part in artifact.get("parts", []):
                            if part.get("kind") == "text":
                                print(part.get("text", "No response text"))
                else:
                    print(json.dumps(result["result"], indent=2))
            elif "error" in result:
                error = result['error']
                print(f"Error: {error.get('message', 'Unknown error')}")
                if 'code' in error:
                    print(f"Code: {error['code']}")
                if 'data' in error:
                    print(f"Details: {error['data']}")
            else:
                print(json.dumps(result, indent=2))

            print("-" * 50)
        else:
            print(f"\n✗ Error: HTTP {response.status_code}")
            print(response.text)

    except requests.exceptions.Timeout:
        print("\n✗ Error: Request timed out (30s)")
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to agent at http://localhost:8000")
        print("  Make sure your agent is running: agentup agent serve")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def test_text_only(prompt="Hello, are you working?"):
    """Test basic text-only functionality first."""
    print("Testing text-only request...")

    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": "test-text-1",
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": prompt
                    }
                ]
            }
        },
        "id": "test-text-1"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer Xdha8o8Rnxihs_KZ9FVn8egc2qtAgOdz"
    }

    try:
        response = requests.post(
            "http://localhost:8000",
            json=payload,
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            if DEBUG:
                print("\nDEBUG: Text-only response:")
                print(json.dumps(result, indent=2))
            if "error" in result:
                print(f"Error: {result['error'].get('message', 'Unknown')}")
                return False
            else:
                print("✓ Text-only request successful")
                return True
        else:
            print(f"✗ HTTP {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_multimodal.py <image_file> [prompt]")
        print("       python test_multimodal.py --test-text")
        print("Example: python test_multimodal.py cat.jpg 'What animal is this?'")
        sys.exit(1)

    if sys.argv[1] == "--test-text":
        # Test text-only first
        test_text_only()
        sys.exit(0)

    # Resolve the image path (cwd or script directory)
    image_arg = sys.argv[1]
    image_path = resolve_image_path(image_arg)
    if not image_path.exists():
        print(f"Error: Image file '{image_arg}' not found")
        sys.exit(1)

    # Optional prompt
    prompt = sys.argv[2] if len(sys.argv) > 2 else (
        "What do you see in this image? Please describe it in detail."
    )

    # First verify connectivity via text-only
    print("Testing connectivity with text-only request first...\n")
    if test_text_only("echo test"):
        print("\n" + "=" * 50 + "\n")

    # Then test multi-modal
    test_multimodal(str(image_path), prompt)
