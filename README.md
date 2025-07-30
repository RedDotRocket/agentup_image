# AgentUp Image

<p align="center">
  <img src="static/images_logo.png" alt="AgentUp Image Plugin" width="400"/>
</p>

A comprehensive image processing and analysis plugin for the AgentUp framework
that provides advanced image manipulation, transformation, and analysis
capabilities.

## Requirements

You will need to use a model that supports multimodal requests, such as `gpt-4-turbo` or `gpt-4o-mini`.

This works with local models, but I only tested llava:7b so far, running on Ollama.

### Required Scopes

This plugin requires specific permission scopes to be granted in your agent configuration:

- **`image:read`** - Required for image analysis capabilities
- **`image:write`** - Required for image transformation and format conversion capabilities

Make sure your agent is configured with the necessary scopes to use these features.

## Features

- **Image Analysis**: Extract metadata, dimensions, color information, and visual characteristics
- **Image Transformation**: Resize, rotate, flip, and apply filters to images
- **Format Conversion**: Convert between PNG, JPEG, WebP, BMP, and other formats
- **Thumbnail Generation**: Create optimized thumbnails with custom sizes
- **Filter Application**: Apply blur, sharpen, edge detection, emboss, and enhancement filters
- **AI-Powered Routing**: Intelligent task routing based on content analysis and keywords

## Installation

### From PyPI (Recommended)

```bash
pip install image-vision
```

### From Source

```bash
git clone https://github.com/RedDotRocket/image-vision.git
cd image-vision
pip install -e .
```

## Quick Start

### 1. Install the Plugin

```bash
pip install image-vision
```

### 2. Configure Your Agent

Add the plugin to your agent's configuration in `agent_config.yaml`:

```yaml
plugins:
  - plugin_id: image-vision
    name: Image Processing
    description: Process and analyze images
    tags: [image, processing, analysis]
    input_mode: multimodal
    output_mode: text
    priority: 85
```

### 3. Use in Your Agent

The plugin automatically handles image-related requests when images are uploaded or when users ask for image processing tasks.

## Usage Examples

## Image Recognition

You will find a script in the `examples` directory that demonstrates how to use the plugin for multi-modal requests.

```bash
python examples/test_multimodal.py examples/guess.jpeg "what vehicle model is this?"
Testing connectivity with text-only request first...

Testing text-only request...
✓ Text-only request successful

==================================================

✓ Encoded image: examples/guess.jpeg (17216 base64 chars)
→ Sending request to http://localhost:8000
  Prompt: what vehicle model is this?
  Image: image/jpeg

✓ Response received:
--------------------------------------------------
The vehicle model shown in the image is a Land Rover Defender. This model is known for its rugged design and off-road capabilities.
--------------------------------------------------
```

**Remember to replace the API key in the script with your own.**

### Image Analysis

Upload an image and ask:
- "Analyze this image"
- "What are the dimensions of this image?"
- "Get detailed color analysis of this photo"

**Response:**
```
Image Analysis Results:
- Format: JPEG
- Dimensions: 1920x1080 pixels
- Mode: RGB
- File Hash: a1b2c3d4...
- Mean Brightness: 128.5
- RGB Channel Means: R=142.3, G=128.7, B=115.2
```

### Image Transformation

- "Resize this image to 800x600"
- "Create a thumbnail"
- "Rotate the image 90 degrees"
- "Flip the image horizontally"
- "Apply a blur filter"

**Response:**
```
Image Transformation Complete:
- Resized from (1920, 1080) to (800, 600)
- Output format: PNG
- Result encoded as base64 (length: 45231 chars)
```

### Format Conversion

- "Convert this image to JPEG"
- "Change the format to PNG"
- "Convert to WebP format"

**Response:**
```
Image Format Conversion Complete:
- Converted to JPEG format
- Original format: PNG
- Result encoded as base64 (length: 32415 chars)
```

## Example script

```bash
python examples/test_multimodal.py examples/guess.jpeg "what vehicle model is this?"
Testing connectivity with text-only request first...

Testing text-only request...
✓ Text-only request successful

==================================================

✓ Encoded image: examples/guess.jpeg (17216 base64 chars)
→ Sending request to http://localhost:8000
  Prompt: what vehicle model is this?
  Image: image/jpeg

✓ Response received:
--------------------------------------------------
The vehicle model shown in the image is a Land Rover Defender. This model is known for its rugged design and off-road capabilities.
--------------------------------------------------
```

## AI Function Integration

The plugin provides AI-callable functions for intelligent routing:

### `analyze_image`
Analyzes uploaded images and returns detailed insights.

**Parameters:**
- `analysis_type` (optional): "basic", "detailed", or "color"

### `transform_image`
Transforms images with various operations.

**Parameters:**
- `operation`: "resize", "rotate", "flip", "thumbnail", or "filter"
- `target_size` (optional): Target size for resize operations (e.g., "800x600")
- `degrees` (optional): Degrees to rotate
- `direction` (optional): "horizontal" or "vertical" for flip operations
- `filter_name` (optional): Filter type for filter operations

### `convert_image_format`
Converts images between different formats.

**Parameters:**
- `target_format`: "PNG", "JPEG", "WEBP", or "BMP"
- `quality` (optional): Quality for JPEG conversion (1-100)

## Configuration

### Plugin Configuration

```yaml
plugins:
  - plugin_id: image-vision
    name: Image Processing
    description: Process and analyze images
    tags: [image, processing, analysis]
    input_mode: multimodal
    output_mode: text
    priority: 85

services:
  image_vision:
    type: plugin
    enabled: true
    config:
      # Maximum image size in MB
      max_image_size_mb: 10

      # Supported image formats
      supported_formats:
        - "image/png"
        - "image/jpeg"
        - "image/webp"
        - "image/gif"
        - "image/bmp"

      # Default thumbnail size [width, height]
      default_thumbnail_size: [200, 200]
```

### Required Scopes by Capability

Each image processing capability requires specific permission scopes:

| Capability | Required Scope | Description |
|------------|----------------|-------------|
| `analyze_image` | `image:read` | Analyze uploaded images and extract metadata |
| `transform_image` | `image:write` | Transform images (resize, rotate, flip, filters) |
| `convert_image_format` | `image:write` | Convert images between different formats |

Configure your agent with the appropriate scopes in `agent_config.yaml`:

```yaml
security:
  scopes:
    - image:read      # For image analysis
    - image:write     # For image transformation and conversion
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_image_size_mb` | number | 10 | Maximum image size in MB |
| `supported_formats` | array | All formats | List of supported MIME types |
| `default_thumbnail_size` | array | [200, 200] | Default thumbnail dimensions |

## Supported Image Formats

- **PNG** (`image/png`) - Lossless compression with transparency
- **JPEG** (`image/jpeg`) - Lossy compression, good for photos
- **WebP** (`image/webp`) - Modern format with better compression
- **GIF** (`image/gif`) - Animated images and simple graphics
- **BMP** (`image/bmp`) - Uncompressed bitmap format

## Available Operations

### Analysis Operations
- Basic metadata extraction (dimensions, format, mode)
- Detailed analysis (brightness, color channels)
- Color analysis (RGB channel means)
- File hash generation for deduplication

### Transformation Operations
- **Resize**: Change image dimensions while maintaining aspect ratio
- **Rotate**: Rotate images by specified degrees
- **Flip**: Mirror images horizontally or vertically
- **Thumbnail**: Create optimized thumbnails
- **Filters**: Apply various visual filters

### Available Filters
- **Blur**: Soften image details
- **Sharpen**: Enhance image clarity
- **Edge**: Detect and highlight edges
- **Emboss**: Create embossed effect
- **Enhance**: Improve overall image quality
- **Brightness**: Adjust image brightness
- **Contrast**: Adjust image contrast

## Development

### Setting Up Development Environment

```bash
git clone https://github.com/agentup-ai/image-vision.git
cd image-vision

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Run linting
ruff check src/

# Format code
black src/
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=image_vision

# Run specific test file
pytest tests/test_processor.py
```

### Project Structure

```
image-vision/
├── src/
│   └── image_vision/
│       ├── __init__.py
│       ├── plugin.py          # Main plugin implementation
│       └── processor.py       # Image processing utilities
├── tests/
│   ├── __init__.py
│   ├── test_plugin.py         # Plugin tests
│   └── test_processor.py      # Processor tests
├── pyproject.toml             # Package configuration
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## Error Handling

The plugin includes comprehensive error handling:

- **Invalid image data**: Returns descriptive error messages
- **Unsupported formats**: Validates format support before processing
- **Size limits**: Enforces configurable file size limits
- **Processing errors**: Graceful handling of PIL/image processing errors

## Performance Considerations

- **Memory Usage**: Large images are processed efficiently using PIL
- **File Size Limits**: Configurable limits prevent memory issues
- **Caching**: Results can be cached using AgentUp's middleware system
- **Format Optimization**: Automatic format optimization for better performance

## Integration with AgentUp

### Middleware Support

The plugin works seamlessly with AgentUp middleware:

```yaml
middleware:
  - name: cached
    params:
      ttl: 300  # Cache results for 5 minutes
  - name: rate_limited
    params:
      requests_per_minute: 60
  - name: timed
    params: {}
```

### State Management

The plugin supports stateful operations for conversation context:

```yaml
state_management:
  enabled: true
  backend: valkey
  ttl: 3600
```

### Security

- Input validation for all image data
- File size limits to prevent DoS attacks
- Format validation before processing
- Secure base64 encoding/decoding

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Guidelines

1. Follow the existing code style
2. Add tests for new features
3. Update documentation as needed
4. Ensure all tests pass before submitting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://docs.agentup.dev/plugins/image-processing](https://docs.agentup.dev/plugins/image-processing)
- **Issues**: [GitHub Issues](https://github.com/RedDotRocket/image-vision/issues)
- **Discussions**: [GitHub Discussions](https://github.com/RedDotRocket/image-visiong/discussions)

## Changelog

### v1.0.0
- Initial release
- Complete image processing pipeline
- AI function integration
- Comprehensive test suite
- Full documentation

## Related Projects

- [AgentUp Framework](https://github.com/RedDoctRocket/agentup) - The main AgentUp framework
- [AgentUp Document Processing](https://github.com/RedDotRocket/image-vision) - Document processing plugin
- [A2A SDK](https://github.com/a2a-ai/a2a-sdk) - A2A protocol implementation

---

**Made with ❤️ by the AgentUp Team**