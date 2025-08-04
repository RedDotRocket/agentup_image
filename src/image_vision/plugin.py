"""
Image Processing Plugin for AgentUp

This plugin provides comprehensive image processing capabilities including
analysis, transformation, and manipulation features using the new decorator-based system.
"""

import structlog
from agent.plugins.base import Plugin
from agent.plugins.decorators import capability
from agent.plugins.models import CapabilityContext, CapabilityResult

from .processor import ImageProcessor

logger = structlog.get_logger(__name__)


class ImageProcessingPlugin(Plugin):
    """
    AgentUp plugin for image processing capabilities using the decorator-based system.

    Provides capabilities for:
    - Image analysis and metadata extraction
    - Image transformation (resize, rotate, flip, filters)
    - Format conversion
    - Thumbnail generation
    """

    def __init__(self):
        """Initialize the image processing plugin."""
        super().__init__()
        self.processor = ImageProcessor()

    def can_handle_task(
        self, capability_id: str, context: CapabilityContext
    ) -> bool | float:
        """
        Simplified task handling - let the AI model handle everything by default.
        Only claim tasks when explicitly called with our capability IDs.
        """
        # Only handle if we're explicitly asked for a specific capability we own
        if capability_id in self._capabilities:
            return True

        # For everything else, return 0.0 so the AI model handles it
        return 0.0

    @capability(
        "analyze_image",
        name="Image Analysis and Visual Understanding",
        description="Analyze images for both technical metadata AND visual content understanding. Use this when you need to understand what's in an image or answer questions about image content.",
        scopes=["image:read"],
        ai_function=True,
        ai_parameters={
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": ["content", "technical", "both"],
                    "description": "Type of analysis: 'content' for visual understanding, 'technical' for metadata, 'both' for comprehensive analysis",
                    "default": "content",
                },
                "question": {
                    "type": "string",
                    "description": "Specific question about the image content (optional)",
                },
            },
            "required": [],
        },
        input_mode="multimodal",
        output_mode="text",
        tags=["image", "analysis", "content", "visual", "understanding", "multimodal"],
        multimodal=True,
        config_schema={
            "type": "object",
            "properties": {
                "max_image_size": {
                    "type": "integer",
                    "description": "Maximum image size in bytes",
                    "default": 10485760,
                },
                "supported_formats": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Supported image formats",
                    "default": ["JPEG", "PNG", "GIF", "BMP", "TIFF"],
                },
            },
        },
    )
    async def analyze_image(self, context: CapabilityContext) -> CapabilityResult:
        """Analyze an image for both content understanding and technical metadata."""
        try:
            image_part, user_input_or_error = self._extract_image_from_context(context)
            if image_part is None:
                return CapabilityResult(
                    content=user_input_or_error,
                    success=False,
                    error="Image extraction failed",
                )

            result = ImageProcessor.process_image(
                image_part["data"], image_part["mimeType"]
            )

            if not result["success"]:
                return CapabilityResult(
                    content=f"Error processing image: {result.get('error', 'Unknown error')}",
                    success=False,
                    error=result.get("error", "Unknown error"),
                )

            # Get parameters from AI function call
            parameters = context.metadata.get("parameters", {})
            analysis_type = parameters.get("analysis_type", "content")
            question = parameters.get("question", "")

            # Note: user_input available in user_input_or_error if needed for future enhancements

            # Build response based on analysis type
            metadata = result["metadata"]
            response_parts = []

            if analysis_type in ["content", "both"]:
                # For content analysis, we need to provide visual understanding
                # Since we can't actually "see" the image content with this implementation,
                # we'll provide what information we can extract and suggest the user
                # describe the image or use a vision-capable model
                response_parts.extend(
                    [
                        "Visual Content Analysis:",
                        "I can analyze the technical properties of this image, but for detailed visual content understanding (like identifying objects, people, text, or answering 'what is this?'), you may need to:",
                        "1. Describe the image content to me, or",
                        "2. Use a vision-capable AI model directly",
                        "",
                        "However, I can tell you about the image's technical characteristics:",
                    ]
                )

            if analysis_type in ["technical", "both"]:
                response_parts.extend(
                    [
                        "Technical Analysis Results:",
                        f"- Format: {metadata['format']}",
                        f"- Dimensions: {metadata['width']}x{metadata['height']} pixels",
                        f"- Mode: {metadata['mode']}",
                        f"- File Hash: {metadata['hash'][:8]}...",
                    ]
                )

                # Add detailed technical info
                if "mean_brightness" in metadata:
                    response_parts.append(
                        f"- Mean Brightness: {metadata['mean_brightness']:.1f}"
                    )

                if "shape" in metadata:
                    response_parts.append(f"- Shape: {metadata['shape']}")

                if "channel_means" in metadata:
                    channels = metadata["channel_means"]
                    response_parts.append(
                        f"- RGB Channel Means: R={channels[0]:.1f}, G={channels[1]:.1f}, B={channels[2]:.1f}"
                    )

            # If there was a specific question, acknowledge it
            if question:
                response_parts.insert(0, f"Regarding your question: '{question}'")
                response_parts.insert(1, "")

            return CapabilityResult(
                content="\n".join(response_parts),
                success=True,
                metadata={
                    "operation": "analyze",
                    "analysis_type": analysis_type,
                    "question": question,
                },
            )

        except Exception as e:
            logger.error(f"Error analyzing image: {e}", exc_info=True)
            return CapabilityResult(
                content=f"Error analyzing image: {str(e)}", success=False, error=str(e)
            )

    @capability(
        "transform_image",
        name="Image Transformation",
        description="Transform images with operations like resize, rotate, flip, and apply filters",
        scopes=["image:write"],
        ai_function=True,
        ai_parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["resize", "rotate", "flip", "thumbnail", "filter"],
                    "description": "Operation to perform on the image",
                },
                "target_size": {
                    "type": "string",
                    "description": "Target size for resize operations (e.g., '800x600')",
                },
                "degrees": {
                    "type": "number",
                    "description": "Degrees to rotate (for rotate operation)",
                },
                "direction": {
                    "type": "string",
                    "enum": ["horizontal", "vertical"],
                    "description": "Direction to flip (for flip operation)",
                },
                "filter_name": {
                    "type": "string",
                    "enum": [
                        "blur",
                        "sharpen",
                        "edge",
                        "emboss",
                        "enhance",
                        "brightness",
                        "contrast",
                    ],
                    "description": "Filter to apply (for filter operation)",
                },
            },
            "required": ["operation"],
        },
        input_mode="multimodal",
        output_mode="text",
        tags=["image", "transformation", "resize", "rotate", "filter", "multimodal"],
        multimodal=True,
    )
    async def transform_image(self, context: CapabilityContext) -> CapabilityResult:
        """Transform an image based on user input or AI function parameters."""
        try:
            image_part, user_input_or_error = self._extract_image_from_context(context)
            if image_part is None:
                return CapabilityResult(
                    content=user_input_or_error,
                    success=False,
                    error="Image extraction failed",
                )

            result = ImageProcessor.process_image(
                image_part["data"], image_part["mimeType"]
            )

            if not result["success"]:
                return CapabilityResult(
                    content=f"Error loading image: {result.get('error', 'Unknown error')}",
                    success=False,
                    error=result.get("error", "Unknown error"),
                )

            image = result["image"]
            original_size = image.size
            user_input_lower = user_input_or_error.lower()

            # Get parameters from AI function or parse from user input
            parameters = context.metadata.get("parameters", {})
            operation = parameters.get("operation")

            # If no operation specified, determine from user input
            if not operation:
                operation = self._determine_operation(user_input_or_error)

            # Perform transformation based on operation
            if operation == "resize":
                target_size = parameters.get("target_size")
                if target_size:
                    try:
                        width, height = map(int, target_size.split("x"))
                    except (ValueError, AttributeError):
                        width, height = 800, 600
                else:
                    # Extract size from user input
                    import re

                    size_match = re.search(r"(\d+)[x×](\d+)", user_input_or_error)
                    if size_match:
                        width, height = (
                            int(size_match.group(1)),
                            int(size_match.group(2)),
                        )
                    else:
                        width, height = 800, 600

                image = ImageProcessor.resize_image(image, (width, height))
                transform_msg = f"Resized from {original_size} to {image.size}"

            elif operation == "rotate":
                degrees = parameters.get("degrees")
                if degrees is None:
                    # Extract degrees from user input
                    import re

                    degree_match = re.search(
                        r"(\d+)\s*(?:degree|deg)", user_input_or_error
                    )
                    degrees = float(degree_match.group(1)) if degree_match else 90.0

                image = ImageProcessor.rotate_image(image, degrees)
                transform_msg = f"Rotated {degrees} degrees"

            elif operation == "flip":
                direction = parameters.get("direction")
                if not direction:
                    direction = (
                        "horizontal" if "horizontal" in user_input_lower else "vertical"
                    )
                image = ImageProcessor.flip_image(image, direction)
                transform_msg = f"Flipped {direction}ly"

            elif operation == "thumbnail":
                target_size = parameters.get("target_size")
                if target_size:
                    try:
                        width, height = map(int, target_size.split("x"))
                    except (ValueError, AttributeError):
                        width, height = 200, 200
                else:
                    # Extract size from user input or use default
                    import re

                    size_match = re.search(r"(\d+)[x×](\d+)", user_input_or_error)
                    if size_match:
                        width, height = (
                            int(size_match.group(1)),
                            int(size_match.group(2)),
                        )
                    else:
                        width, height = 200, 200

                image = ImageProcessor.create_thumbnail(image, (width, height))
                transform_msg = (
                    f"Created thumbnail from {original_size} to {image.size}"
                )

            elif operation == "filter":
                filter_name = parameters.get("filter_name")
                if not filter_name:
                    # Determine filter type from user input
                    for filter_option in [
                        "blur",
                        "sharpen",
                        "edge",
                        "emboss",
                        "enhance",
                        "brightness",
                        "contrast",
                    ]:
                        if filter_option in user_input_lower:
                            filter_name = filter_option
                            break
                    filter_name = filter_name or "enhance"  # Default filter

                image = ImageProcessor.apply_filter(image, filter_name)
                transform_msg = f"Applied {filter_name} filter"

            else:
                # Default to thumbnail
                image = ImageProcessor.create_thumbnail(image, (200, 200))
                transform_msg = (
                    f"Created default thumbnail from {original_size} to {image.size}"
                )

            # Encode result
            encoded = ImageProcessor.encode_image_base64(image, "PNG")
            return CapabilityResult(
                content=(
                    f"Image Transformation Complete:\n"
                    f"- {transform_msg}\n"
                    f"- Output format: PNG\n"
                    f"- Result encoded as base64 (length: {len(encoded)} chars)"
                ),
                success=True,
                metadata={"operation": "transform", "encoded_image": encoded},
            )

        except Exception as e:
            logger.error(f"Image transformation error: {e}")
            return CapabilityResult(
                content=f"Error during transformation: {e}", success=False, error=str(e)
            )

    @capability(
        "convert_image_format",
        name="Image Format Conversion",
        description="Convert images between different formats (PNG, JPEG, WebP, etc.)",
        scopes=["image:write"],
        ai_function=True,
        ai_parameters={
            "type": "object",
            "properties": {
                "target_format": {
                    "type": "string",
                    "enum": ["PNG", "JPEG", "WEBP", "BMP"],
                    "description": "Target format for conversion",
                },
                "quality": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Quality for JPEG conversion (1-100)",
                },
            },
            "required": ["target_format"],
        },
        input_mode="multimodal",
        output_mode="text",
        tags=["image", "conversion", "format", "multimodal"],
        multimodal=True,
    )
    async def convert_image_format(
        self, context: CapabilityContext
    ) -> CapabilityResult:
        """Convert image format."""
        try:
            image_part, user_input_or_error = self._extract_image_from_context(context)
            if image_part is None:
                return CapabilityResult(
                    content=user_input_or_error,
                    success=False,
                    error="Image extraction failed",
                )

            result = ImageProcessor.process_image(
                image_part["data"], image_part["mimeType"]
            )

            if not result["success"]:
                return CapabilityResult(
                    content=f"Error loading image: {result.get('error', 'Unknown error')}",
                    success=False,
                    error=result.get("error", "Unknown error"),
                )

            image = result["image"]

            # Get target format from parameters or determine from user input
            parameters = context.metadata.get("parameters", {})
            target_format = parameters.get("target_format")

            if not target_format:
                # Determine target format from user input
                user_input_lower = user_input_or_error.lower()
                if "png" in user_input_lower:
                    target_format = "PNG"
                elif "jpeg" in user_input_lower or "jpg" in user_input_lower:
                    target_format = "JPEG"
                elif "webp" in user_input_lower:
                    target_format = "WEBP"
                elif "bmp" in user_input_lower:
                    target_format = "BMP"
                else:
                    target_format = "PNG"  # Default

            encoded = ImageProcessor.encode_image_base64(image, target_format)
            return CapabilityResult(
                content=(
                    f"Image Format Conversion Complete:\n"
                    f"- Converted to {target_format} format\n"
                    f"- Original format: {image.format}\n"
                    f"- Result encoded as base64 (length: {len(encoded)} chars)"
                ),
                success=True,
                metadata={
                    "operation": "convert",
                    "target_format": target_format,
                    "encoded_image": encoded,
                },
            )

        except Exception as e:
            logger.error(f"Image conversion error: {e}")
            return CapabilityResult(
                content=f"Error during conversion: {e}", success=False, error=str(e)
            )

    def _extract_user_input(self, context: CapabilityContext) -> str:
        """Extract user input from the task context."""
        if hasattr(context.task, "history") and context.task.history:
            for message in context.task.history:
                if hasattr(message, "parts") and message.parts:
                    for part in message.parts:
                        if hasattr(part, "root") and part.root.kind == "text":
                            return part.root.text
        return ""

    def _determine_operation(self, user_input: str) -> str:
        """Determine the operation based on user input."""
        user_input = user_input.lower()

        if any(word in user_input for word in ["resize", "scale", "size"]):
            return "resize"
        elif any(word in user_input for word in ["rotate", "turn"]):
            return "rotate"
        elif any(word in user_input for word in ["flip", "mirror"]):
            return "flip"
        elif any(
            word in user_input for word in ["filter", "blur", "sharpen", "enhance"]
        ):
            return "filter"
        elif any(word in user_input for word in ["convert", "format", "change to"]):
            return "convert"
        elif any(word in user_input for word in ["thumbnail", "thumb"]):
            return "thumbnail"
        else:
            return "analyze"

    def _extract_image_from_context(
        self, context: CapabilityContext
    ) -> tuple[dict | None, str]:
        """Extract image data from context. Returns (image_part, user_input) or (None, error_msg)."""
        user_input = self._extract_user_input(context)

        # Extract images from the task
        if not hasattr(context.task, "history") or not context.task.history:
            return (
                None,
                "Error: No message history found. Please provide an image to process.",
            )

        image_parts = []
        for message in context.task.history:
            if hasattr(message, "parts") and message.parts:
                image_parts.extend(ImageProcessor.extract_image_parts(message.parts))

        if not image_parts:
            return (
                None,
                "Error: No images found in the message. Please upload an image to process.",
            )

        # Process the first image
        image_part = image_parts[0]
        if not image_part["data"]:
            return (
                None,
                "Error: No image data found. Images must be embedded as base64 data.",
            )

        return image_part, user_input
