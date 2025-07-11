"""
Image Processing Plugin for AgentUp

This plugin provides comprehensive image processing capabilities including
analysis, transformation, and manipulation features.
"""

# Import plugin system components
import pluggy
import structlog
from a2a.types import Task
from agent.plugins import (
    AIFunction,
    CapabilityType,
    CapabilityContext,
    CapabilityInfo,
    CapabilityResult,
    ValidationResult,
)

from .processor import ImageProcessor

logger = structlog.get_logger(__name__)

# Hook implementation marker
hookimpl = pluggy.HookimplMarker("agentup")


class ImageProcessingPlugin:
    """
    AgentUp plugin for image processing capabilities.

    Provides capabilities for:
    - Image analysis and metadata extraction
    - Image transformation (resize, rotate, flip, filters)
    - Format conversion
    - Thumbnail generation
    """

    def __init__(self):
        """Initialize the image processing plugin."""
        self.processor = ImageProcessor()
        self.config = {}

    @hookimpl
    def register_capability(self) -> list[CapabilityInfo]:
        """Register the image processing capabilities."""
        base_config_schema = {
            "type": "object",
            "properties": {
                "max_image_size": {
                    "type": "integer",
                    "description": "Maximum image size in bytes",
                    "default": 10485760
                },
                "supported_formats": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Supported image formats",
                    "default": ["JPEG", "PNG", "GIF", "BMP", "TIFF"]
                },
                "quality_default": {
                    "type": "integer",
                    "description": "Default image quality for compression",
                    "default": 85,
                    "minimum": 1,
                    "maximum": 100
                }
            }
        }
        
        return [
            CapabilityInfo(
                id="analyze_image",
                name="Image Analysis",
                version="1.0.0",
                description="Analyze images and extract metadata, dimensions, and visual characteristics",
                plugin_name="agentup_image",
                capabilities=[CapabilityType.AI_FUNCTION, CapabilityType.MULTIMODAL],
                input_mode="multimodal",
                output_mode="text",
                tags=["image", "analysis", "metadata", "multimodal"],
                config_schema=base_config_schema
            ),
            CapabilityInfo(
                id="transform_image",
                name="Image Transformation", 
                version="1.0.0",
                description="Transform images with operations like resize, rotate, flip, and apply filters",
                plugin_name="agentup_image",
                capabilities=[CapabilityType.AI_FUNCTION, CapabilityType.MULTIMODAL],
                input_mode="multimodal",
                output_mode="text",
                tags=["image", "transformation", "resize", "rotate", "filter", "multimodal"],
                config_schema=base_config_schema
            ),
            CapabilityInfo(
                id="convert_image_format",
                name="Image Format Conversion",
                version="1.0.0", 
                description="Convert images between different formats (PNG, JPEG, WebP, etc.)",
                plugin_name="agentup_image",
                capabilities=[CapabilityType.AI_FUNCTION, CapabilityType.MULTIMODAL],
                input_mode="multimodal",
                output_mode="text",
                tags=["image", "conversion", "format", "multimodal"],
                config_schema=base_config_schema
            ),
        ]

    @hookimpl
    def can_handle_task(self, context: CapabilityContext) -> float:
        """Check if this plugin can handle the given task."""
        try:
            # Extract user input and check for image-related keywords
            user_input = self._extract_user_input(context).lower()

            # Check if task contains images
            has_images = False
            if hasattr(context.task, "history") and context.task.history:
                for message in context.task.history:
                    if hasattr(message, "parts") and message.parts:
                        image_parts = ImageProcessor.extract_image_parts(message.parts)
                        if image_parts:
                            has_images = True
                            break

            # Image-related keywords and their confidence scores
            image_keywords = {
                "analyze image": 1.0,
                "process image": 1.0,
                "transform image": 1.0,
                "image analysis": 1.0,
                "resize image": 0.9,
                "rotate image": 0.9,
                "flip image": 0.9,
                "image filter": 0.9,
                "convert image": 0.9,
                "thumbnail": 0.8,
                "image": 0.7,
                "photo": 0.6,
                "picture": 0.6,
            }

            confidence = 0.0

            # Boost confidence if images are present
            if has_images:
                confidence = 0.8

            # Check for image-related keywords
            for keyword, score in image_keywords.items():
                if keyword in user_input:
                    confidence = max(confidence, score)

            return confidence

        except Exception as e:
            logger.error(f"Error in can_handle_task: {e}")
            return 0.0

    @hookimpl
    def execute_capability(self, context: CapabilityContext) -> CapabilityResult:
        """Execute the image processing capability."""
        try:
            # Get the specific capability being invoked
            capability_id = context.metadata.get("capability_id", "unknown")
            
            # Route to specific capability handler based on capability_id
            capability_map = {
                "analyze_image": self._handle_analyze_image_internal,
                "transform_image": self._handle_transform_image_internal,
                "convert_image_format": self._handle_convert_image_internal,
            }
            
            if capability_id in capability_map:
                handler = capability_map[capability_id]
                return handler(context)
            else:
                # Fallback to legacy operation-based routing for unknown capabilities
                user_input = self._extract_user_input(context)
                logger.info(f"Processing image task: {user_input[:100]}...")

                # Extract images from the task
                if not hasattr(context.task, "history") or not context.task.history:
                    return CapabilityResult(
                        content="Error: No message history found. Please provide an image to process.",
                        success=False,
                        error="No message history",
                    )

                image_parts = []
                for message in context.task.history:
                    if hasattr(message, "parts") and message.parts:
                        image_parts.extend(
                            ImageProcessor.extract_image_parts(message.parts)
                        )

                if not image_parts:
                    return CapabilityResult(
                        content="Error: No images found in the message. Please upload an image to process.",
                        success=False,
                        error="No images found",
                    )

                # Determine the operation based on user input
                operation = self._determine_operation(user_input)

                # Process the first image
                image_part = image_parts[0]
                if not image_part["data"]:
                    return CapabilityResult(
                        content="Error: No image data found. Images must be embedded as base64 data.",
                        success=False,
                        error="No image data",
                    )

                # Process based on operation
                if operation == "analyze":
                    return self._analyze_image(image_part, user_input)
                elif operation == "transform":
                    return self._transform_image(image_part, user_input)
                elif operation == "convert":
                    return self._convert_image(image_part, user_input)
                else:
                    # Default to analysis
                    return self._analyze_image(image_part, user_input)

        except Exception as e:
            logger.error(f"Error executing image processing capability: {e}", exc_info=True)
            return CapabilityResult(
                content=f"Error processing image: {str(e)}", success=False, error=str(e)
            )

    @hookimpl
    def get_ai_functions(self) -> list[AIFunction]:
        """Get AI functions provided by this plugin."""
        return [
            AIFunction(
                name="analyze_image",
                description="Analyze an uploaded image and return detailed insights including metadata, dimensions, and visual characteristics",
                parameters={
                    "type": "object",
                    "properties": {
                        "analysis_type": {
                            "type": "string",
                            "enum": ["basic", "detailed", "color"],
                            "description": "Type of analysis to perform",
                        }
                    },
                    "required": [],
                },
                handler=self._handle_analyze_image,
            ),
            AIFunction(
                name="transform_image",
                description="Transform images with various operations like resize, rotate, flip, and apply filters",
                parameters={
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
                handler=self._handle_transform_image,
            ),
            AIFunction(
                name="convert_image_format",
                description="Convert images between different formats (PNG, JPEG, WebP, etc.)",
                parameters={
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
                handler=self._handle_convert_image,
            ),
        ]

    @hookimpl
    def validate_config(self, config: dict) -> ValidationResult:
        """Validate plugin configuration."""
        errors = []
        warnings = []

        # Check max image size
        max_size = config.get("max_image_size_mb", 10)
        if not isinstance(max_size, (int, float)) or max_size <= 0:
            errors.append("max_image_size_mb must be a positive number")
        elif max_size > 100:
            warnings.append(
                "max_image_size_mb is very large, consider reducing for performance"
            )

        # Check supported formats
        supported_formats = config.get("supported_formats", [])
        if not isinstance(supported_formats, list):
            errors.append("supported_formats must be a list")
        else:
            valid_formats = set(ImageProcessor.IMAGE_FORMATS.keys())
            for fmt in supported_formats:
                if fmt not in valid_formats:
                    errors.append(f"Unsupported image format: {fmt}")

        # Check thumbnail size
        thumbnail_size = config.get("default_thumbnail_size", [200, 200])
        if not isinstance(thumbnail_size, list) or len(thumbnail_size) != 2:
            errors.append(
                "default_thumbnail_size must be a list of two integers [width, height]"
            )
        elif not all(isinstance(x, int) and x > 0 for x in thumbnail_size):
            errors.append("default_thumbnail_size values must be positive integers")

        return ValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    @hookimpl
    def configure_services(self, services: dict) -> None:
        """Configure services for the plugin."""
        # Store services if needed for future use
        self.services = services
        logger.info("Image processing plugin configured with services")

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
            return "transform"
        elif any(word in user_input for word in ["rotate", "turn"]):
            return "transform"
        elif any(word in user_input for word in ["flip", "mirror"]):
            return "transform"
        elif any(
            word in user_input for word in ["filter", "blur", "sharpen", "enhance"]
        ):
            return "transform"
        elif any(word in user_input for word in ["convert", "format", "change to"]):
            return "convert"
        elif any(word in user_input for word in ["thumbnail", "thumb"]):
            return "transform"
        else:
            return "analyze"

    def _analyze_image(self, image_part: dict, user_input: str) -> CapabilityResult:
        """Analyze an image and return insights."""
        result = ImageProcessor.process_image(
            image_part["data"], image_part["mimeType"]
        )

        if not result["success"]:
            return CapabilityResult(
                content=f"Error processing image: {result.get('error', 'Unknown error')}",
                success=False,
                error=result.get("error", "Unknown error"),
            )

        # Determine analysis type
        analysis_type = "basic"
        if "detailed" in user_input.lower():
            analysis_type = "detailed"
        elif "color" in user_input.lower():
            analysis_type = "color"

        # Build analysis response
        metadata = result["metadata"]
        response_parts = [
            "Image Analysis Results:",
            f"- Format: {metadata['format']}",
            f"- Dimensions: {metadata['width']}x{metadata['height']} pixels",
            f"- Mode: {metadata['mode']}",
            f"- File Hash: {metadata['hash'][:8]}...",
        ]

        if analysis_type in ["detailed", "color"]:
            response_parts.extend(
                [
                    f"- Mean Brightness: {metadata.get('mean_brightness', 'N/A'):.1f}",
                    f"- Shape: {metadata.get('shape', 'N/A')}",
                ]
            )

            if analysis_type == "color" and "channel_means" in metadata:
                channels = metadata["channel_means"]
                response_parts.append(
                    f"- RGB Channel Means: R={channels[0]:.1f}, G={channels[1]:.1f}, B={channels[2]:.1f}"
                )

        return CapabilityResult(
            content="\n".join(response_parts),
            success=True,
            metadata={"operation": "analyze", "analysis_type": analysis_type},
        )

    def _transform_image(self, image_part: dict, user_input: str) -> CapabilityResult:
        """Transform an image based on user input."""
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
        user_input_lower = user_input.lower()

        try:
            # Determine transformation type and parameters
            if "resize" in user_input_lower or "scale" in user_input_lower:
                # Extract size from input (default to 800x600)
                import re

                size_match = re.search(r"(\d+)[x×](\d+)", user_input)
                if size_match:
                    width, height = int(size_match.group(1)), int(size_match.group(2))
                else:
                    width, height = 800, 600

                image = ImageProcessor.resize_image(image, (width, height))
                transform_msg = f"Resized from {original_size} to {image.size}"

            elif "thumbnail" in user_input_lower:
                # Extract size or use default
                import re

                size_match = re.search(r"(\d+)[x×](\d+)", user_input)
                if size_match:
                    width, height = int(size_match.group(1)), int(size_match.group(2))
                else:
                    width, height = 200, 200

                image = ImageProcessor.create_thumbnail(image, (width, height))
                transform_msg = (
                    f"Created thumbnail from {original_size} to {image.size}"
                )

            elif "rotate" in user_input_lower:
                # Extract degrees
                import re

                degree_match = re.search(r"(\d+)\s*(?:degree|deg)", user_input)
                if degree_match:
                    degrees = float(degree_match.group(1))
                else:
                    degrees = 90.0  # Default rotation

                image = ImageProcessor.rotate_image(image, degrees)
                transform_msg = f"Rotated {degrees} degrees"

            elif "flip" in user_input_lower:
                direction = (
                    "horizontal" if "horizontal" in user_input_lower else "vertical"
                )
                image = ImageProcessor.flip_image(image, direction)
                transform_msg = f"Flipped {direction}ly"

            elif any(
                filter_name in user_input_lower
                for filter_name in [
                    "blur",
                    "sharpen",
                    "edge",
                    "emboss",
                    "enhance",
                    "brightness",
                    "contrast",
                ]
            ):
                # Determine filter type
                for filter_name in [
                    "blur",
                    "sharpen",
                    "edge",
                    "emboss",
                    "enhance",
                    "brightness",
                    "contrast",
                ]:
                    if filter_name in user_input_lower:
                        image = ImageProcessor.apply_filter(image, filter_name)
                        transform_msg = f"Applied {filter_name} filter"
                        break
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

    def _convert_image(self, image_part: dict, user_input: str) -> CapabilityResult:
        """Convert image format."""
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

        # Determine target format
        user_input_lower = user_input.lower()
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

        try:
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

    # Internal capability handlers for routing
    def _handle_analyze_image_internal(self, context: CapabilityContext) -> CapabilityResult:
        """Handle analyze_image capability internally."""
        user_input = self._extract_user_input(context)
        
        # Extract images from the task
        if not hasattr(context.task, "history") or not context.task.history:
            return CapabilityResult(
                content="Error: No message history found. Please provide an image to process.",
                success=False,
                error="No message history",
            )

        image_parts = []
        for message in context.task.history:
            if hasattr(message, "parts") and message.parts:
                image_parts.extend(
                    ImageProcessor.extract_image_parts(message.parts)
                )

        if not image_parts:
            return CapabilityResult(
                content="Error: No images found in the message. Please upload an image to process.",
                success=False,
                error="No images found",
            )

        # Process the first image
        image_part = image_parts[0]
        if not image_part["data"]:
            return CapabilityResult(
                content="Error: No image data found. Images must be embedded as base64 data.",
                success=False,
                error="No image data",
            )

        return self._analyze_image(image_part, user_input)

    def _handle_transform_image_internal(self, context: CapabilityContext) -> CapabilityResult:
        """Handle transform_image capability internally."""
        user_input = self._extract_user_input(context)
        
        # Extract images from the task
        if not hasattr(context.task, "history") or not context.task.history:
            return CapabilityResult(
                content="Error: No message history found. Please provide an image to process.",
                success=False,
                error="No message history",
            )

        image_parts = []
        for message in context.task.history:
            if hasattr(message, "parts") and message.parts:
                image_parts.extend(
                    ImageProcessor.extract_image_parts(message.parts)
                )

        if not image_parts:
            return CapabilityResult(
                content="Error: No images found in the message. Please upload an image to process.",
                success=False,
                error="No images found",
            )

        # Process the first image
        image_part = image_parts[0]
        if not image_part["data"]:
            return CapabilityResult(
                content="Error: No image data found. Images must be embedded as base64 data.",
                success=False,
                error="No image data",
            )

        return self._transform_image(image_part, user_input)

    def _handle_convert_image_internal(self, context: CapabilityContext) -> CapabilityResult:
        """Handle convert_image_format capability internally."""
        user_input = self._extract_user_input(context)
        
        # Extract images from the task
        if not hasattr(context.task, "history") or not context.task.history:
            return CapabilityResult(
                content="Error: No message history found. Please provide an image to process.",
                success=False,
                error="No message history",
            )

        image_parts = []
        for message in context.task.history:
            if hasattr(message, "parts") and message.parts:
                image_parts.extend(
                    ImageProcessor.extract_image_parts(message.parts)
                )

        if not image_parts:
            return CapabilityResult(
                content="Error: No images found in the message. Please upload an image to process.",
                success=False,
                error="No images found",
            )

        # Process the first image
        image_part = image_parts[0]
        if not image_part["data"]:
            return CapabilityResult(
                content="Error: No image data found. Images must be embedded as base64 data.",
                success=False,
                error="No image data",
            )

        return self._convert_image(image_part, user_input)

    # AI Function handlers
    async def _handle_analyze_image(
        self, task: Task, context: CapabilityContext
    ) -> CapabilityResult:
        """Handle AI function call for image analysis."""
        return self._handle_analyze_image_internal(context)

    async def _handle_transform_image(
        self, task: Task, context: CapabilityContext
    ) -> CapabilityResult:
        """Handle AI function call for image transformation."""
        return self._handle_transform_image_internal(context)

    async def _handle_convert_image(
        self, task: Task, context: CapabilityContext
    ) -> CapabilityResult:
        """Handle AI function call for image format conversion."""
        return self._handle_convert_image_internal(context)
