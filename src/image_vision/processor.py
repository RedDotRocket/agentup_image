"""
Image processing utilities for the AgentUp image processing plugin.

This module provides all the core image processing functionality extracted
from the original MultiModalProcessor.
"""

import base64
import hashlib
import io
import mimetypes
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from a2a.types import DataPart, Part
from PIL import Image

logger = structlog.get_logger(__name__)


class ImageProcessor:
    """Process image inputs and outputs."""

    # Supported image formats
    IMAGE_FORMATS = {
        "image/png": [".png"],
        "image/jpeg": [".jpg", ".jpeg"],
        "image/webp": [".webp"],
        "image/gif": [".gif"],
        "image/bmp": [".bmp"],
    }

    # File size limits (in MB)
    MAX_FILE_SIZES = {"image": 10, "default": 25}

    @classmethod
    def extract_image_parts(cls, parts: list[Part]) -> list[dict[str, str]]:
        """Extract image file parts from message parts."""
        image_parts = []

        for part in parts:
            # Images should be FilePart according to A2A spec
            if hasattr(part, "root") and part.root.kind == "file":
                file_part = part.root
                # Get mime type using the proper attribute (prefer snake_case over camelCase)
                mime_type = getattr(file_part.file, "mime_type", None) or getattr(
                    file_part.file, "mimeType", None
                )

                if mime_type and mime_type.startswith("image/"):
                    # Return dict with file info for easier processing
                    image_info = {
                        "name": file_part.file.name or "image",
                        "mimeType": mime_type,
                        "data": file_part.file.bytes
                        if hasattr(file_part.file, "bytes")
                        else None,
                        "uri": file_part.file.uri
                        if hasattr(file_part.file, "uri")
                        else None,
                    }
                    image_parts.append(image_info)

        return image_parts

    @classmethod
    def extract_parts_by_type(
        cls, parts: list[Part], mime_type_prefix: str = "image/"
    ) -> list[Part]:
        """Extract parts matching the image mime type prefix."""
        matching_parts = []

        for part in parts:
            # Check FilePart (for images)
            if hasattr(part, "root") and part.root.kind == "file":
                file_part = part.root
                # Get mime type using the proper attribute (prefer snake_case over camelCase)
                mime_type = getattr(file_part.file, "mime_type", None) or getattr(
                    file_part.file, "mimeType", None
                )
                if mime_type and mime_type.startswith(mime_type_prefix):
                    matching_parts.append(part)
            # Check DataPart (for structured data)
            elif hasattr(part, "root") and part.root.kind == "data":
                data_part = part.root
                # Get mime type using the proper attribute (prefer snake_case over camelCase)
                mime_type = getattr(data_part, "mime_type", None) or getattr(
                    data_part, "mimeType", None
                )
                if mime_type and mime_type.startswith(mime_type_prefix):
                    matching_parts.append(part)

        return matching_parts

    @classmethod
    def process_image(cls, image_data: str, mime_type: str) -> dict[str, Any]:
        """Process base64 encoded image data."""
        try:
            # Decode base64 data
            image_bytes = base64.b64decode(image_data)

            # Open image with PIL
            image = Image.open(io.BytesIO(image_bytes))

            # Extract metadata
            metadata = {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.width,
                "height": image.height,
                "mime_type": mime_type,
            }

            # Convert to numpy array for processing
            image_array = np.array(image)

            # Basic image analysis
            metadata["shape"] = image_array.shape
            metadata["dtype"] = str(image_array.dtype)

            # Calculate basic statistics
            if len(image_array.shape) == 2:  # Grayscale
                metadata["mean_brightness"] = float(np.mean(image_array))
                metadata["std_brightness"] = float(np.std(image_array))
            elif len(image_array.shape) == 3:  # Color
                metadata["mean_brightness"] = float(np.mean(image_array))
                metadata["channel_means"] = [
                    float(np.mean(image_array[:, :, i]))
                    for i in range(image_array.shape[2])
                ]

            # Generate hash for deduplication
            metadata["hash"] = hashlib.sha256(image_bytes).hexdigest()

            return {
                "success": True,
                "metadata": metadata,
                "image": image,  # Return PIL Image object for further processing
            }

        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return {"success": False, "error": str(e)}

    @classmethod
    def save_image(
        cls, image: Image.Image, output_path: str | Path, format: str | None = None
    ) -> bool:
        """Save PIL Image to file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine format from extension if not provided
            if not format:
                format = output_path.suffix[1:].upper()
                if format == "JPG":
                    format = "JPEG"

            image.save(output_path, format=format)
            logger.info(f"Saved image to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False

    @classmethod
    def resize_image(cls, image: Image.Image, max_size: tuple) -> Image.Image:
        """Resize image maintaining aspect ratio."""
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image

    @classmethod
    def convert_image_format(cls, image: Image.Image, target_format: str) -> bytes:
        """Convert image to different format."""
        output = io.BytesIO()

        # Handle format conversions
        if target_format.upper() == "JPEG" and image.mode == "RGBA":
            # Convert RGBA to RGB for JPEG
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image

        image.save(output, format=target_format.upper())
        return output.getvalue()

    @classmethod
    def encode_image_base64(cls, image: Image.Image, format: str = "PNG") -> str:
        """Encode PIL Image to base64 string."""
        buffer = io.BytesIO()

        # Handle format conversions for encoding
        if format.upper() == "JPEG" and image.mode == "RGBA":
            # Convert RGBA to RGB for JPEG
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image

        image.save(buffer, format=format.upper())
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @classmethod
    def validate_file_size(cls, data: str, file_type: str = "image") -> bool:
        """Validate file size against limits."""
        # Calculate size in MB
        size_bytes = len(base64.b64decode(data))
        size_mb = size_bytes / (1024 * 1024)

        # Get limit for file type
        limit_mb = cls.MAX_FILE_SIZES.get(file_type, cls.MAX_FILE_SIZES["default"])

        return size_mb <= limit_mb

    @classmethod
    def create_data_part(
        cls, file_path: str | Path, name: str | None = None
    ) -> DataPart | None:
        """Create DataPart from image file."""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            # Determine mime type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type or not mime_type.startswith("image/"):
                logger.error(f"File is not an image: {file_path}")
                return None

            # Read and encode file
            with open(file_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")

            # Create DataPart
            return DataPart(name=name or file_path.name, mimeType=mime_type, data=data)

        except Exception as e:
            logger.error(f"Failed to create DataPart from file: {e}")
            return None

    @classmethod
    def extract_all_images(cls, parts: list[Part]) -> list[dict[str, Any]]:
        """Extract all image content from message parts."""
        images = []

        for part in parts:
            if hasattr(part, "root") and part.root.kind == "file":
                file_part = part.root
                # Get mime type using the proper attribute (prefer snake_case over camelCase)
                mime_type = getattr(file_part.file, "mime_type", None) or getattr(
                    file_part.file, "mimeType", None
                )

                if mime_type and mime_type.startswith("image/"):
                    image_info = {
                        "name": file_part.file.name or "image",
                        "mime_type": mime_type,
                        "data": file_part.file.bytes
                        if hasattr(file_part.file, "bytes")
                        else None,
                        "uri": file_part.file.uri
                        if hasattr(file_part.file, "uri")
                        else None,
                    }
                    images.append(image_info)

        return images

    @classmethod
    def get_image_info(cls, image: Image.Image) -> dict[str, Any]:
        """Get comprehensive information about a PIL Image."""
        info = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height,
        }

        # Add EXIF data if available
        if hasattr(image, "_getexif") and image._getexif():
            info["exif"] = image._getexif()

        # Add additional info if available
        if hasattr(image, "info") and image.info:
            info["additional"] = image.info

        return info

    @classmethod
    def create_thumbnail(
        cls, image: Image.Image, size: tuple = (200, 200)
    ) -> Image.Image:
        """Create a thumbnail of the image."""
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail

    @classmethod
    def apply_filter(cls, image: Image.Image, filter_name: str) -> Image.Image:
        """Apply basic filters to an image."""
        from PIL import ImageEnhance, ImageFilter

        if filter_name.lower() == "blur":
            return image.filter(ImageFilter.BLUR)
        elif filter_name.lower() == "sharpen":
            return image.filter(ImageFilter.SHARPEN)
        elif filter_name.lower() == "edge":
            return image.filter(ImageFilter.FIND_EDGES)
        elif filter_name.lower() == "emboss":
            return image.filter(ImageFilter.EMBOSS)
        elif filter_name.lower() == "enhance":
            enhancer = ImageEnhance.Sharpness(image)
            return enhancer.enhance(1.5)
        elif filter_name.lower() == "brightness":
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(1.2)
        elif filter_name.lower() == "contrast":
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.2)
        else:
            logger.warning(f"Unknown filter: {filter_name}")
            return image

    @classmethod
    def rotate_image(cls, image: Image.Image, degrees: float) -> Image.Image:
        """Rotate an image by the specified degrees."""
        return image.rotate(degrees, expand=True)

    @classmethod
    def flip_image(cls, image: Image.Image, direction: str) -> Image.Image:
        """Flip an image horizontally or vertically."""
        if direction.lower() == "horizontal":
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif direction.lower() == "vertical":
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        else:
            logger.warning(f"Unknown flip direction: {direction}")
            return image
