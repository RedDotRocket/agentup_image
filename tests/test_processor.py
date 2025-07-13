"""Tests for the ImageProcessor class."""

import base64
import io
import pytest
from PIL import Image

from image_vision.processor import ImageProcessor


class TestImageProcessor:
    """Test cases for ImageProcessor class."""

    @pytest.fixture
    def sample_image_data(self):
        """Create a sample image as base64 data."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @pytest.fixture
    def sample_image(self):
        """Create a sample PIL Image."""
        return Image.new("RGB", (100, 100), color="blue")

    def test_process_image_success(self, sample_image_data):
        """Test successful image processing."""
        result = ImageProcessor.process_image(sample_image_data, "image/png")

        assert result["success"] is True
        assert "metadata" in result
        assert "image" in result

        metadata = result["metadata"]
        assert metadata["format"] == "PNG"
        assert metadata["width"] == 100
        assert metadata["height"] == 100
        assert metadata["mode"] == "RGB"

    def test_process_image_invalid_data(self):
        """Test image processing with invalid data."""
        result = ImageProcessor.process_image("invalid_base64", "image/png")

        assert result["success"] is False
        assert "error" in result

    def test_resize_image(self, sample_image):
        """Test image resizing."""
        resized = ImageProcessor.resize_image(sample_image, (50, 50))

        # Note: thumbnail maintains aspect ratio, so exact size may vary
        assert resized.size[0] <= 50
        assert resized.size[1] <= 50

    def test_rotate_image(self, sample_image):
        """Test image rotation."""
        rotated = ImageProcessor.rotate_image(sample_image, 90)

        # After 90-degree rotation, width and height should be swapped
        assert rotated.size == (100, 100)  # Square image stays same

    def test_flip_image_horizontal(self, sample_image):
        """Test horizontal image flip."""
        flipped = ImageProcessor.flip_image(sample_image, "horizontal")

        assert flipped.size == sample_image.size

    def test_flip_image_vertical(self, sample_image):
        """Test vertical image flip."""
        flipped = ImageProcessor.flip_image(sample_image, "vertical")

        assert flipped.size == sample_image.size

    def test_encode_image_base64(self, sample_image):
        """Test encoding image to base64."""
        encoded = ImageProcessor.encode_image_base64(sample_image, "PNG")

        assert isinstance(encoded, str)
        assert len(encoded) > 0

        # Verify it's valid base64
        try:
            base64.b64decode(encoded)
        except Exception:
            pytest.fail("Generated string is not valid base64")

    def test_validate_file_size_valid(self, sample_image_data):
        """Test file size validation with valid size."""
        result = ImageProcessor.validate_file_size(sample_image_data, "image")

        assert result is True

    def test_validate_file_size_too_large(self):
        """Test file size validation with oversized data."""
        # Create a very large base64 string (simulating large file)
        large_data = "A" * (15 * 1024 * 1024)  # 15MB of 'A' characters
        large_b64 = base64.b64encode(large_data.encode()).decode()

        result = ImageProcessor.validate_file_size(large_b64, "image")

        assert result is False

    def test_create_thumbnail(self, sample_image):
        """Test thumbnail creation."""
        thumbnail = ImageProcessor.create_thumbnail(sample_image, (50, 50))

        assert thumbnail.size[0] <= 50
        assert thumbnail.size[1] <= 50

    def test_apply_filter_blur(self, sample_image):
        """Test applying blur filter."""
        filtered = ImageProcessor.apply_filter(sample_image, "blur")

        assert filtered.size == sample_image.size

    def test_apply_filter_unknown(self, sample_image):
        """Test applying unknown filter returns original image."""
        filtered = ImageProcessor.apply_filter(sample_image, "unknown_filter")

        assert filtered == sample_image

    def test_get_image_info(self, sample_image):
        """Test getting image information."""
        info = ImageProcessor.get_image_info(sample_image)

        assert "format" in info
        assert "mode" in info
        assert "size" in info
        assert "width" in info
        assert "height" in info
        assert info["width"] == 100
        assert info["height"] == 100

    def test_convert_image_format_rgba_to_jpeg(self):
        """Test converting RGBA image to JPEG format."""
        # Create RGBA image
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))

        # Convert to JPEG format (should handle RGBA -> RGB conversion)
        result = ImageProcessor.convert_image_format(img, "JPEG")

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_image_base64_rgba_to_jpeg(self):
        """Test encoding RGBA image as JPEG base64."""
        # Create RGBA image
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))

        # Encode as JPEG (should handle RGBA -> RGB conversion)
        encoded = ImageProcessor.encode_image_base64(img, "JPEG")

        assert isinstance(encoded, str)
        assert len(encoded) > 0
