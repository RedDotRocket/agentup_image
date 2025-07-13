"""Tests for the ImageProcessingPlugin class."""

import base64
import io
from unittest.mock import Mock
import pytest
from PIL import Image

from image_vision.plugin import ImageProcessingPlugin


class MockTask:
    """Mock task for testing."""

    def __init__(self, history=None):
        self.history = history or []


class MockMessage:
    """Mock message for testing."""

    def __init__(self, parts=None):
        self.parts = parts or []


class MockPart:
    """Mock part for testing."""

    def __init__(self, kind, content=None):
        self.root = Mock()
        self.root.kind = kind
        if kind == "text":
            self.root.text = content
        elif kind == "file":
            self.root.file = Mock()
            self.root.file.mimeType = content.get("mimeType") if content else None
            self.root.file.bytes = content.get("data") if content else None
            self.root.file.name = content.get("name") if content else "image.png"


class MockCapabilityContext:
    """Mock capability context for testing."""

    def __init__(self, task, config=None, metadata=None):
        self.task = task
        self.config = config or {}
        self.metadata = metadata or {}


class TestImageProcessingPlugin:
    """Test cases for ImageProcessingPlugin class."""

    @pytest.fixture
    def plugin(self):
        """Create plugin instance."""
        return ImageProcessingPlugin()

    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data."""
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @pytest.fixture
    def mock_context_with_image(self, sample_image_data):
        """Create mock context with image."""
        # Create mock parts
        text_part = MockPart("text", "analyze this image")
        image_part = MockPart(
            "file",
            {"mimeType": "image/png", "data": sample_image_data, "name": "test.png"},
        )

        # Create mock message and task
        message = MockMessage([text_part, image_part])
        task = MockTask([message])

        return MockCapabilityContext(task, metadata={})

    @pytest.fixture
    def mock_context_text_only(self):
        """Create mock context with text only."""
        text_part = MockPart("text", "analyze image")
        message = MockMessage([text_part])
        task = MockTask([message])

        return MockCapabilityContext(task, metadata={})

    def test_register_capability(self, plugin):
        """Test capability registration."""
        capabilities = plugin.register_capability()

        # Should return a list of capabilities
        assert isinstance(capabilities, list)
        assert len(capabilities) == 3  # analyze_image, transform_image, convert_image_format
        
        # Check first capability (analyze_image)
        analyze_capability = capabilities[0]
        assert analyze_capability.id == "analyze_image"
        assert analyze_capability.name == "Image Analysis"
        assert "multimodal" in [cap.value for cap in analyze_capability.capabilities]
        assert "ai_function" in [cap.value for cap in analyze_capability.capabilities]

    def test_can_handle_task_with_images(self, plugin, mock_context_with_image):
        """Test task handling with images present."""
        confidence = plugin.can_handle_task(mock_context_with_image)

        # Should have high confidence due to image presence
        assert confidence >= 0.8

    def test_can_handle_task_with_keywords(self, plugin, mock_context_text_only):
        """Test task handling with image-related keywords."""
        confidence = plugin.can_handle_task(mock_context_text_only)

        # Should have some confidence due to "analyze image" keywords
        assert confidence > 0.0

    def test_can_handle_task_no_images_no_keywords(self, plugin):
        """Test task handling with no images or keywords."""
        text_part = MockPart("text", "hello world")
        message = MockMessage([text_part])
        task = MockTask([message])
        context = MockCapabilityContext(task, metadata={})

        confidence = plugin.can_handle_task(context)

        # Should have low confidence
        assert confidence == 0.0

    def test_execute_capability_analyze(self, plugin, mock_context_with_image):
        """Test capability execution for image analysis."""
        result = plugin.execute_capability(mock_context_with_image)

        assert result.success is True
        assert "Image Analysis Results:" in result.content
        assert "Format:" in result.content
        assert "Dimensions:" in result.content

    def test_execute_capability_no_images(self, plugin, mock_context_text_only):
        """Test capability execution with no images."""
        result = plugin.execute_capability(mock_context_text_only)

        assert result.success is False
        assert "No images found" in result.content

    def test_execute_capability_no_history(self, plugin):
        """Test capability execution with no message history."""
        task = MockTask([])
        context = MockCapabilityContext(task, metadata={})

        result = plugin.execute_capability(context)

        assert result.success is False
        assert "No message history found" in result.content

    def test_get_ai_functions(self, plugin):
        """Test getting AI functions."""
        ai_functions = plugin.get_ai_functions()

        assert len(ai_functions) == 3
        function_names = [f.name for f in ai_functions]
        assert "analyze_image" in function_names
        assert "transform_image" in function_names
        assert "convert_image_format" in function_names

    def test_validate_config_valid(self, plugin):
        """Test configuration validation with valid config."""
        config = {
            "max_image_size_mb": 10,
            "supported_formats": ["image/png", "image/jpeg"],
            "default_thumbnail_size": [200, 200],
        }

        result = plugin.validate_config(config)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_config_invalid_size(self, plugin):
        """Test configuration validation with invalid size."""
        config = {"max_image_size_mb": -1}

        result = plugin.validate_config(config)

        assert result.valid is False
        assert len(result.errors) > 0
        assert any("positive number" in error for error in result.errors)

    def test_validate_config_invalid_format(self, plugin):
        """Test configuration validation with invalid format."""
        config = {"supported_formats": ["image/invalid"]}

        result = plugin.validate_config(config)

        assert result.valid is False
        assert len(result.errors) > 0
        assert any("Unsupported image format" in error for error in result.errors)

    def test_validate_config_invalid_thumbnail_size(self, plugin):
        """Test configuration validation with invalid thumbnail size."""
        config = {
            "default_thumbnail_size": [200]  # Should be [width, height]
        }

        result = plugin.validate_config(config)

        assert result.valid is False
        assert len(result.errors) > 0
        assert any("two integers" in error for error in result.errors)

    def test_determine_operation_resize(self, plugin):
        """Test operation determination for resize."""
        operation = plugin._determine_operation("resize the image to 800x600")
        assert operation == "transform"

    def test_determine_operation_convert(self, plugin):
        """Test operation determination for convert."""
        operation = plugin._determine_operation("convert to JPEG format")
        assert operation == "convert"

    def test_determine_operation_analyze(self, plugin):
        """Test operation determination for analyze (default)."""
        operation = plugin._determine_operation("what's in this image?")
        assert operation == "analyze"

    def test_transform_image_resize(self, plugin, sample_image_data):
        """Test image transformation (resize)."""
        image_part = {"data": sample_image_data, "mimeType": "image/png"}

        result = plugin._transform_image(image_part, "resize to 50x50")

        assert result.success is True
        assert "Transformation Complete" in result.content
        assert "Resized" in result.content

    def test_convert_image_png_to_jpeg(self, plugin, sample_image_data):
        """Test image format conversion."""
        image_part = {"data": sample_image_data, "mimeType": "image/png"}

        result = plugin._convert_image(image_part, "convert to JPEG")

        assert result.success is True
        assert "Format Conversion Complete" in result.content
        assert "JPEG" in result.content

    def test_extract_user_input(self, plugin, mock_context_with_image):
        """Test extracting user input from context."""
        user_input = plugin._extract_user_input(mock_context_with_image)

        assert user_input == "analyze this image"

    def test_extract_user_input_no_text(self, plugin):
        """Test extracting user input with no text parts."""
        message = MockMessage([])
        task = MockTask([message])
        context = MockCapabilityContext(task, metadata={})

        user_input = plugin._extract_user_input(context)

        assert user_input == ""
