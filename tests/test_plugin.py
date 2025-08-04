"""Tests for the ImageProcessingPlugin class using the new decorator-based system."""

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
    """Test cases for ImageProcessingPlugin class with decorator-based system."""

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

    def test_plugin_initialization(self, plugin):
        """Test plugin initialization and capability discovery."""
        # Plugin should inherit from Plugin base class
        assert hasattr(plugin, "_capabilities")
        assert (
            len(plugin._capabilities) == 3
        )  # analyze_image, transform_image, convert_image_format

        # Check that capabilities are discovered
        capability_ids = list(plugin._capabilities.keys())
        assert "analyze_image" in capability_ids
        assert "transform_image" in capability_ids
        assert "convert_image_format" in capability_ids

    def test_get_capability_definitions(self, plugin):
        """Test getting capability definitions."""
        definitions = plugin.get_capability_definitions()

        # Should return a list of capabilities
        assert isinstance(definitions, list)
        assert len(definitions) == 3

        # Check first capability (analyze_image)
        analyze_capability = next(d for d in definitions if d.id == "analyze_image")
        assert analyze_capability.name == "Image Analysis"
        assert "image:read" in analyze_capability.required_scopes
        assert "multimodal" in [cap.value for cap in analyze_capability.capabilities]

    def test_can_handle_task_with_capability_id(self, plugin, mock_context_with_image):
        """Test task handling with specific capability ID."""
        # Should return True for known capabilities when explicitly called
        assert plugin.can_handle_task("analyze_image", mock_context_with_image) is True
        assert (
            plugin.can_handle_task("transform_image", mock_context_with_image) is True
        )
        assert (
            plugin.can_handle_task("convert_image_format", mock_context_with_image)
            is True
        )

        # Should return 0.0 for unknown capabilities
        result = plugin.can_handle_task("unknown_capability", mock_context_with_image)
        assert result == 0.0

    def test_can_handle_task_with_images(self, plugin, mock_context_with_image):
        """Test task handling with images present."""
        confidence = plugin.can_handle_task(
            "unknown_capability", mock_context_with_image
        )

        # Should return 0.0 for unknown capabilities (let AI model handle)
        assert confidence == 0.0

    def test_can_handle_task_with_keywords(self, plugin, mock_context_text_only):
        """Test task handling with image-related keywords."""
        confidence = plugin.can_handle_task(
            "unknown_capability", mock_context_text_only
        )

        # Should return 0.0 for unknown capabilities (let AI model handle)
        assert confidence == 0.0

    def test_can_handle_task_with_processing_keywords(self, plugin):
        """Test task handling with explicit image processing keywords."""
        # Create context with explicit processing request
        text_part = MockPart("text", "please resize image to 800x600")
        message = MockMessage([text_part])
        task = MockTask([message])
        context = MockCapabilityContext(task, metadata={})

        confidence = plugin.can_handle_task("unknown_capability", context)

        # Should return 0.0 for unknown capabilities (let AI model handle)
        assert confidence == 0.0

    def test_can_handle_task_with_content_questions(self, plugin):
        """Test that content questions are not handled by this plugin."""
        # Create context with content question
        text_part = MockPart("text", "what type of vehicle is this?")
        message = MockMessage([text_part])
        task = MockTask([message])
        context = MockCapabilityContext(task, metadata={})

        confidence = plugin.can_handle_task("unknown_capability", context)

        # Should return 0.0 for unknown capabilities (let AI model handle)
        assert confidence == 0.0

    def test_can_handle_task_processing_with_capability_id(self, plugin):
        """Test that specific capability IDs are handled."""
        # Create context with any text
        text_part = MockPart("text", "analyze image metadata")
        message = MockMessage([text_part])
        task = MockTask([message])
        context = MockCapabilityContext(task, metadata={})

        # Should handle when explicitly called by capability ID
        assert plugin.can_handle_task("analyze_image", context) is True
        assert plugin.can_handle_task("transform_image", context) is True
        assert plugin.can_handle_task("convert_image_format", context) is True

    def test_can_handle_task_no_images_no_keywords(self, plugin):
        """Test task handling with no images or keywords."""
        text_part = MockPart("text", "hello world")
        message = MockMessage([text_part])
        task = MockTask([message])
        context = MockCapabilityContext(task, metadata={})

        confidence = plugin.can_handle_task("unknown_capability", context)

        # Should have low confidence
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_execute_capability_analyze_image(
        self, plugin, mock_context_with_image
    ):
        """Test capability execution for image analysis."""
        result = await plugin.execute_capability(
            "analyze_image", mock_context_with_image
        )

        assert result.success is True
        assert "Image Analysis Results:" in result.content
        assert "Format:" in result.content
        assert "Dimensions:" in result.content

    @pytest.mark.asyncio
    async def test_execute_capability_with_parameters(
        self, plugin, mock_context_with_image
    ):
        """Test capability execution with AI function parameters."""
        # Set parameters for detailed analysis
        mock_context_with_image.metadata["parameters"] = {"analysis_type": "detailed"}

        result = await plugin.execute_capability(
            "analyze_image", mock_context_with_image
        )

        assert result.success is True
        assert "Mean Brightness:" in result.content

    @pytest.mark.asyncio
    async def test_execute_capability_no_images(self, plugin, mock_context_text_only):
        """Test capability execution with no images."""
        result = await plugin.execute_capability(
            "analyze_image", mock_context_text_only
        )

        assert result.success is False
        assert "No images found" in result.content

    @pytest.mark.asyncio
    async def test_execute_capability_no_history(self, plugin):
        """Test capability execution with no message history."""
        task = MockTask([])
        context = MockCapabilityContext(task, metadata={})

        result = await plugin.execute_capability("analyze_image", context)

        assert result.success is False
        assert "No message history found" in result.content

    @pytest.mark.asyncio
    async def test_execute_capability_unknown(self, plugin, mock_context_with_image):
        """Test executing unknown capability."""
        result = await plugin.execute_capability(
            "unknown_capability", mock_context_with_image
        )

        assert result.success is False
        assert "Capability 'unknown_capability' not found" in result.content

    def test_get_ai_functions(self, plugin):
        """Test getting AI functions."""
        ai_functions = plugin.get_ai_functions()

        assert len(ai_functions) == 3
        function_names = [f.name for f in ai_functions]
        assert "analyze_image" in function_names
        assert "transform_image" in function_names
        assert "convert_image_format" in function_names

    def test_get_ai_functions_specific_capability(self, plugin):
        """Test getting AI functions for specific capability."""
        ai_functions = plugin.get_ai_functions("analyze_image")

        assert len(ai_functions) == 1
        assert ai_functions[0].name == "analyze_image"
        assert "analysis_type" in str(ai_functions[0].parameters)

    @pytest.mark.asyncio
    async def test_get_health_status(self, plugin):
        """Test getting plugin health status."""
        status = await plugin.get_health_status()

        assert status["status"] == "healthy"
        assert status["version"] == "1.0.0"
        assert len(status["capabilities"]) == 3
        assert "analyze_image" in status["capabilities"]

    def test_configure_plugin(self, plugin):
        """Test plugin configuration."""
        config = {"max_image_size": 5242880}
        plugin.configure(config)

        assert plugin._config["max_image_size"] == 5242880

    def test_configure_services(self, plugin):
        """Test service configuration."""
        services = {"llm": Mock()}
        plugin.configure_services(services)

        assert "llm" in plugin._services

    def test_determine_operation_resize(self, plugin):
        """Test operation determination for resize."""
        operation = plugin._determine_operation("resize the image to 800x600")
        assert operation == "resize"

    def test_determine_operation_convert(self, plugin):
        """Test operation determination for convert."""
        operation = plugin._determine_operation("convert to JPEG format")
        assert operation == "convert"

    def test_determine_operation_analyze(self, plugin):
        """Test operation determination for analyze (default)."""
        operation = plugin._determine_operation("what's in this image?")
        assert operation == "analyze"

    @pytest.mark.asyncio
    async def test_transform_image_resize(self, plugin, sample_image_data):
        """Test image transformation (resize)."""
        # Create context with image and parameters
        image_part = MockPart(
            "file",
            {"mimeType": "image/png", "data": sample_image_data, "name": "test.png"},
        )
        message = MockMessage([MockPart("text", "resize to 50x50"), image_part])
        task = MockTask([message])
        context = MockCapabilityContext(
            task,
            metadata={"parameters": {"operation": "resize", "target_size": "50x50"}},
        )

        result = await plugin.execute_capability("transform_image", context)

        assert result.success is True
        assert "Image Transformation Complete" in result.content
        assert "Resized" in result.content

    @pytest.mark.asyncio
    async def test_convert_image_png_to_jpeg(self, plugin, sample_image_data):
        """Test image format conversion."""
        # Create context with image and parameters
        image_part = MockPart(
            "file",
            {"mimeType": "image/png", "data": sample_image_data, "name": "test.png"},
        )
        message = MockMessage([MockPart("text", "convert to JPEG"), image_part])
        task = MockTask([message])
        context = MockCapabilityContext(
            task, metadata={"parameters": {"target_format": "JPEG"}}
        )

        result = await plugin.execute_capability("convert_image_format", context)

        assert result.success is True
        assert "Image Format Conversion Complete" in result.content
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

    def test_plugin_id_property(self, plugin):
        """Test plugin ID property."""
        assert plugin.plugin_id == "imageprocessing"

    @pytest.mark.asyncio
    async def test_lifecycle_methods(self, plugin):
        """Test optional lifecycle methods don't raise errors."""
        # These methods should exist and not raise errors
        plugin.on_install()
        plugin.on_uninstall()
        plugin.on_enable()
        plugin.on_disable()

    def test_capability_metadata_validation(self, plugin):
        """Test that capabilities have proper metadata."""
        for capability_id, capability_meta in plugin._capabilities.items():
            # All capabilities should have proper IDs and names
            assert capability_meta.id == capability_id
            assert capability_meta.name is not None
            assert capability_meta.description is not None

            # All capabilities should be AI functions with parameters
            assert capability_meta.ai_function is True
            assert capability_meta.ai_parameters is not None
            assert "type" in capability_meta.ai_parameters

            # All capabilities should be multimodal
            assert capability_meta.multimodal is True

            # All capabilities should have scopes
            assert len(capability_meta.scopes) > 0

    @pytest.mark.asyncio
    async def test_error_handling_invalid_image_data(self, plugin):
        """Test error handling with invalid image data."""
        # Create context with invalid image data
        image_part = MockPart(
            "file",
            {"mimeType": "image/png", "data": "invalid_base64", "name": "test.png"},
        )
        message = MockMessage([MockPart("text", "analyze this image"), image_part])
        task = MockTask([message])
        context = MockCapabilityContext(task, metadata={})

        result = await plugin.execute_capability("analyze_image", context)

        assert result.success is False
        assert "Error" in result.content
