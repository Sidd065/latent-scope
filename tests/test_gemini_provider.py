"""Tests for the native Gemini model providers."""
import sys
import types as py_types
from types import SimpleNamespace

import pytest


class FakeEmbedContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakePart:
    def __init__(self, text=None):
        self.text = text


class FakeContent:
    def __init__(self, role=None, parts=None):
        self.role = role
        self.parts = parts or []


class FakeModels:
    def __init__(self):
        self.embed_calls = []
        self.generate_calls = []

    def embed_content(self, **kwargs):
        self.embed_calls.append(kwargs)
        contents = kwargs["contents"]
        values = contents if isinstance(contents, list) else [contents]
        embeddings = [
            SimpleNamespace(values=[float(index), float(len(str(value)))])
            for index, value in enumerate(values)
        ]
        return SimpleNamespace(embeddings=embeddings)

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        return SimpleNamespace(text="Short Label")


@pytest.fixture
def fake_google_genai(monkeypatch):
    fake_models = FakeModels()

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = fake_models

    google_module = py_types.ModuleType("google")
    genai_module = py_types.ModuleType("google.genai")
    types_module = py_types.ModuleType("google.genai.types")

    types_module.EmbedContentConfig = FakeEmbedContentConfig
    types_module.GenerateContentConfig = FakeGenerateContentConfig
    types_module.Part = FakePart
    types_module.Content = FakeContent
    genai_module.Client = FakeClient
    genai_module.types = types_module
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)
    monkeypatch.setattr(
        "latentscope.models.providers.gemini.get_key",
        lambda key: "gemini-test-key",
    )
    return fake_models


def test_embedding_001_sends_list_and_task_type(fake_google_genai):
    from latentscope.models.providers.gemini import GeminiEmbedProvider

    provider = GeminiEmbedProvider(
        "gemini-embedding-001",
        {"max_tokens": 8192, "task_type": "CLUSTERING"},
    )
    provider.load_model()

    embeddings = provider.embed(["alpha", "beta"], dimensions=768)

    assert len(fake_google_genai.embed_calls) == 1
    call = fake_google_genai.embed_calls[0]
    assert call["model"] == "gemini-embedding-001"
    assert call["contents"] == ["alpha", "beta"]
    assert call["config"].output_dimensionality == 768
    assert call["config"].task_type == "CLUSTERING"
    assert embeddings == [[0.0, 5.0], [1.0, 4.0]]


def test_embedding_2_sends_one_request_per_input_without_task_type(fake_google_genai):
    from latentscope.models.providers.gemini import GeminiEmbedProvider

    provider = GeminiEmbedProvider("gemini-embedding-2", {"max_tokens": 8192})
    provider.load_model()

    embeddings = provider.embed(["alpha", "beta"], dimensions=1536)

    assert len(fake_google_genai.embed_calls) == 2
    assert [call["contents"] for call in fake_google_genai.embed_calls] == ["alpha", "beta"]
    assert all(call["config"].output_dimensionality == 1536 for call in fake_google_genai.embed_calls)
    assert all("task_type" not in call["config"].kwargs for call in fake_google_genai.embed_calls)
    assert embeddings == [[0.0, 5.0], [0.0, 4.0]]


def test_gemini_chat_summarize_returns_response_text(fake_google_genai, monkeypatch):
    prompts_module = py_types.ModuleType("latentscope.models.providers.prompts")
    prompts_module.summarize_system_prompt = "system prompt"
    prompts_module.summarize = lambda items, context="": f"{context}: {', '.join(items)}"
    monkeypatch.setitem(sys.modules, "latentscope.models.providers.prompts", prompts_module)

    from latentscope.models.providers.gemini import GeminiChatProvider

    provider = GeminiChatProvider("gemini-2.5-flash", {"max_tokens": 1048576})
    provider.load_model()

    label = provider.summarize(["alpha", "beta"], "context")

    assert label == "Short Label"
    assert len(fake_google_genai.generate_calls) == 1
    call = fake_google_genai.generate_calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["config"].system_instruction == "system prompt"
    assert call["contents"][0].role == "user"
    assert call["contents"][0].parts[0].text == "context: alpha, beta"
