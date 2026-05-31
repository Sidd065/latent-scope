"""Tests for latentscope.models (model listing and provider resolution)."""
import pytest


class TestEmbeddingModelList:
    def test_returns_list(self):
        from latentscope.models import get_embedding_model_list
        models = get_embedding_model_list()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_each_model_has_required_fields(self):
        from latentscope.models import get_embedding_model_list
        for model in get_embedding_model_list():
            assert 'id' in model, f"Model missing 'id': {model}"
            assert 'provider' in model, f"Model missing 'provider': {model}"
            assert 'name' in model, f"Model missing 'name': {model}"

    def test_model_ids_are_unique(self):
        from latentscope.models import get_embedding_model_list
        ids = [m['id'] for m in get_embedding_model_list()]
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"

    def test_no_emoji_in_ids(self):
        """Model IDs should not contain emoji."""
        from latentscope.models import get_embedding_model_list
        for model in get_embedding_model_list():
            assert '🤗' not in model['id'], f"Emoji found in model id: {model['id']}"

    def test_no_local_embedding_providers(self):
        from latentscope.models import get_embedding_model_list

        local_providers = {"colbert", "colpali", "huggingface", "transformers", "🤗"}
        local_prefixes = ("colbert-", "colpali-", "huggingface-", "transformers-", "🤗-")
        for model in get_embedding_model_list():
            assert model["provider"] not in local_providers
            assert not model["id"].startswith(local_prefixes)


class TestChatModelList:
    def test_returns_list(self):
        from latentscope.models import get_chat_model_list
        models = get_chat_model_list()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_each_model_has_required_fields(self):
        from latentscope.models import get_chat_model_list
        for model in get_chat_model_list():
            assert 'id' in model
            assert 'provider' in model
            assert 'name' in model

    def test_no_local_chat_providers(self):
        from latentscope.models import get_chat_model_list

        local_providers = {"nltk", "ollama", "huggingface", "transformers", "🤗"}
        local_prefixes = ("nltk-", "ollama-", "huggingface-", "transformers-", "🤗-")
        for model in get_chat_model_list():
            assert model["provider"] not in local_providers
            assert not model["id"].startswith(local_prefixes)


class TestGeminiProviderResolution:
    def test_returns_gemini_embedding_provider(self):
        from latentscope.models import get_embedding_model
        from latentscope.models.providers.gemini import GeminiEmbedProvider

        provider = get_embedding_model("gemini-gemini-embedding-001")
        assert isinstance(provider, GeminiEmbedProvider)

    def test_returns_gemini_chat_provider(self):
        from latentscope.models import get_chat_model
        from latentscope.models.providers.gemini import GeminiChatProvider

        provider = get_chat_model("gemini-gemini-2.5-flash")
        assert isinstance(provider, GeminiChatProvider)


class TestGetEmbeddingModelDict:
    def test_returns_known_model(self):
        from latentscope.models import get_embedding_model_dict, get_embedding_model_list
        first_model = get_embedding_model_list()[0]
        result = get_embedding_model_dict(first_model['id'])
        assert result['id'] == first_model['id']

    def test_raises_for_unknown_model(self):
        from latentscope.models import get_embedding_model_dict
        with pytest.raises(ValueError, match="not found"):
            get_embedding_model_dict("nonexistent-model-xyz")

