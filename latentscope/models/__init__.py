import json

from .providers.cohereai import CohereAIEmbedProvider
from .providers.gemini import GeminiChatProvider, GeminiEmbedProvider
from .providers.mistralai import MistralAIChatProvider, MistralAIEmbedProvider
from .providers.openai import OpenAIChatProvider, OpenAIEmbedProvider
from .providers.togetherai import TogetherAIEmbedProvider
from .providers.voyageai import VoyageAIEmbedProvider

# Universal model ID scheme:
#   <provider>-<model-name>  where "/" in the model name is replaced by "___"
#
# Examples:
#   "text-embedding-3-small"          →  "openai-text-embedding-3-small"


def get_embedding_model_list():
    """Return the list of available embedding models."""
    from importlib.resources import files
    embedding_path = files('latentscope.models').joinpath('embedding_models.json')
    with open(embedding_path) as f:
        return json.load(f)


def get_embedding_model_dict(model_id):
    embed_model_list = get_embedding_model_list()
    embed_model_dict = {model['id']: model for model in embed_model_list}
    model = embed_model_dict.get(model_id)
    if not model:
        raise ValueError(f"Embedding model '{model_id}' not found")
    return model


def get_embedding_model(model_id):
    """Return a ModelProvider instance for the given embedding model id."""
    if model_id.startswith("custom_embedding-"):
        import os

        from latentscope.util import get_data_dir
        data_dir = get_data_dir()
        custom_models_path = os.path.join(data_dir, "custom_embedding_models.json")
        if os.path.exists(custom_models_path):
            with open(custom_models_path) as f:
                custom_models = json.load(f)
            model = next((m for m in custom_models if m["id"] == model_id), None)
            if model is None:
                raise ValueError(
                    f"Custom embedding model '{model_id}' not found in custom_embedding_models.json"
                )
        else:
            raise ValueError("No custom_embedding_models.json found in data directory")
        # Route directly — custom embedding models always use OpenAI-compatible API
        return OpenAIEmbedProvider(
            model['name'], model.get('params', {}), base_url=model['base_url']
        )
    else:
        model = get_embedding_model_dict(model_id)

    provider = model['provider']
    if provider == "openai":
        return OpenAIEmbedProvider(model['name'], model['params'])
    if provider == "gemini":
        return GeminiEmbedProvider(model['name'], model['params'])
    if provider == "mistralai":
        return MistralAIEmbedProvider(model['name'], model['params'])
    if provider == "cohereai":
        return CohereAIEmbedProvider(model['name'], model['params'])
    if provider == "togetherai":
        return TogetherAIEmbedProvider(model['name'], model['params'])
    if provider == "voyageai":
        return VoyageAIEmbedProvider(model['name'], model['params'])
    if provider == "custom_embedding":
        return OpenAIEmbedProvider(model['name'], model['params'], base_url=model['url'])
    raise ValueError(f"Unknown embedding provider '{provider}' for model '{model_id}'")


def get_chat_model_list():
    """Return the list of available chat models."""
    from importlib.resources import files
    chat_path = files('latentscope.models').joinpath('chat_models.json')
    with open(chat_path) as f:
        return json.load(f)


def get_chat_model_dict(model_id):
    chat_model_list = get_chat_model_list()
    chat_model_dict = {model['id']: model for model in chat_model_list}
    model = chat_model_dict.get(model_id)
    if not model:
        raise ValueError(f"Chat model '{model_id}' not found")
    return model


def get_chat_model(model_id):
    """Return a ModelProvider instance for the given chat model id."""
    if model_id.startswith("custom-"):
        import os

        from latentscope.util import get_data_dir
        data_dir = get_data_dir()
        custom_models_path = os.path.join(data_dir, "custom_models.json")
        if os.path.exists(custom_models_path):
            with open(custom_models_path) as f:
                custom_models = json.load(f)
            model = next((m for m in custom_models if m["id"] == model_id), None)
            if model is None:
                raise ValueError(f"Custom model '{model_id}' not found in custom_models.json")
        else:
            raise ValueError("No custom_models.json found in data directory")
    else:
        model = get_chat_model_dict(model_id)

    provider = model['provider']
    if provider == "openai":
        return OpenAIChatProvider(model['name'], model['params'])
    if provider == "gemini":
        return GeminiChatProvider(model['name'], model['params'])
    if provider == "custom":
        return OpenAIChatProvider(model['name'], model['params'], base_url=model['url'])
    if provider == "mistralai":
        return MistralAIChatProvider(model['name'], model['params'])
    raise ValueError(f"Unknown chat provider '{provider}' for model '{model_id}'")
