import os
import time

from latentscope.util import get_key

from .base import ChatModelProvider, EmbedModelProvider


class ApproxGeminiEncoder:
    """Small fallback tokenizer approximation for label truncation.

    Gemini docs describe a token as roughly four characters. This encoder keeps
    the labeler truncation path working without adding a tokenizer dependency.
    """

    def encode(self, text):
        text = "" if text is None else str(text)
        return [text[i:i + 4] for i in range(0, len(text), 4)]

    def decode(self, tokens):
        return "".join(tokens)


def _embedding_values(embedding):
    if isinstance(embedding, dict):
        return embedding["values"]
    return embedding.values


class GeminiEmbedProvider(EmbedModelProvider):
    def load_model(self):
        from google import genai

        api_key = get_key("GEMINI_API_KEY")
        if api_key is None:
            print("ERROR: No API key found for Gemini")
            print("Missing 'GEMINI_API_KEY' variable in:", f"{os.getcwd()}/.env")
            self.client = genai.Client()
        else:
            self.client = genai.Client(api_key=api_key)
        self.encoder = ApproxGeminiEncoder()

    def _config(self, dimensions=None):
        from google.genai import types

        kwargs = {}
        if dimensions is not None and dimensions > 0:
            kwargs["output_dimensionality"] = dimensions
        if self.name == "gemini-embedding-001" and self.params.get("task_type"):
            kwargs["task_type"] = self.params["task_type"]
        if not kwargs:
            return None
        return types.EmbedContentConfig(**kwargs)

    def _truncate_inputs(self, inputs):
        max_tokens = self.params.get("max_tokens")
        if not max_tokens:
            return inputs
        return [
            self.encoder.decode(self.encoder.encode(text)[:max_tokens])
            if len(self.encoder.encode(text)) > max_tokens
            else text
            for text in inputs
        ]

    def _embed_content(self, contents, dimensions=None):
        kwargs = {"model": self.name, "contents": contents}
        config = self._config(dimensions)
        if config is not None:
            kwargs["config"] = config
        return self.client.models.embed_content(**kwargs)

    def embed(self, inputs, dimensions=None):
        time.sleep(0.01)  # TODO proper rate limiting
        inputs = [text.replace("\n", " ") for text in inputs]
        inputs = self._truncate_inputs(inputs)

        # gemini-embedding-2 aggregates multiple inputs in a single request, so
        # call it once per row to preserve latent-scope's one-vector-per-row API.
        if self.name == "gemini-embedding-2":
            embeddings = []
            for text in inputs:
                response = self._embed_content(text, dimensions=dimensions)
                embeddings.append(_embedding_values(response.embeddings[0]))
            return embeddings

        response = self._embed_content(inputs, dimensions=dimensions)
        return [_embedding_values(embedding) for embedding in response.embeddings]


class GeminiChatProvider(ChatModelProvider):
    def load_model(self):
        from google import genai

        api_key = get_key("GEMINI_API_KEY")
        if api_key is None:
            print("ERROR: No API key found for Gemini")
            print("Missing 'GEMINI_API_KEY' variable in:", f"{os.getcwd()}/.env")
            self.client = genai.Client()
        else:
            self.client = genai.Client(api_key=api_key)
        self.encoder = ApproxGeminiEncoder()

    def _generation_config(self, system_instruction=None):
        from google.genai import types

        kwargs = {}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        for param in ("temperature", "top_p", "top_k", "max_output_tokens"):
            if param in self.params:
                kwargs[param] = self.params[param]
        if not kwargs:
            return None
        return types.GenerateContentConfig(**kwargs)

    def _contents_from_messages(self, messages):
        from google.genai import types

        contents = []
        system_messages = []
        for message in messages:
            role = message.get("role", "user")
            content = "" if message.get("content") is None else str(message.get("content"))
            if role == "system":
                system_messages.append(content)
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(role=gemini_role, parts=[types.Part(text=content)])
            )
        return contents, "\n\n".join(system_messages)

    def chat(self, messages):
        contents, system_instruction = self._contents_from_messages(messages)
        config = self._generation_config(system_instruction=system_instruction)
        kwargs = {"model": self.name, "contents": contents or ""}
        if config is not None:
            kwargs["config"] = config
        response = self.client.models.generate_content(**kwargs)
        return response.text

    def summarize(self, items, context=""):
        from .prompts import summarize, summarize_system_prompt

        prompt = summarize(items, context)
        return self.chat([
            {"role": "system", "content": summarize_system_prompt},
            {"role": "user", "content": prompt},
        ])
