import time
import os

from .base import ChatModelProvider, EmbedModelProvider

from latentscope.util import get_key


class ApproxTextEncoder:
    def encode(self, text):
        text = "" if text is None else str(text)
        return [text[i:i + 4] for i in range(0, len(text), 4)]

    def decode(self, tokens):
        return "".join(tokens)

class MistralAIEmbedProvider(EmbedModelProvider):
    def load_model(self):
        from mistralai.client import MistralClient
        api_key = get_key("MISTRAL_API_KEY")
        if api_key is None:
            print("ERROR: No API key found for Mistral")
            print("Missing 'MISTRAL_API_KEY' variable in:", f"{os.getcwd()}/.env")
        self.client = MistralClient(api_key=api_key)

    def embed(self, inputs, dimensions=None):
        time.sleep(0.1) # TODO proper rate limiting
        response = self.client.embeddings(input=inputs, model=self.name)
        return [e.embedding for e in response.data]

class MistralAIChatProvider(ChatModelProvider):
    def load_model(self):
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        self.ChatMessage = ChatMessage
        api_key = get_key("MISTRAL_API_KEY")
        if api_key is None:
            print("ERROR: No API key found for Mistral")
            print("Missing 'MISTRAL_API_KEY' variable in:", f"{os.getcwd()}/.env")
        self.client = MistralClient(api_key=api_key)
        self.encoder = ApproxTextEncoder()

    def chat(self, messages):
        instances = [self.ChatMessage(content=message["content"], role=message["role"]) for message in messages]
        response = self.client.chat(
            model=self.name,
            messages=instances
        )
        return response.choices[0].message.content
