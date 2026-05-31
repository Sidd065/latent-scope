import os
import time
from .base import EmbedModelProvider


class ApproxTextEncoder:
    def encode(self, text):
        text = "" if text is None else str(text)
        return [text[i:i + 4] for i in range(0, len(text), 4)]

    def decode(self, tokens):
        return "".join(tokens)


class VoyageAIEmbedProvider(EmbedModelProvider):
    def load_model(self):
        import voyageai
        from latentscope.util import get_key
        api_key = get_key("VOYAGE_API_KEY")
        if api_key is None:
            print("ERROR: No API key found for Voyage")
            print("Missing 'VOYAGE_API_KEY' variable in:", f"{os.getcwd()}/.env")
        self.client = voyageai.Client(api_key)
        self.encoder = ApproxTextEncoder()

    def embed(self, inputs, dimensions=None):
        time.sleep(0.1) # TODO proper rate limiting
        # We truncate the input ourselves, even though the API supports truncation its still possible to send too big a batch
        enc = self.encoder
        max_tokens = self.params["max_tokens"]
        inputs = [
            enc.decode(enc.encode(b)[:max_tokens]) if len(enc.encode(b)) > max_tokens else b
            for b in inputs
        ]
        response = self.client.embed(texts=inputs, model=self.name, truncation=self.params["truncation"])
        embeddings = response.embeddings
        return embeddings
