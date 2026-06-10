from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = get_embedding_model(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"query: {t}" if not t.startswith("passage:") else t for t in texts]
        vectors = self._model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_passage(self, text: str) -> list[float]:
        return self.embed([f"passage: {text}"])[0]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([f"query: {text}"])[0]
