import os
import transformers
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

def get_embedding_model(model_embedding_name, model_kwargs, encode_kwargs):
    if model_embedding_name == "text-embedding-ada-002":
        model_embedding = OpenAIEmbeddings(
            model = model_embedding_name,
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )
    else:
        model_embedding = HuggingFaceEmbeddings(
            model_name = model_embedding_name,
            model_kwargs = model_kwargs,
            encode_kwargs = encode_kwargs
        )
    return model_embedding
class Embedding():
    def __init__(self, model_embedding_name,device, batch_size):
        self.model_embedding = get_embedding_model(
            model_embedding_name=model_embedding_name,
            model_kwargs = {'device':device},
            encode_kwargs = {'device':device, 'batch_size':batch_size}
            )
    def __call__(self, *args, **kwargs):
        return self.model_embedding
