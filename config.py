from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API=os.getenv("GROQ_API")
def apply_settings():
    # 🔹 Configure LLM (Groq-hosted Meta Llama 4 Scout)
    Settings.llm = Groq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=GROQ_API
    )

    # 🔹 Configure Embeddings (HuggingFace)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 🔹 Optional: Configure Node Parser (chunking)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    # 🔹 Optional: Context parameters
    Settings.num_output = 512
    Settings.context_window = 3900
