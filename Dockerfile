FROM python:3.12-slim

WORKDIR /app

# CPU-only torch first, from PyTorch's CPU index. sentence-transformers would
# otherwise pull the default CUDA build (~2.5GB of GPU runtime this container
# can never use -- reranking here is CPU-bound and the GPU is reserved for
# Ollama's models on this box's shared/unified memory).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir mcp "psycopg[binary]" pgvector sentence-transformers

# Bake the cross-encoder weights into the image so the first real query does not
# pay a HuggingFace download, and so the container needs no outbound network at
# runtime beyond Ollama and Postgres.
#
# Deliberately BEFORE the COPY: this layer is ~90MB of model weights that never
# change, while the source below changes constantly. With the order reversed
# every one-line code edit invalidated this layer and re-downloaded the model.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY embed_client.py rerank.py search.py kg_ontology.py kg_query.py library_mcp_server.py ./

ENV PORT=8080
# /api/embed (not the legacy /api/embeddings) -- only this endpoint honours the
# top-level "dimensions" parameter that truncates qwen3-embedding's native 1024
# dims to the 768 the rag_library_chunks vector column and HNSW index expect.
ENV OLLAMA_URL=http://host.containers.internal:11434/api/embed
ENV EMBED_MODEL=qwen3-embedding:0.6b
ENV EMBED_DIMENSIONS=768
ENV DB_HOST=host.containers.internal

EXPOSE 8080

CMD ["python", "library_mcp_server.py"]
