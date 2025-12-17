uv run main.py --rek=REK.txt pancreatic_cancers.json --model-openai=gpt-4o-mini-2024-07-18 --api-key=openai_key.txt results/gpt-4o-mini-without-RAG
uv run main.py --rek=REK.txt pancreatic_cancers.json --model-openai=gpt-4o-mini-2024-07-18 --api-key=openai_key.txt --rag results/gpt-4o-mini-with-RAG
uv run main.py --rek=REK.txt pancreatic_cancers.json --model-ollama=llama3.2-vision:11b results/llama11b-without-RAG
uv run main.py --rek=REK.txt pancreatic_cancers.json --model-ollama=llama3.2-vision:11b --rag results/llama11b-with-RAG
uv run main.py --rek=REK.txt pancreatic_cancers.json --model-ollama=gemma3:27b results/gemma27b-without-RAG
uv run main.py --rek=REK.txt pancreatic_cancers.json --model-ollama=gemma3:27b --rag results/gemma27b-with-RAG