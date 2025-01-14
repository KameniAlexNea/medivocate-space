---
title: Medivocate
emoji: 🐢
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Medivocate is an AI-driven platform leveraging Retrieval-Aug
---

# Medivocate

An AI-driven platform empowering users with trustworthy, personalized history guidance to combat misinformation and promote equitable history.

## Follows us [here](https://github.com/KameniAlexNea/medivocate)

* [**Alex Kameni**](https://www.linkedin.com/in/elie-alex-kameni-ngangue/)
* [**Esdras Fandio**](https://www.linkedin.com/in/esdras-fandio/)
* [**Patric Zeufack**](https://www.linkedin.com/in/zeufack-patric-hermann-7a9256143/)

## Project Overview

**Medivocate** is structured for modular development and ease of scalability, as seen in its directory layout:

```
📦 ./
├── 📁 docs/
├── 📁 src/
│   ├── 📁 ocr/
│   ├── 📁 preprocessing/
│   ├── 📁 chunking/
│   ├── 📁 vector_store/
│   ├── 📁 rag_pipeline/
│   ├── 📁 llm_integration/
│   └── 📁 prompt_engineering/
├── 📁 tests/
│   ├── 📁 unit/
│   └── 📁 integration/
├── 📁 examples/
├── 📁 notebooks/
├── 📁 config/
├── 📄 README.md
├── 📄 CONTRIBUTING.md
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 LICENSE
```

### Key Features

1. **Trustworthy Information Access** : Using RAG (Retrieval-Augmented Generation) pipelines to deliver fact-based responses.
2. **Advanced Document Handling** : Leveraging OCR, preprocessing, and chunking for scalable document ingestion.
3. **Integrated Tools** : Supports integration with vector databases (e.g., Chroma), LLMs, and advanced prompt engineering techniques.

### Recommendations for Integration

* **Groq** : Utilize Groq APIs for free-tier LLM support, perfect for prototyping RAG applications.
* **LangChain + LangSmith** : Build and monitor intelligent agents with LangChain and enhance debugging and evaluation using LangSmith.
* **Hugging Face Datasets** : For one-liner dataset loading and preprocessing, supporting efficient ML training pipelines.
* **Search Index** : Include Chroma for robust semantic search capabilities in RAG.

This modular design and extensive integration make Medivocate a powerful tool for historical education and research.