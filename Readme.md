#Langchain RAG App

✅ Features:
🔧 Organized LangChain architecture
📄 Modular components: LLMs, chains, vector DBs, loaders
🧠 RAG + Chatbot ready
🌐 FastAPI-ready (optional)
📦 Environment config support
🧪 Testable modules

langchain-rag-app/
├── app/
│   ├── chains/
│   │   └── chat_chain.py
│   ├── retriever/
│   │   └── vectorstore.py
│   ├── loaders/
│   │   └── load_docs.py
│   ├── llms/
│   │   └── openai_llm.py
│   ├── agents/
│   │   └── simple_agent.py
│   ├── graph/
│   │   └── rag_graph.py
│   └── config.py
│
├── data/
│   └── sample_docs/
│       └── cars.txt
│
├── tests/
│   └── test_chat.py
│
├── .env
├── main.py
├── requirements.txt
└── README.md
