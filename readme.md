# 🚀 InsightNest

> An AI-powered, multi-source document intelligence and conversational analytics platform powered by Retrieval-Augmented Generation (RAG).

---

## 📌 What is InsightNest?

**InsightNest** is an AI-first platform designed to seamlessly accept content from various sources (like YouTube, Google Drive, LinkedIn, and more), analyze the data, extract meaningful insights, and provide natural language interaction via a powerful RAG-based chatbot.

It's more than just document parsing—**InsightNest** delivers:

- 📄 **Universal Document Support** – Upload files from any platform or format.
- 🧠 **Smart RAG Chatbot** – Ask questions, extract insights, and interact with your documents naturally.
- 📊 **Automated Data Analysis (EDA)** – Get instant exploratory data analysis on your structured files.
- 🔐 **Nested RAG for Private Info** – For sensitive data, a secondary RAG layer requires authentication for secure access.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 📂 **Multi-source Upload** | Upload from YouTube (transcripts), Drive (docs/spreadsheets), LinkedIn (profiles, posts), and local files |
| 🤖 **RAG Chatbot Interface** | Conversational interface for real-time querying and data interaction |
| 📊 **EDA Module** | Automated visualization and summary of uploaded datasets |
| 🔐 **RAG-in-RAG Security Layer** | Controlled access to sensitive data, requiring additional user verification |
| 🧩 **Extensible Architecture** | Easily plug in new document types, models, or data pipelines |

---

## 🔧 Tech Stack

- **Backend**: Python, FastAPI
- **AI Engine**: LangChain + Grok + Llama 3 
- **Frontend**: React / Next.js (or your stack)
- **Storage**: PInecone
- **Authentication**: OAuth 2.0 + Role-based RAG access
- **Data Analysis**: Pandas, Seaborn, Plotly

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/insightnest.git
cd insightnest
pip install -r requirements.txt
