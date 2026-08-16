📜 VerilyAI

A retrieval-grounded knowledge assistant that never guesses — every answer is traced back to the exact passage it came from.

VerilyAI lets you turn any set of websites or PDFs into a chatbot that answers only from that material, with inline citations you can click open and verify. Unlike a generic AI chatbot, it doesn't rely on what the model "remembers" from training — it searches your actual documents first, and shows you the receipts.

Why VerilyAI is different

Most chatbots answer from memory and can confidently make things up with no way to check. VerilyAI is built around three ideas instead:

Grounded by default — answers are generated only from the passages retrieved from your own sources, not the model's general training data.
Every claim is cited — answers include inline markers like [2] pointing to the exact chunk of text that backs it up, expandable right in the chat.
Transparent about its limits — in Strict mode, if your documents don't cover something, it says so instead of guessing. In Hybrid mode, anything outside your documents is clearly labelled "Outside the knowledge base" so sourced and unsourced claims are never blended together.
Features
🌐 Website ingestion — paste any URL and it's read, cleaned, and indexed
📄 PDF ingestion — text-based PDFs, parsed page by page
🔍 Citation-backed answers — click any citation to see the original passage it came from
🧠 Conversation memory — follow-up questions like "what about pricing?" are automatically resolved against chat history
🔒 Strict / Hybrid grounding modes — choose whether the assistant can ever step outside your documents
🗂️ Private per-session workspace — your uploaded sources are isolated to your own browser session
♻️ Smart deduplication — re-uploading the same source skips re-embedding automatically (saves API cost)
📤 Transcript export — download any conversation as Markdown or JSON
Tech stack
Layer	Tool
UI / app framework	Streamlit
LLM + embeddings	Google Gemini (gemini-3-flash-preview, gemini-embedding-001) via langchain-google-genai
Orchestration	LangChain (langchain-core, langchain-text-splitters)
Vector store	ChromaDB, via langchain-chroma
PDF parsing	pypdf
Web ingestion	requests + beautifulsoup4

The entire app — config, ingestion, chunking, embeddings, vector store, RAG pipeline, transcript export, and UI — lives in a single app.py. No package folders, no local imports. Two files is the whole project.

Project structure
VerilyAI/
├── app.py              # everything — the whole app
└── requirements.txt    # pinned, verified-together dependencies
Setup
1. Clone and install
bash
git clone https://github.com/<your-username>/VerilyAI.git
cd VerilyAI
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Add your Gemini API key

Get a free key at aistudio.google.com/apikey.

Create .streamlit/secrets.toml in the project root:

toml
GOOGLE_API_KEY = "your-key-here"

(Alternatively, export it as an environment variable: GOOGLE_API_KEY or GEMINI_API_KEY.)

3. Run it
bash
streamlit run app.py

Open the local URL Streamlit prints (usually http://localhost:8501).

Deploying on Streamlit Community Cloud
Push app.py and requirements.txt to the root of a GitHub repo (no subfolders)
Go to share.streamlit.io → New app
Select your repo, branch, and set Main file path to app.py
Under Advanced settings → Secrets, add:
toml
   GOOGLE_API_KEY = "your-key-here"
Deploy
How it works
Ingest — a URL or PDF is fetched, cleaned, and split into sections
Chunk — sections are split into overlapping passages sized for retrieval
Embed — each passage is embedded with Gemini's embedding model and cached to disk (so re-runs don't re-pay for unchanged content)
Store — embeddings are written to a per-session ChromaDB collection
Retrieve — on each question, the most relevant passages are pulled from the vector store, filtered by a relevance floor
Generate — Gemini answers using only the retrieved passages (Strict) or those passages plus clearly-labelled general knowledge (Hybrid), citing every claim
Grounding modes
Mode	Behavior
Strict	Answers only from your ingested sources. If the answer isn't in your documents, it says so — no guessing.
Hybrid	Prefers your documents. Falls back to general knowledge only when needed, and always labels that part > **Outside the knowledge base:** so it's never confused with sourced fact.
License

Add your preferred license here (e.g. MIT).

Built as part of an AI, Automation & Security Engineering internship project.
