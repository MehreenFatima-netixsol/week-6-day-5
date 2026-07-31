# AFL Analyst Pro – AI-Powered AFL Assistant

AFL Analyst Pro is a production-ready AI assistant developed as the **Week 6 Capstone Project**. It combines **LangGraph**, **FastAPI**, **Machine Learning**, and **LLM-powered conversations** to deliver an intelligent assistant capable of answering AFL questions, retrieving historical statistics, predicting match outcomes, and maintaining natural multi-turn conversations.

The assistant is strictly **domain-locked to the Australian Football League (AFL)** and includes guardrails to prevent hallucinations, prompt injection attacks, and off-topic responses.

---

## Features

- General AFL knowledge assistant
- Team and player statistics retrieval
- Match winner prediction
- Top player prediction
- Multi-turn conversational memory
- LangGraph-based workflow orchestration
- Prompt injection and jailbreak protection
- Structured logging and monitoring
- FastAPI REST API
- Optional web-based chat interface

---

## System Architecture

```
                User
                  │
                  ▼
            FastAPI Endpoint
                  │
                  ▼
            LangGraph Workflow
                  │
        ┌─────────┼──────────┐
        │         │          │
        ▼         ▼          ▼
  Guardrails   Intent     Context
               Detection   Memory
        │         │
        └────┬────┘
             ▼
   ┌────────────────────┐
   │ Intent Router      │
   └────────────────────┘
      │      │       │
      ▼      ▼       ▼
 Retrieval Prediction General Knowledge
      │      │       │
      └──┬───┴───────┘
         ▼
   Response Formatter
         │
         ▼
       User
```

---

## Key Capabilities

### General AFL Knowledge

Answer questions about:

- AFL history
- Rules of the game
- Competition structure
- Finals system
- Brownlow Medal
- Coleman Medal
- AFL Draft
- Player positions
- Stadiums
- AFL terminology

---

### Statistics Retrieval

Retrieve information from AFL datasets including:

- Team statistics
- Player statistics
- Match summaries
- AFL ladder
- Season performance
- Head-to-head records
- Recent match information

---

### Prediction Engine

Supports:

- Match winner prediction
- Top player prediction

Every prediction includes the disclaimer:

> **"This is a predicted probability, not a certainty."**

---

### Multi-turn Conversations

The assistant maintains conversation context and understands follow-up questions such as:

- "What about Carlton?"
- "Compare them."
- "Who scored more?"
- "What about their next match?"

---

### Guardrails

The assistant is restricted to AFL-related topics and resists:

- Prompt injection
- Jailbreak attempts
- System prompt requests
- Off-topic conversations
- Hallucinated statistics

---

## Tech Stack

- Python
- LangGraph
- FastAPI
- Scikit-learn
- Pandas
- Pydantic
- OpenAI SDK (Netixsol Endpoint)
- Uvicorn
- HTML/CSS/JavaScript (Optional UI)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/afl-analyst-pro.git
```

Navigate to the project directory:

```bash
cd afl-analyst-pro
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Start the FastAPI server:

```bash
python -m uvicorn api:app --reload --port 8000
```

Open the API documentation:

```
http://127.0.0.1:8000/docs
```

(Optional) Launch the chat interface:

```bash
streamlit run ui/app.py
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /chat` | Chat with the AFL assistant |
| `GET /health` | Health check |
| `GET /conversations/{id}` | Retrieve conversation history |

---

## Evaluation

The assistant was evaluated across multiple categories:

- General AFL knowledge
- Retrieval accuracy
- Match prediction
- Top player prediction
- Multi-turn conversations
- Clarification handling
- Guardrail effectiveness
- Prompt injection resistance
- API reliability

Performance was measured using:

- Response accuracy
- Pass rate
- Tool success rate
- Prediction sanity
- Conversation coherence
- Average latency

---

## Monitoring

The application records:

- Query
- Intent detected
- Tools executed
- API latency
- Token usage
- Prediction metadata
- Conversation trace
- Tool failures

These logs support monitoring, debugging, and future model improvements.

---

## Known Limitations

- Predictions depend on historical training data.
- Retrieval is limited to the available AFL datasets.
- General knowledge requires an active LLM endpoint.
- Conversation history is stored in memory.
- Real-time AFL statistics are not currently supported.

---

## Future Improvements

- Live AFL API integration
- Persistent conversation storage
- Weekly model retraining
- Cloud deployment
- Docker support
- User authentication
- Real-time analytics dashboard
- CI/CD pipeline

---

## Demo Workflow

1. Ask a general AFL question.
2. Retrieve team statistics.
3. Predict a match winner.
4. Predict the top player.
5. Continue with a follow-up question.
6. Trigger a clarification request.
7. Test an off-topic question.
8. Attempt a prompt injection.
9. Review conversation logs.

---

## Author

**Mehreen Fatima**

## License

This project was developed for educational purposes as part of the **Week 6 Capstone Project**. It demonstrates the integration of LangGraph, FastAPI, Retrieval-Augmented Generation (RAG), Machine Learning, and Responsible AI principles to build a production-style AFL assistant.
