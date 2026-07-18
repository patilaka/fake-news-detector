# VERITAS.AI — Fake News Detection System

An AI-powered web app that predicts whether a news headline/article is **Fake** or **Real**, using a fine-tuned DistilBERT transformer model. Built end-to-end: frontend, backend, ML model, and database — fully deployed and live.

## 🚀 Live Demo

- **Frontend:** [Coming soon — currently improving model accuracy]
- **GitHub:** https://github.com/Dhanashri-Adawade/fake-news-detector

*(Note: Live link temporarily not shared while I improve model accuracy on tricky real-world headlines. Demo video below shows the app working.)*

## 📹 Demo Video

https://drive.google.com/file/d/1f7fCID-QDkD9mXk4Wb7WLHuZTIsAX2t1/view?usp=sharing

## 🧠 How It Works

1. User enters a news headline or article text
2. Text is sent to the FastAPI backend
3. Backend runs the text through a fine-tuned DistilBERT model
4. Model returns a prediction (`Fake` / `Real`) with a confidence score
5. Prediction is saved to database and shown in history

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React (Vite), Tailwind CSS v4 |
| Backend | FastAPI, Python 3.12 |
| ML Model | DistilBERT (Hugging Face Transformers, PyTorch) |
| Database | MySQL (hosted on Aiven) |
| Model Hosting | Hugging Face Hub |
| Backend Hosting | Hugging Face Spaces (Docker) |
| Frontend Hosting | Vercel |

## 📁 Project Structure

```
fake-news-detector/
├── frontend/          # React app (Vite + Tailwind)
│   └── src/
├── backend/           # FastAPI app
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   └── database.py
├── model/             # Trained DistilBERT model files
└── training/          # Model training notebooks/scripts
```

## ⚙️ Running Locally

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

You'll need a `.env` file in both `backend/` and `frontend/` — see `.env.example` for required variables (not committed, for security).

## 📊 Model Performance (Honest Assessment)

The model performs strongly on:
- Obvious clickbait/hoax headlines (83–99% confidence)
- Calm, factual government/policy announcements (98–99% confidence)

**Known limitations (currently being improved):**
- Very short inputs (under 5 words) → lower confidence
- Some real crime/investigation reports occasionally misclassified
- Some space/science news occasionally misclassified

I'm actively working on retraining with more topic-balanced data to fix these blind spots. Full transparency — this is a real learning project, not a polished final product.

## 👩‍💻 Author

**Dhanashri Adawade**
Computer Engineering student | AI/ML enthusiast
www.linkedin.com/in/dhanashri-adawade-32746b337

