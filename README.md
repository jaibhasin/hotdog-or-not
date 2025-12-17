# 🌭 Hot Dog or Not Hot Dog

A tribute to **Jian Yang's SeeFood app** from HBO's Silicon Valley.

> "What would you say if I told you there is an app on the market that tells you if you have a hot dog or not a hot dog?"

![Silicon Valley](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExdGF1NnlsNThjY2FlNTk1d3lqZGlhcThlcnFoamM2bWJzYjB6dDR4ciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0Iy9iqThC2ueLTkA/giphy.gif)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load images from `dataset/train/` and `dataset/test/`
- Train a MobileNetV2-based classifier
- Save the model as `hotdog_model.keras`

**Expected training time:** ~10-15 minutes on CPU, ~2-3 minutes with GPU

### 3. Start the Server

```bash
python server.py
```

Then open http://localhost:8000 in your browser!

## 📁 Project Structure

```
├── dataset/
│   ├── train/
│   │   ├── hotdog/      # ~2,600 hot dog images
│   │   └── nothotdog/   # ~2,600 non-hot dog images
│   └── test/
│       ├── hotdog/      # ~400 hot dog images
│       └── nothotdog/   # ~250 non-hot dog images
├── train_model.py       # Model training script
├── server.py            # FastAPI backend
├── index.html           # Frontend HTML
├── styles.css           # Premium styling
├── app.js               # Frontend logic
├── requirements.txt     # Python dependencies
└── hotdog_model.keras   # Trained model (after training)
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | TensorFlow/Keras |
| Base Model | MobileNetV2 (transfer learning) |
| Backend | FastAPI |
| Frontend | HTML/CSS/JS |

## 📊 Expected Results

The model should achieve **>90% accuracy** on the test set after training.

## 🎨 Features

- **Drag & Drop** - Drop images directly onto the upload zone
- **Real-time Preview** - See your image before analysis
- **Dramatic Reveal** - Results displayed with style
- **Dark Theme** - Neon green accents, glassmorphism effects
- **Mobile Responsive** - Works on all screen sizes

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the frontend |
| `/predict` | POST | Analyze an image |
| `/health` | GET | Health check |

### Example API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@hotdog.jpg"
```

Response:
```json
{
  "prediction": "Hot Dog",
  "confidence": 0.95,
  "raw_score": 0.05,
  "is_hotdog": true
}
```

## 🙏 Credits

- Inspired by HBO's **Silicon Valley** S4E4
- Character: **Jian Yang** (played by Jimmy O. Yang)
- Original app concept: **SeeFood** (Not Hotdog)

---

*"It's like Shazam... for food!"* 🌭
