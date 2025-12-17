# 🌭 Hot Dog or Not Hot Dog

A tribute to **Jian Yang's SeeFood app** from HBO's Silicon Valley.

> "What would you say if I told you there is an app on the market that tells you if you have a hot dog or not a hot dog?"

![Silicon Valley](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExdGF1NnlsNThjY2FlNTk1d3lqZGlhcThlcnFoamM2bWJzYjB6dDR4ciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0Iy9iqThC2ueLTkA/giphy.gif)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python server.py
```

Then open http://localhost:8000 in your browser!

**Note:** A pre-trained model (`hotdog_model.keras`) is included, so you can start using the app immediately!

### (Optional) Train Your Own Model

Want to train the model yourself? Run:

```bash
python train_model.py
```

This will:
- Load images from `dataset/train/` and `dataset/test/`
- Train a MobileNetV2-based classifier
- Save the model as `hotdog_model.keras`

**Expected training time:** ~10-15 minutes on CPU, ~2-3 minutes with GPU

**Note:** You'll need to download the dataset first (see Dataset Setup below).

## 📁 Project Structure

```
├── train_model.py       # Model training script (optional)
├── server.py            # FastAPI backend
├── index.html           # Frontend HTML
├── styles.css           # Premium styling
├── app.js               # Frontend logic
├── requirements.txt     # Python dependencies
└── hotdog_model.keras   # Pre-trained model (included!)
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | TensorFlow/Keras |
| Base Model | MobileNetV2 (transfer learning) |
| Backend | FastAPI |
| Frontend | HTML/CSS/JS |

## 📊 Model Performance

The included pre-trained model achieves **>90% accuracy** on the test set.

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

## 📦 Dataset Setup (Optional - For Training)

If you want to retrain the model, you'll need the dataset:

1. Download from Kaggle:
   - [Hot Dog Dataset 1](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog) (~1,000 images)
   - [Hot Dog Dataset 2](https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog/data) (~4,800 images)

2. Organize into this structure:
```
dataset/
├── train/
│   ├── hotdog/
│   └── nothotdog/
└── test/
    ├── hotdog/
    └── nothotdog/
```

3. Run `python train_model.py`

## 🙏 Credits

- Inspired by HBO's **Silicon Valley** S4E4
- Character: **Jian Yang** (played by Jimmy O. Yang)
- Original app concept: **SeeFood** (Not Hotdog)

---

*"It's like Shazam... for food!"* 🌭
