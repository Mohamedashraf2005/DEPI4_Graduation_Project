# 🚦 Smart-Traffic & Road Guard — Raqib

> **An AI-powered platform for monitoring and securing Egyptian roads.**

---

## 📌 Overview

**Smart-Traffic & Road Guard (Raqib)** is an AI-powered infrastructure monitoring and road safety platform. It automates hazard detection, generates real-time incident reports, and predicts high-risk zones — shifting road maintenance from **reactive** to **proactive**.

The platform serves two portals:
- 🏛️ **Authority Portal** — for government engineers and fleet managers
- 👤 **Citizen Portal** — for public road hazard reporting

---

## ✨ Key Features

- **Computer Vision Detection:** Autonomously identifies potholes, cracks, faulty traffic signals, and active accidents.
- **Automated NLP Reporting:** Instantly generates precise technical reports with hazard type and severity.
- **Predictive Risk Analytics:** ML models analyze structural and environmental factors to flag potential accident zones before incidents occur.
- **Resource Optimization:** Data-driven insights streamline dispatch, cut maintenance costs, and improve emergency response times.

---

## 🌍 Impact

- Reduces traffic fatalities and enhances road infrastructure safety
- Enables surgical precision in maintenance and emergency dispatch

---

## 🧠 AI Services

### 1. Computer Vision — Pothole & Hazard Detector

Detects road damage in real-time from images or video streams using YOLOv8.

| Class | Description |
|---|---|
| `longitudinal_crack` | Cracks along the road direction |
| `transverse_crack` | Cracks across the road |
| `alligator_crack` | Interconnected crack patterns |
| `pothole` | Road surface holes |
| `unknown` | Other damage types |

**Model specs:**
- Architecture: YOLOv8m
- Dataset: RDD-2022 (26,869 training images from 5 countries)
- Input size: 640 x 640
- Confidence threshold: 0.35

### 2. NLP — Automated Incident Reporter (RAG)

Converts detection metadata and voice notes into formal technical reports using **Retrieval-Augmented Generation**.

**How it works:**
1. CV model detects a hazard and extracts metadata (type, location, severity)
2. RAG system retrieves relevant context from the road knowledge base (ChromaDB)
3. LLM generates a formal technical report in Arabic or English
4. Report is sent automatically to the responsible authority

**Components:**
- `ingest.py` — builds the vector knowledge base from road damage documents
- `query.py` — retrieves context and generates incident reports
- `app.py` — FastAPI endpoint connecting CV output to RAG pipeline
- Vector DB: ChromaDB
- Embeddings: Multilingual Sentence Transformers

**Sample RAG output:**
```
Question: Which roads have damage?
Answer: Road damage detected on Cairo-Alexandria highway.
Sources: reports/sample.txt
```

### 3. Machine Learning — Risk & Maintenance Predictor

Predicts road deterioration timelines and accident risk scores using historical detection data.

- Forecasts road condition: "This road will deteriorate completely within 3 months if not repaired"
- Risk scoring per road segment based on traffic density and defect count
- Cost-benefit analysis for maintenance prioritization

---

## 🗂️ Repository Structure

```
DEPI4_Graduation_Project/
│
├── Traffic Sign/                       # Traffic sign defect detection model
│
├── YOLO_Car_Accident_Detection/        # Car accident detection model
│
├── road-guard-rag/                     # RAG system for report generation
│   ├── app.py                          # FastAPI app
│   ├── ingest.py                       # Build vector knowledge base
│   ├── query.py                        # Query & generate reports
│   └── reports/                        # Sample incident reports
│       └── sample.txt
│
├── Raqib_Pothole_Detection.ipynb       # Pothole detection training notebook
├── metadata.json                       # Model metadata & class info
└── README.md
```

---

## 🤖 Pothole Detection Model

| Property | Value |
|---|---|
| Architecture | YOLOv8m |
| Input Size | 640 x 640 |
| Classes | 5 |
| Confidence Threshold | 0.35 |
| IoU Threshold | 0.45 |
| Training Dataset | RDD-2022 |
| Training Images | 26,869 |
| Validation Images | 5,758 |

### Download Model Weights

> `best.pt` is hosted on Google Drive due to GitHub's 100MB file size limit.

**[Download best.pt — Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)**

After downloading, place it at:
```
models/best.pt
```

### Usage

```python
from ultralytics import YOLO

model = YOLO('models/best.pt')
results = model.predict(
    source='road_image.jpg',
    conf=0.35,
    iou=0.45
)
results[0].show()
```

---

## 🚀 RAG System Setup

```bash
# 1. Clone the repo
git clone https://github.com/Mohamedashraf2005/DEPI4_Graduation_Project.git
cd DEPI4_Graduation_Project/road-guard-rag

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key to .env
# ANTHROPIC_API_KEY=your_key_here

# 5. Build knowledge base
python ingest.py

# 6. Run a query
python query.py
```

---

## 📓 Training the Pothole Model on Kaggle

1. Upload `Raqib_Pothole_Detection.ipynb` to [Kaggle](https://kaggle.com)
2. Add datasets:
   - `aliabdelmenam/rdd-2022`
   - `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`
3. Set Accelerator to **GPU T4 x2**
4. Click **Save Version → Save & Run All**
5. Download `best.pt` from the Output tab after training completes

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers (multilingual) |
| Backend | FastAPI |
| Frontend | React.js |
| Training Platform | Kaggle (GPU T4 x2) |

---

## 👥 Team — DEPI4 Graduation Project

| Role | Responsibility |
|---|---|
| CV Engineer | Pothole and hazard detection models |
| NLP / RAG Engineer | RAG system and automated report generation |
| ML Engineer | Risk prediction and maintenance forecasting |
| Frontend Engineer | React dashboard — Raqib UI |
| Backend Engineer | FastAPI and model serving |

---

## 📄 License

This project is developed as a graduation project for the **DEPI4 Program**.

---

<div align="center">
  <br>
  <b>Every hazard we detect early is an accident that never happens — and a life kept safe.</b>
  <br><br>
  <i>Smart-Traffic & Road Guard | DEPI4 Graduation Project</i>
</div>
