# 🐍 DarkNetDetect Backend

This is the **Backend (FastAPI + Machine Learning)** part of the **DarkNetDetect** project.

It powers the **NLP & ML-based detection system** for identifying drug-related activities on encrypted platforms.

---

## 🚀 Features

- ⚡ FastAPI-based backend (high performance, async support)
- 🤖 Machine Learning model for detecting drug-related slang & keywords
- 📦 Pre-trained NLP pipeline (scikit-learn, RapidFuzz)
- 🔌 API endpoints for integration with frontend
- 🛡️ Data preprocessing & safe anonymization
- 📊 Ready for deployment (Uvicorn / Gunicorn)

---

## 📦 Tech Stack

- 🐍 **Python 3.10+**
- ⚡ **FastAPI** - Modern web framework
- 🔥 **Uvicorn** - ASGI server
- 🤖 **Scikit-learn + Joblib** - ML model training and serialization
- 📝 **RapidFuzz** - Text similarity for slang detection

---

## ⚙️ Prerequisites

Before running this project locally, make sure you have installed:

- [Python 3.10+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/)

### Check versions:
```bash
python --version
pip --version
```

---

## 🛠️ Installation & Setup

Follow these steps to set up and run the backend locally:

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-backend-repo>.git
cd <your-backend-repo>
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

**Activate the virtual environment:**

**Windows (PowerShell):**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run FastAPI Server
```bash
uvicorn main:app --reload
```

Your backend will be live at:
```
http://127.0.0.1:8000
```

The interactive API documentation will be available at:
```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints

### Health Check
```http
GET /
```
**Response:** API running status

### Prediction Endpoint
```http
POST /predict
```

**Request Body:**
```json
{
  "text": "Buy weed using btc"
}
```

**Response:**
```json
{
  "prediction": "Drug-related",
  "confidence": 0.92
}
```

---

## 🔧 Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
flake8 .
```

### Model Training
To retrain the ML model with new data:
```bash
python models/train_model.py
```
---

## 🔗 Related Repositories

- **Frontend Repository**: [https://github.com/hellosamyak/DarkNetDetect]

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

---

## ⚠️ Disclaimer

This tool is designed for educational and research purposes only. Please ensure compliance with local laws and regulations when using this software.

---

## 📞 Contact

For questions or support, please open an issue or contact [jainsamyak0805.com]

---

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing libraries
- FastAPI team for the excellent framework
- Scikit-learn contributors for robust ML tools
