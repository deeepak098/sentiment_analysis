1) Project structure
   sentiment_flask_app/
│
├── app.py
├── requirements.txt
├── sentiment_pipeline.joblib          
├── templates/
│   └── index.html
└── static/
    └── style.css


   Step 1: Install joblib inside your venv
   pip install joblib


activate venv
venv\Scripts\Activate.ps1

Step 2: Also install scikit-learn (needed for loading pipelines)
pip install scikit-learn


Install Flask
pip install Flask


run
python -u app.py

