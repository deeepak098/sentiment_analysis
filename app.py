# verbose app.py for debugging startup issues
import os
import sys
import traceback
import joblib
import numpy as np

print("[INFO] app.py starting...", flush=True)
print("[INFO] Python:", sys.version.replace("\n", " "), flush=True)
print("[INFO] CWD:", os.getcwd(), flush=True)
print("[INFO] Listing files in CWD:", flush=True)
for f in os.listdir("."):
    print("  ", f, flush=True)

# candidate model names to try loading
CANDIDATE_PATHS = [
    "sentiment_pipeline.joblib",
    "tfidf_logreg_pipeline.joblib",
    "tfidf_linearsvc_pipeline.joblib",
    "tfidf_nb_pipeline.joblib",
    "../4_models/tfidf_logreg_pipeline.joblib",
    "../4_models/tfidf_linearsvc_pipeline.joblib",
    "../4_models/tfidf_nb_pipeline.joblib"
]

model = None
loaded_path = None

# try loading model files and print detailed errors
for p in CANDIDATE_PATHS:
    try:
        if os.path.exists(p):
            print(f"[INFO] Found candidate model file: {p}", flush=True)
            model = joblib.load(p)
            loaded_path = p
            print(f"[INFO] Successfully loaded model from: {p}", flush=True)
            break
        else:
            print(f"[DEBUG] Not found: {p}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load {p}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

if model is None:
    print("[ERROR] No model loaded. See candidate paths above.", flush=True)
    print("Please place your .joblib pipeline in the app folder and name it 'sentiment_pipeline.joblib',", flush=True)
    # continue — we will not raise here so Flask import errors (if any) are visible
else:
    # show model type summary
    try:
        print("[INFO] Model type:", type(model), flush=True)
        if hasattr(model, "named_steps"):
            print("[INFO] Pipeline steps:", list(model.named_steps.keys()), flush=True)
    except Exception:
        pass

# Now import Flask and build the app inside try/except so errors are visible
try:
    from flask import Flask, render_template, request
    app = Flask(__name__)
    print("[INFO] Imported Flask successfully.", flush=True)

    def predict_sentiment(text):
        if model is None:
            raise RuntimeError("Model not loaded. Aborting prediction.")
        # try pipeline predict
        pred = model.predict([text])[0]
        proba = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([text])[0][1]
            else:
                if hasattr(model, "named_steps"):
                    clf = model.named_steps.get("clf")
                    tfidf = model.named_steps.get("tfidf")
                    if clf is not None and tfidf is not None:
                        X = tfidf.transform([text])
                        if hasattr(clf, "predict_proba"):
                            proba = clf.predict_proba(X)[0][1]
                        elif hasattr(clf, "decision_function"):
                            score = clf.decision_function(X)[0]
                            proba = 1 / (1 + np.exp(-score))
        except Exception as e:
            print("[WARN] probability/score extraction failed:", e, flush=True)
        return int(pred), (float(proba) if proba is not None else None)

    @app.route("/", methods=["GET", "POST"])
    def index():
        prediction = None
        score = None
        error = None
        used_model = loaded_path or "None"
        if request.method == "POST":
            review_text = request.form.get("review", "")
            if not review_text.strip():
                error = "Please enter some text."
            else:
                try:
                    pred, prob = predict_sentiment(review_text)
                    prediction = "Positive" if pred == 1 else "Negative"
                    score = prob
                except Exception as e:
                    error = f"Prediction failed: {type(e).__name__}: {e}"
                    traceback.print_exc()
        return render_template("index.html", prediction=prediction, score=score, error=error, model_path=used_model)

    @app.route("/health")
    def health():
        return {"status": "ok", "model_loaded": loaded_path is not None, "model_path": loaded_path}

    # start the server inside try so we can catch runtime errors
    if __name__ == "__main__":
        host = "127.0.0.1"
        port = 5000
        print(f"[INFO] About to run Flask app on http://{host}:{port}/", flush=True)
        try:
            app.run(host=host, port=port, debug=True)
        except Exception as e:
            print("[ERROR] app.run() failed:", type(e).__name__, e, flush=True)
            traceback.print_exc()

except Exception as e:
    print("[FATAL] Failed to import Flask or initialize app:", type(e).__name__, e, flush=True)
    traceback.print_exc()
    sys.exit(1)
