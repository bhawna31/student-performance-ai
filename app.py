from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if not file:
        return "No file uploaded"
    
    # Load dataset
    data = pd.read_csv(file)

    if "text" not in data.columns or "label" not in data.columns:
        return f"Error: CSV must contain 'text' and 'label' columns. Found: {list(data.columns)}"

    # Convert text to numeric features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data["text"])

    # Encode labels (e.g. High/Low → 0/1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(data["label"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    reports = {}

    # Logistic Regression
    log_model = LogisticRegression(max_iter=500)
    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_test)
    results["Logistic Regression"] = round(accuracy_score(y_test, log_pred), 3)
    reports["Logistic Regression"] = classification_report(y_test, log_pred, target_names=encoder.classes_)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results["Random Forest"] = round(accuracy_score(y_test, rf_pred), 3)
    reports["Random Forest"] = classification_report(y_test, rf_pred, target_names=encoder.classes_)

    # XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    results["XGBoost"] = round(accuracy_score(y_test, xgb_pred), 3)
    reports["XGBoost"] = classification_report(y_test, xgb_pred, target_names=encoder.classes_)

    # Feature importance (Random Forest)
    importance = pd.DataFrame({
        "feature": vectorizer.get_feature_names_out(),
        "importance": rf_model.feature_importances_
    }).sort_values(by="importance", ascending=False).head(20)

    fig = px.bar(importance, x="feature", y="importance", title="Top 20 Important Words (Random Forest)")
    plot_html = pio.to_html(fig, full_html=False)

    return render_template('results.html', results=results, reports=reports, plot=plot_html)

if __name__ == '__main__':
    app.run(debug=True)