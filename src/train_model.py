import os
import time
import joblib
import requests
import shutil
import glob
import boto3
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Início do pipeline
start_time = time.time()
print("🚀 Iniciando pipeline...")

# Configurações
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "QuantumFinance-CreditScore")
API_DEPLOY_HOOK = os.getenv("API_DEPLOY_HOOK", "http://localhost:8000/trigger-deploy")
S3_BUCKET = "quantumfinance-mlflow-artifacts"
S3_MODELS_PREFIX = "models/"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Leitura e preparação dos dados
print("📥 Lendo CSV...")
df = pd.read_csv("data/raw/train.csv", low_memory=False)
df = df[df["Credit_Score"].isin(["Good", "Standard", "Poor"])]
X = df.drop(columns=["Credit_Score", "ID", "Customer_ID", "Name", "SSN", "Month"])
y = df["Credit_Score"]

if "Type_of_Loan" in X.columns:
    X["Type_of_Loan"] = X["Type_of_Loan"].fillna("Unknown").astype(str).apply(lambda x: x.split(",")[0].strip())

cat_features = X.select_dtypes(include="object").columns.tolist()
num_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

for col in cat_features:
    top = X[col].value_counts().nlargest(20).index
    X[col] = X[col].where(X[col].isin(top), other="Rare")

# Pré-processamento
numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

# Classificador base
base_rf = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42)

# Pipeline com SMOTE e seleção de features
pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("selector", SelectFromModel(base_rf)),
    ("classifier", base_rf)
])

# Split
print("✂️ Split treino/teste...")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Treinamento
print("🧠 Treinando modelo...")
with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(pipeline, "model")

    print(f"🎯 Acurácia: {acc:.4f}")
    print("📊 Classification Report:\n", classification_report(y_test, preds))

    # Matriz de confusão
    cm = confusion_matrix(y_test, preds, labels=["Poor", "Standard", "Good"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Poor", "Standard", "Good"])
    disp.plot()
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Importância das features
    classifier = pipeline.named_steps["classifier"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    selected_mask = pipeline.named_steps["selector"].get_support()
    selected_features = feature_names[selected_mask]
    importances = classifier.feature_importances_
    top_idx = importances.argsort()[::-1][:10]

    plt.figure(figsize=(10, 5))
    plt.barh([selected_features[i] for i in top_idx][::-1], importances[top_idx][::-1])
    plt.title("Top 10 Features Selecionadas")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    # Registro no MLflow
    model_name = "quantumfinance-credit-score-model"
    result = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)

    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )

     # Salvamento e upload
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)

    versioned_path = os.path.join(models_dir, f"model_v{result.version}.pkl")
    latest_path = os.path.join(models_dir, "model_latest.pkl")
    joblib.dump(pipeline, versioned_path)
    shutil.copy(versioned_path, latest_path)
    shutil.copy(latest_path, os.path.join(models_dir, "model.pkl"))  # compatibilidade com testes

    model_files = sorted(glob.glob(os.path.join(models_dir, "model_v*.pkl")), key=os.path.getmtime, reverse=True)
    for old_model in model_files[3:]:
        os.remove(old_model)

    s3 = boto3.client("s3")
    for file_path in [versioned_path, latest_path]:
        key = S3_MODELS_PREFIX + os.path.basename(file_path)
        s3.upload_file(file_path, S3_BUCKET, key)

    # Notificação da API
    try:
        response = requests.post(API_DEPLOY_HOOK)
        print(f"🔔 API notificada com status {response.status_code}")
    except Exception as e:
        print(f"⚠️ Erro ao notificar API: {e}")

print("✅ Pipeline finalizada em %.2fs" % (time.time() - start_time))
