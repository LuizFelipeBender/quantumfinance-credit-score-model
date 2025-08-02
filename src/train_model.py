import os
import time
import joblib
import requests
import shutil
import glob
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Timer
start_time = time.time()
print("🚀 Iniciando pipeline...")

# Configuração do MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "QuantumFinance-CreditScore")
API_DEPLOY_HOOK = os.getenv("API_DEPLOY_HOOK", "http://localhost:8000/trigger-deploy")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Leitura dos dados
print("📥 Lendo CSV...")
df = pd.read_csv("data/raw/train.csv", low_memory=False)
df = df[df["Credit_Score"].isin(["Good", "Standard", "Poor"])]
X = df.drop(columns=["Credit_Score", "ID", "Customer_ID", "Name", "SSN", "Month"])
y = df["Credit_Score"]

# Ajuste da coluna Type_of_Loan
if "Type_of_Loan" in X.columns:
    X["Type_of_Loan"] = X["Type_of_Loan"].fillna("Unknown").astype(str).apply(lambda x: x.split(",")[0].strip())

# Identificação de colunas
cat_features = X.select_dtypes(include="object").columns.tolist()
num_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Redução de cardinalidade
for col in cat_features:
    top = X[col].value_counts().nlargest(20).index
    X[col] = X[col].where(X[col].isin(top), other="Rare")

# Pipelines
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=20, random_state=42))
])

# Split e treinamento
print("✂️ Split treino/teste...")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print("🧠 Treinando modelo...")
with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(pipeline, "model")

    model_name = "quantumfinance-credit-score-model"
    result = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)

    # Espera a transição de estágio e define como 'Production'
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"📦 Modelo versionado como {model_name}, versão {result.version} (Production)")
    print(f"\n🎯 Acurácia: {acc:.4f}")
    print("📊 Classification Report:\n", classification_report(y_test, preds))

    # Salva localmente com versionamento
    os.makedirs("models", exist_ok=True)
    versioned_path = f"models/model_v{result.version}.pkl"
    latest_path = "models/model_latest.pkl"

    joblib.dump(pipeline, versioned_path)
    print(f"💾 Modelo salvo como {versioned_path}")

    shutil.copy(versioned_path, latest_path)
    print(f"📋 Cópia salva como {latest_path}")

    # Mantém apenas os últimos 3 modelos locais
    model_files = sorted(
        glob.glob("models/model_v*.pkl"),
        key=os.path.getmtime,
        reverse=True
    )

    for old_model in model_files[3:]:
        os.remove(old_model)
        print(f"🗑️ Modelo antigo removido: {old_model}")

    # Trigger de atualização da API
    try:
        response = requests.post(API_DEPLOY_HOOK)
        print(f"🔔 API notificada com status {response.status_code}")
    except Exception as e:
        print(f"⚠️ Erro ao notificar API: {e}")

print("✅ Pipeline finalizada em %.2fs" % (time.time() - start_time))
