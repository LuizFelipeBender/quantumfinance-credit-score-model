import time
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Timer para medir tempo de execução
start_time = time.time()
print("🚀 Iniciando pipeline...")

# Configuração do MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("QuantumFinance-CreditScore")

# Carregando dados
print("📥 Lendo CSV...")
df = pd.read_csv("data/raw/train.csv", low_memory=False)
df = df[df["Credit_Score"].isin(["Good", "Standard", "Poor"])]

X = df.drop(columns=["Credit_Score", "ID", "Customer_ID", "Name", "SSN", "Month"])
y = df["Credit_Score"]

# Tratar coluna 'Type_of_Loan' que tem listas de empréstimos
if "Type_of_Loan" in X.columns:
    X["Type_of_Loan"] = X["Type_of_Loan"].fillna("Unknown").astype(str).apply(lambda x: x.split(",")[0].strip())

# Identificar colunas
cat_features = X.select_dtypes(include="object").columns.tolist()
num_features = X.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns.tolist()

print("🧹 Pré-processando features:")
print(f"  - Numéricas: {len(num_features)}")
print(f"  - Categóricas: {len(cat_features)}")

# Reduzir cardinalidade de colunas categóricas
for col in cat_features:
    top = X[col].value_counts().nlargest(20).index
    X[col] = X[col].where(X[col].isin(top), other="Rare")

# Pré-processadores
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

# Pipeline final com Random Forest (pode aumentar os estimators depois)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=20, random_state=42))
])

# Split
print("✂️ Separando treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Treinamento
print("🧠 Treinando modelo...")
with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Após logar no MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(pipeline, "model")

    print(f"\n🎯 Acurácia: {acc:.4f}\n")
    print("📊 Classification Report:\n", classification_report(y_test, preds))


    output_path = "models"
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(pipeline, os.path.join(output_path, "model.pkl"))
    print(f"💾 Modelo salvo localmente em: {os.path.join(output_path, 'model.pkl')}")

    print("\n✅ Pipeline finalizada com sucesso.")
