import mlflow
import os
from datetime import datetime
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "QuantumFinance-CreditScore"

try:
    # Obter o experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado")
    
    # Buscar a última execução
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                            order_by=["start_time DESC"], 
                            max_results=1)
    
    if runs.empty:
        raise ValueError("Nenhuma execução encontrada para o experimento")
    
    run = runs.iloc[0]
    accuracy = run["metrics.accuracy"]
    run_id = run["run_id"]
    model_uri = f"runs:/{run_id}/model"
    
    # Converter o timestamp corretamente
    start_time = pd.to_datetime(run["start_time"])
    formatted_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Criar relatório
    report_md = f"""# 📄 MLflow Experimento - QuantumFinance

**Data:** {formatted_time}  
**Accuracy:** {accuracy:.4f}  
**Run ID:** {run_id}  
**Modelo:** `{model_uri}`

🧪 Relatório gerado automaticamente via GitHub Actions.
"""
    
    # Criar diretório se não existir
    os.makedirs("reports", exist_ok=True)
    
    # Salvar relatório
    with open("reports/mlflow_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print("Relatório gerado com sucesso em reports/mlflow_report.md")

except Exception as e:
    print(f"Erro ao gerar relatório: {str(e)}")
    raise