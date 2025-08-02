# QuantumFinance Credit Score Model

Modelo simplificado de score de crédito com rastreamento e deploy.

## API

Uma API `FastAPI` permite consumir o modelo treinado.
Para rodar localmente:

```bash
export API_TOKEN=seu-token
uvicorn api.main:app --reload
```

Endpoints principais:

| Método | Rota       | Descrição                          |
| ------ | ---------- | ---------------------------------- |
| `GET`  | `/health`  | Verifica se a API está no ar.      |
| `POST` | `/predict` | Retorna o score de crédito. Exige header `Authorization: Bearer <API_TOKEN>` e está limitado a 5 requisições por minuto. |

A documentação interativa está disponível em `/docs`.

## Aplicação de Exemplo

O diretório `app/` contém um exemplo em [Streamlit](https://streamlit.io) que consome a API:

```bash
export API_TOKEN=seu-token
streamlit run app/streamlit_app.py
```

## Testes

```bash
pip install -r requirements.txt
pytest
```

## Organização

- `data/` – datasets versionados com DVC  
- `notebooks/` – análises exploratórias  
- `src/train_model.py` – treinamento com rastreamento no MLflow  
- `models/` – versionamento local do modelo  
- `api/` – código da API  
- `app/` – exemplo em Streamlit

