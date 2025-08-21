Como rodar 

0) Pré-requisitos

Git (para clonar)

Conda (Anaconda/Miniconda/Miniforge) ou Python 3.9 + pip

1) Clonar o repositório

Abra um terminal e digite:

git clone <URL_DO_SEU_REPOSITORIO> monitoramento-ml
cd monitoramento-ml

2) Preparar o ambiente
Opção A — Conda (recomendado)
conda env create -f environment.yml
conda activate neurotech-compat

Opção B — pip (se você não usa Conda)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r monitoring/requirements.txt
# garantir compatibilidade do stack científico:
pip install "numpy==1.21.6" "scipy==1.7.3" "scikit-learn==1.0.2" "pandas==1.4.4"
pip install fastapi "uvicorn[standard]" pydantic joblib requests jupyter

3) Subir a API

No terminal com o ambiente ativo, na raiz do repositório, digite:

uvicorn monitoring.app.main:app --reload


Aguarde a mensagem: Uvicorn running on http://127.0.0.1:8000

Não feche este terminal.

Porta ocupada? rode com outra:
uvicorn monitoring.app.main:app --reload --port 8001

4) Testar pelo Swagger

Abra o navegador em http://127.0.0.1:8000/docs
.

4.1) POST /performance

Clique em POST /performance → Try it out.

Abra monitoring/batch_records.json e copie o conteúdo.

Cole no Request body:

Se o arquivo começa com [ (lista), embrulhe assim:

{ "records": AQUI_VAI_O_CONTEUDO }


Se o arquivo já é { "records": [...] }, cole do jeito que está.

Execute → espere Code 200 e veja:

volumetry_by_month (contagem por YYYY-MM)

roc_auc, n_used_for_auc, n_total

4.2) POST /adherence

Clique em POST /adherence → Try it out.

Cole um por vez (caminhos relativos, portáteis):

Treino (train.gz)

{ "dataset_path": "./challenge-data-scientist/datasets/credit_01/train.gz" }


OOT (oot.gz)

{ "dataset_path": "./challenge-data-scientist/datasets/credit_01/oot.gz" }


Execute → espere Code 200 com ks_statistic, p_value, n_dataset, n_reference.

(Opcional) Testar pelo terminal (curl)
# adherence (train)
curl -X POST "http://127.0.0.1:8000/adherence" \
  -H "Content-Type: application/json" \
  -d '{"dataset_path":"./challenge-data-scientist/datasets/credit_01/train.gz"}'

# adherence (oot)
curl -X POST "http://127.0.0.1:8000/adherence" \
  -H "Content-Type: application/json" \
  -d '{"dataset_path":"./challenge-data-scientist/datasets/credit_01/oot.gz"}'

Notebook de validação (entregável)

Abra outro terminal (deixe a API rodando no anterior).

Ative o ambiente e rode:

jupyter lab


Abra monitoring_demo.ipynb.

Execute as células que chamam:

/performance com batch_records.json;

/adherence com train.gz e com oot.gz.

Salve o notebook com as saídas (STATUS 200 + JSON) visíveis.
(Opcional) Exportar em HTML: File → Export Notebook As → HTML.

Dúvidas comuns

JSON inválido no /performance: se o arquivo é uma lista [...], embrulhe como { "records": [...] }.

Caminho do dataset: use caminho relativo (ex.: ./challenge-data-scientist/...) para ser portátil.

p-valor muito pequeno: normal com bases grandes — interprete pelo KS.

AUC não calculou: verifique se há coluna TARGET no lote.

Decisões técnicas (resumo)

Alinhamento de colunas: usa feature_names_in_ do modelo; senão, usa o schema do test.gz.

Numéricas: to_numeric + imputação pela mediana da referência.

Categóricas: ausentes e categorias desconhecidas viram a moda da referência (evita erro de encoder).

Pontuação: tenta predict_proba; se não houver, decision_function; se não, predict (serve para KS).

Modelo cacheado em memória (melhora latência).

Licença / uso

Projeto destinado à avaliação técnica. Dados e modelo são de uso exclusivo do desafio.

TL;DR (super curto)
git clone <URL_DO_REPO> monitoramento-ml
cd monitoramento-ml
conda env create -f environment.yml && conda activate neurotech-compat
uvicorn monitoring.app.main:app --reload
# → http://127.0.0.1:8000/docs