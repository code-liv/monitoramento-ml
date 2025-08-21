from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints.performance import router as performance_router
from app.api.endpoints.aderencia import router as aderencia_router

app = FastAPI(
    title="Neurotech - Monitoring API",
    description="API para performance/volumetria e aderência (KS) do modelo",
    version="1.0.0",
)

# CORS simples para facilitar testes locais
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "docs": "http://127.0.0.1:8000/docs"}

# inclui os dois conjuntos de rotas
app.include_router(performance_router, tags=["performance"])
app.include_router(aderencia_router, tags=["adherence"])
