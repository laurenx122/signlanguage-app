from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.gesture.ws_fsl_server import router as fsl_router
from src.stt.stt_http import router as stt_router
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ THIS IS REQUIRED
app.include_router(fsl_router)
app.include_router(stt_router)