from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import init_db

# Import routers
from routers import inference, analytics, tip

app = FastAPI(
    title="Dual-Energy X-Ray Security Scanner API",
    description="Backend services for deep property-based threat detection and Beer-Lambert TIP simulation",
    version="2.0.0" # Major bump for professional backend features
)

# Enable CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify front-end domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Include Routers
app.include_router(inference.router)
app.include_router(analytics.router)
app.include_router(tip.router)

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy", 
        "engine_mode": "Deep Learning" if inference.api_engine is not None else "Computer Vision (Fallback)"
    }
