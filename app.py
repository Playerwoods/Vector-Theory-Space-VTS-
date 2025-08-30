from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, uuid, numpy as np, os
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional

class GenerativeTheoryEngine:
    def __init__(self):
        print("🧠 Initializing Generative Theory Engine...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.theoretical_axes = {
            "Authority & Power": ("centralized control", "distributed autonomy"),
            "Legitimacy & Consent": ("popular mandate", "imposed authority"),
            "Institutional Design": ("formal rules", "informal norms"),
            "Collective Action": ("cooperation", "free riding"),
            "Justice & Rights": ("equality", "hierarchy"),
            "Resource Distribution": ("concentration", "redistribution"),
            "Social Order": ("stability", "transformation"),
            "Political Participation": ("active engagement", "passive compliance"),
            "Conflict Resolution": ("negotiation", "coercion"),
            "Democratic Processes": ("transparency", "opacity"),
        }
        
        self.axis_vectors = {}
        for name, (pole_a, pole_b) in self.theoretical_axes.items():
            vec_a = self.encoder.encode(pole_a)
            vec_b = self.encoder.encode(pole_b)
            direction = vec_b - vec_a
            self.axis_vectors[name] = direction / (np.linalg.norm(direction) + 1e-9)
        
        print(f"✅ Engine ready with {len(self.axis_vectors)} generative dimensions")

    def generate_theories(self, seed_concept: str, n_theories: int = 5) -> List[Dict]:
        seed_vector = self.encoder.encode(seed_concept)
        seed_vector = seed_vector / (np.linalg.norm(seed_vector) + 1e-9)
        
        theories = []
        attempts = 0
        
        while len(theories) < n_theories and attempts < n_theories * 4:
            attempts += 1
            candidate_vector = self._explore_theoretical_space(seed_vector)
            scores = self._evaluate_theory_quality(candidate_vector, seed_vector)
            
            if scores["total_quality"] >= 0.6:
                theory = {
                    "id": str(uuid.uuid4())[:8],
                    "description": self._decode_theory_to_language(candidate_vector),
                    "conjectures": self._extract_testable_conjectures(candidate_vector),
                    "scores": scores,
                    "dimensional_profile": self._compute_dimensional_scores(candidate_vector)
                }
                theories.append(theory)
        
        return sorted(theories, key=lambda t: t["scores"]["novelty"], reverse=True)

    def _explore_theoretical_space(self, seed_vector: np.ndarray) -> np.ndarray:
        exploration_vector = np.zeros_like(seed_vector)
        for axis_name, axis_vector in self.axis_vectors.items():
            weight = np.random.normal(0, 0.35)
            exploration_vector += weight * axis_vector
        candidate_vector = seed_vector + 0.4 * exploration_vector
        return candidate_vector / (np.linalg.norm(candidate_vector) + 1e-9)

    def _evaluate_theory_quality(self, theory_vec: np.ndarray, seed_vec: np.ndarray) -> Dict:
        novelty = float(np.linalg.norm(theory_vec - seed_vec))
        dimensional_engagements = [abs(float(np.dot(theory_vec, axis))) for axis in self.axis_vectors.values()]
        coherence = float(np.mean(dimensional_engagements))
        testability = float(np.std(dimensional_engagements))
        generativity = float(sum(1 for e in dimensional_engagements if e > 0.15) / len(dimensional_engagements))
        total_quality = 0.30 * novelty + 0.30 * coherence + 0.25 * testability + 0.15 * generativity
        
        return {
            "novelty": novelty, "coherence": coherence,
            "testability": testability, "generativity": generativity,
            "total_quality": total_quality
        }

    def _decode_theory_to_language(self, theory_vector: np.ndarray) -> str:
        alignments = {axis: float(np.dot(theory_vector, vec)) for axis, vec in self.axis_vectors.items()}
        top_alignments = sorted(alignments.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        description_parts = []
        for axis_name, strength in top_alignments:
            if abs(strength) > 0.2:
                direction = "emphasizes" if strength > 0 else "challenges conventional"
                readable_axis = axis_name.lower().replace(' & ', ' and ')
                description_parts.append(f"{direction} {readable_axis}")
        
        return f"Theory that {', and '.join(description_parts)}." if description_parts else "Theory representing a novel theoretical position."

    def _extract_testable_conjectures(self, theory_vector: np.ndarray) -> List[str]:
        alignments = {axis: float(np.dot(theory_vector, vec)) for axis, vec in self.axis_vectors.items()}
        strong_dimensions = [(axis, strength) for axis, strength in alignments.items() if abs(strength) > 0.25][:4]
        
        conjectures = []
        for axis, strength in strong_dimensions:
            direction = "increases" if strength > 0 else "decreases"
            conjectures.append(f"Higher {axis.lower()} {direction} policy effectiveness under institutional constraints")
        
        if len(strong_dimensions) >= 2:
            dim1, dim2 = strong_dimensions[0][0].lower(), strong_dimensions[1][0].lower()
            conjectures.append(f"Interaction between {dim1} and {dim2} exhibits threshold effects")
        
        return conjectures[:4]

    def _compute_dimensional_scores(self, theory_vector: np.ndarray) -> Dict[str, int]:
        return {
            axis_name: max(0, min(100, int(50 + 40 * np.tanh(float(np.dot(theory_vector, axis_vector))))))
            for axis_name, axis_vector in self.axis_vectors.items()
        }

# FastAPI Application
app = FastAPI(title="Generative Theory Engine", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ENGINE = GenerativeTheoryEngine()
TOKENS = {}  # In-memory storage for MVP

class GenerateRequest(BaseModel):
    seed_concept: str
    n_theories: Optional[int] = 5

@app.post("/purchase-token")
def purchase_token(email: str):
    token = str(uuid.uuid4())
    TOKENS[token] = {"uses": 15, "email": email}
    return {"token": token, "uses_remaining": 15}

@app.post("/generate")
def generate_theories(request: GenerateRequest, token: str):
    if token not in TOKENS or TOKENS[token]["uses"] <= 0:
        raise HTTPException(status_code=402, detail="Invalid token or no uses remaining")
    
    theories = ENGINE.generate_theories(request.seed_concept, request.n_theories)
    TOKENS[token]["uses"] -= 1
    
    return {
        "theories": theories,
        "usage_info": {"uses_remaining": TOKENS[token]["uses"], "total_used": 15 - TOKENS[token]["uses"]}
    }

@app.get("/preview")
def preview_capabilities():
    return {"sample_theories": ENGINE.generate_theories("social capital", 2)}

@app.get("/health")
def health_check():
    return {"status": "healthy", "active_tokens": len(TOKENS)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
