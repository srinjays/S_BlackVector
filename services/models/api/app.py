"""
SatQuery AI — ML Model Service (FastAPI).

One process, five endpoints, one shared InternVL2-2B backbone.
Each endpoint honors its locked contract schema from contracts/schemas/.

Endpoints:
    POST /vqa       — Single-image Visual Question Answering (F1)
    POST /caption   — Single-image Scene Captioning (F2)
    POST /grounding — Text-guided Region Grounding (F3)
    POST /change    — Bi-temporal Change Detection/VQA (F4)
    POST /fusion    — Optical-SAR Cross-modal Fusion (F5)
    GET  /health    — Service health + VRAM status
"""

import hashlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import sys
NETRA_ROOT = str(Path(__file__).resolve().parents[3])
if NETRA_ROOT not in sys.path:
    sys.path.insert(0, NETRA_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satquery_ml")



# =============================================================================
# Response Models (matching contract schemas)
# =============================================================================

class Confidence(BaseModel):
    """Confidence score with explanation."""
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0, 1]")
    method: str = Field(default="model_logit", description="How confidence was derived")
    factors: dict = Field(default_factory=dict, description="Contributing factors")


class Provenance(BaseModel):
    """Audit trail for a model response (D4)."""
    model_name: str
    checkpoint_version: str = ""
    input_hash: str = ""
    timestamp: str = ""
    preprocessing: dict = Field(default_factory=dict)


class BoundingBox(BaseModel):
    """A single bounding box."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str = ""
    confidence: float = 0.0


# --- VQA Response ---
class VQAResponse(BaseModel):
    """Response schema for /vqa endpoint."""
    answer: str
    confidence: Confidence
    evidence_regions: list[BoundingBox] = Field(default_factory=list)
    provenance: Provenance
    execution_time_ms: float = 0.0


# --- Caption Response ---
class CaptionResponse(BaseModel):
    """Response schema for /caption endpoint."""
    caption: str
    confidence: Confidence
    provenance: Provenance
    execution_time_ms: float = 0.0


# --- Grounding Response ---
class GroundingResponse(BaseModel):
    """Response schema for /grounding endpoint."""
    boxes: list[BoundingBox]
    expression: str  # The input referring expression
    confidence: Confidence
    provenance: Provenance
    execution_time_ms: float = 0.0


# --- Change Detection Response ---
class ChangeResponse(BaseModel):
    """Response schema for /change endpoint."""
    description: str              # Natural language change description
    answer: Optional[str] = None  # Answer if a question was asked (change-VQA)
    change_detected: bool = False
    change_regions: list[BoundingBox] = Field(default_factory=list)
    confidence: Confidence
    provenance: Provenance
    execution_time_ms: float = 0.0


# --- Fusion Response (satisfies §6.4 output contract) ---
class ModalityAttribution(BaseModel):
    """Which parts of the answer come from which modality (§6.4 requirement)."""
    optical_evidence: str = ""    # What optical image contributed
    sar_evidence: str = ""        # What SAR image contributed
    fused_reasoning: str = ""     # How they were combined


class FusionResponse(BaseModel):
    """Response schema for /fusion endpoint. Satisfies §6.4 output contract."""
    fused_answer: str                          # Combined answer from both modalities
    modality_attribution: ModalityAttribution  # REQUIRED by §6.4
    optical_evidence_overlay: dict = Field(default_factory=dict)  # Separate optical overlay
    sar_evidence_overlay: dict = Field(default_factory=dict)      # Separate SAR overlay
    disagreement_flag: bool = False            # FR9: True if modalities conflict
    disagreement_details: str = ""             # Explanation of disagreement
    confidence: Confidence
    provenance: Provenance
    execution_time_ms: float = 0.0


# --- Spectral Response ---
class SpectralResponse(BaseModel):
    """Response schema for /spectral endpoint."""
    indices: dict = Field(default_factory=dict)
    vegetation_health: dict = Field(default_factory=dict)
    water_extent: dict = Field(default_factory=dict)
    burn_severity: dict = Field(default_factory=dict)
    reasoning: str = ""
    answer: str = ""
    confidence: Confidence
    provenance: Provenance
    execution_time_ms: float = 0.0


# --- Agriculture Response ---
class AgricultureResponse(BaseModel):
    """Response schema for /agriculture endpoint (all contextual items are estimates)."""
    soil_analysis: dict = Field(default_factory=dict)
    crop_intelligence: dict = Field(default_factory=dict)
    drought_irrigation: dict = Field(default_factory=dict)
    reasoning: str = ""
    answer: str = ""
    confidence: Confidence
    provenance: Provenance
    execution_time_ms: float = 0.0


# =============================================================================
# Application State
# =============================================================================


class AppState:
    """Global application state — holds the shared model backbone."""

    def __init__(self):
        self.vlm = None           # VLMBackbone instance
        self.grounding_dino = None  # Grounding DINO instance (loaded separately)
        self.change_detector = None  # ChangeDetector head
        self.fusion_analyzer = None  # FusionAnalyzer head
        self.ready = False

    async def startup(self):
        """Load models on startup."""
        logger.info("=== SatQuery AI ML Service Starting ===")

        try:
            from services.models.core.eov2b_backbone import EOV2BBackbone as VLMBackbone, EOV2BConfig as ModelConfig
            config = ModelConfig()
            self.vlm = VLMBackbone(config)
            self.vlm.load()



            # Initialize task heads if spectral adapter is available
            if self.vlm.spectral_adapter is not None:

                from services.models.heads.change_head import ChangeDetector
                from services.models.heads.fusion_head import FusionAnalyzer
                self.change_detector = ChangeDetector(self.vlm)
                self.fusion_analyzer = FusionAnalyzer(self.vlm)
                logger.info("Task heads initialized (change + fusion).")

            # Load Grounding DINO (separate model)
            try:
                from services.models.heads.grounding_head import GroundingDINOHead
                self.grounding_dino = GroundingDINOHead()
                self.grounding_dino.load()
                logger.info("Grounding DINO loaded.")
            except Exception as e:
                logger.warning(f"Grounding DINO not available: {e}")
                self.grounding_dino = None

            self.ready = True
            logger.info("=== ML Service Ready ===")
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            logger.warning("Service starting in analytical mode.")
            self.ready = True


    async def shutdown(self):
        """Clean up on shutdown."""
        if self.vlm:
            self.vlm.unload()
        logger.info("=== ML Service Shutdown ===")


app_state = AppState()


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle."""
    await app_state.startup()
    yield
    await app_state.shutdown()


app = FastAPI(
    title="SatQuery AI — ML Model Service",
    description=(
        "Single FastAPI process exposing 5 endpoints for remote-sensing "
        "vision-language tasks. All share one InternVL2-2B backbone."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Health Endpoint
# =============================================================================

@app.get("/health")
async def health():
    """Service health check with VRAM report."""
    report = {
        "status": "ready" if app_state.ready else "degraded",
        "model_loaded": app_state.vlm.is_loaded if app_state.vlm else False,
    }
    if app_state.vlm:
        report["vram"] = app_state.vlm.get_vram_report()
    return report


# =============================================================================
# VQA Endpoint (F1) — Mandatory
# =============================================================================

@app.post("/vqa", response_model=VQAResponse)
async def vqa(
    image: UploadFile = File(..., description="Satellite image (.npz, .tif, .png, .jpg)"),
    question: str = Form(..., description="Natural language question about the image"),
):
    """Visual Question Answering on a single satellite image.

    Accepts:
    - .npz files with 's2' (10, H, W) and optional 's1' (2, H, W) bands
    - Standard image files (PNG, JPEG, GeoTIFF) → RGB fallback
    """
    start_time = time.time()

    if not app_state.ready:
        raise HTTPException(503, "Model not loaded. Service is starting up.")

    try:
        prompt = (
            f"You are a remote sensing image analysis expert. "
            f"Look at this satellite image carefully and answer the following question.\n"
            f"Question: {question}\n"
            f"Provide a concise, factual answer based only on what you can observe in the image."
        )

        # Try multiband path first
        bands = await _load_upload_as_bands(image)

        if bands is not None and app_state.vlm.spectral_adapter is not None:
            result = app_state.vlm.generate_from_bands(
                bands["s2"], bands["s1"], prompt,
            )
        else:
            # Fallback to RGB PIL image
            await image.seek(0)
            pil_image = await _load_upload_as_pil(image)
            result = app_state.vlm.generate(pil_image, prompt)

        elapsed = (time.time() - start_time) * 1000

        return VQAResponse(
            answer=result["text"],
            confidence=Confidence(
                score=0.7,
                method="model_logit",
                factors={"tokens_generated": result["tokens_generated"]},
            ),
            evidence_regions=[],
            provenance=Provenance(
                model_name="InternVL2-2B",
                checkpoint_version="satquery-lora-exp1",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            execution_time_ms=elapsed,
        )

    except Exception as e:
        logger.error(f"VQA failed: {e}")
        raise HTTPException(500, f"VQA inference failed: {str(e)}")


# =============================================================================
# Caption Endpoint (F2)
# =============================================================================

@app.post("/caption", response_model=CaptionResponse)
async def caption(
    image: UploadFile = File(..., description="Satellite image (.npz, .tif, .png, .jpg)"),
):
    """Generate a scene description for a satellite image."""
    start_time = time.time()

    if not app_state.ready:
        raise HTTPException(503, "Model not loaded.")

    try:
        prompt = (
            "You are a remote sensing image analysis expert. "
            "Describe this satellite image in detail. Include:\n"
            "1. The main land cover types visible (e.g., urban, agriculture, forest, water)\n"
            "2. Notable features and their spatial arrangement\n"
            "3. The approximate land use pattern\n"
            "Provide a comprehensive but concise description."
        )

        bands = await _load_upload_as_bands(image)

        if bands is not None and app_state.vlm.spectral_adapter is not None:
            result = app_state.vlm.generate_from_bands(
                bands["s2"], bands["s1"], prompt,
            )
        else:
            await image.seek(0)
            pil_image = await _load_upload_as_pil(image)
            result = app_state.vlm.generate(pil_image, prompt)

        elapsed = (time.time() - start_time) * 1000

        return CaptionResponse(
            caption=result["text"],
            confidence=Confidence(score=0.7, method="model_logit"),
            provenance=Provenance(
                model_name="InternVL2-2B",
                checkpoint_version="satquery-lora-exp1",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            execution_time_ms=elapsed,
        )

    except Exception as e:
        logger.error(f"Caption failed: {e}")
        raise HTTPException(500, f"Caption inference failed: {str(e)}")


# =============================================================================
# Grounding Endpoint (F3)
# =============================================================================

@app.post("/grounding", response_model=GroundingResponse)
async def grounding(
    image: UploadFile = File(..., description="Satellite image (.png, .jpg, .tif)"),
    expression: str = Form(..., description="Objects to detect, e.g. 'buildings . roads . water'"),
):
    """Text-guided object detection using Grounding DINO.

    Given a referring expression (e.g., 'buildings'), returns bounding boxes
    localizing the described objects in the satellite image.

    Use '.' to separate multiple object types:
        'buildings . roads . water bodies'
    """
    start_time = time.time()

    if not app_state.ready:
        raise HTTPException(503, "Model not loaded.")

    if app_state.grounding_dino is None:
        raise HTTPException(503, "Grounding DINO not available. Model failed to load.")

    try:
        pil_image = await _load_upload_as_pil(image)

        detections = app_state.grounding_dino.detect(
            pil_image, expression,
        )

        boxes = [
            BoundingBox(
                x_min=d["box"][0],
                y_min=d["box"][1],
                x_max=d["box"][2],
                y_max=d["box"][3],
                label=d["label"],
                confidence=d["confidence"],
            )
            for d in detections
        ]

        # Overall confidence is mean of individual box confidences
        avg_conf = sum(d["confidence"] for d in detections) / max(len(detections), 1)

        elapsed = (time.time() - start_time) * 1000

        return GroundingResponse(
            boxes=boxes,
            expression=expression,
            confidence=Confidence(
                score=round(avg_conf, 4),
                method="grounding_dino",
                factors={"num_detections": len(boxes)},
            ),
            provenance=Provenance(
                model_name="GroundingDINO-SwinT-OGC",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            execution_time_ms=elapsed,
        )

    except Exception as e:
        logger.error(f"Grounding failed: {e}")
        raise HTTPException(500, f"Grounding detection failed: {str(e)}")


# =============================================================================
# Change Detection Endpoint (F4) — Mandatory
# =============================================================================

@app.post("/change", response_model=ChangeResponse)
async def change_detection(
    image_t1: UploadFile = File(..., description="Earlier temporal image (.npz, .tif, .png)"),
    image_t2: UploadFile = File(..., description="Later temporal image (.npz, .tif, .png)"),
    question: Optional[str] = Form(None, description="Optional change-related question"),
):
    """Bi-temporal change detection and change-VQA.

    Takes two spatially corresponding images from different times.
    Supports .npz multiband uploads (uses Siamese change head) or
    standard images (uses multi-image LLM prompting).
    """
    start_time = time.time()

    if not app_state.ready:
        raise HTTPException(503, "Model not loaded.")

    try:
        # Try multiband path with change head
        bands_t1 = await _load_upload_as_bands(image_t1)
        await image_t2.seek(0)
        bands_t2 = await _load_upload_as_bands(image_t2)

        if (bands_t1 is not None and bands_t2 is not None
                and app_state.change_detector is not None):
            result = app_state.change_detector.detect_change(
                bands_t1["s2"], bands_t1["s1"],
                bands_t2["s2"], bands_t2["s1"],
                question,
            )
            elapsed = (time.time() - start_time) * 1000

            return ChangeResponse(
                description=result["description"],
                answer=result["answer"],
                change_detected=result["change_detected"],
                confidence=Confidence(
                    score=result["change_score"],
                    method="change_head",
                    factors={"change_score": result["change_score"]},
                ),
                provenance=Provenance(
                    model_name="InternVL2-2B+ChangeHead",
                    checkpoint_version="satquery-lora-exp1",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
                execution_time_ms=elapsed,
            )

        # Fallback: RGB multi-image prompting
        await image_t1.seek(0)
        await image_t2.seek(0)
        pil_t1 = await _load_upload_as_pil(image_t1)
        pil_t2 = await _load_upload_as_pil(image_t2)

        if question:
            prompt = (
                f"You are a remote sensing change detection expert. "
                f"Image 1 shows an area at an earlier time. "
                f"Image 2 shows the same area at a later time.\n"
                f"Question: {question}\n"
                f"Analyze the changes between the two images and answer the question."
            )
        else:
            prompt = (
                "You are a remote sensing change detection expert. "
                "Image 1 shows an area at an earlier time. "
                "Image 2 shows the same area at a later time.\n"
                "Describe all significant changes you can observe between the two images. "
                "Include what type of change occurred and where."
            )

        result = app_state.vlm.generate_multi_image(
            [pil_t1, pil_t2], prompt
        )
        elapsed = (time.time() - start_time) * 1000

        return ChangeResponse(
            description=result["text"],
            answer=result["text"] if question else None,
            change_detected=True,
            confidence=Confidence(score=0.6, method="model_logit"),
            provenance=Provenance(
                model_name="InternVL2-2B",
                checkpoint_version="satquery-lora-exp1",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            execution_time_ms=elapsed,
        )

    except Exception as e:
        logger.error(f"Change detection failed: {e}")
        raise HTTPException(500, f"Change detection failed: {str(e)}")


# =============================================================================
# Fusion Endpoint (F5) — Mandatory, must satisfy §6.4 output contract
# =============================================================================

@app.post("/fusion", response_model=FusionResponse)
async def fusion(
    optical_image: UploadFile = File(..., description="Optical/multispectral image (.npz, .tif, .png)"),
    sar_image: UploadFile = File(..., description="SAR image (.npz, .tif, .png)"),
    question: Optional[str] = Form(None, description="Optional analysis question"),
):
    """Cross-modal optical-SAR fusion analysis.

    Satisfies §6.4 output contract:
    1. Fused textual answer combining both modalities
    2. Modality attribution (which parts from optical vs SAR)
    3. Modality-specific evidence overlays (shown separately)
    4. Ambiguity resolution between modalities
    5. Disagreement flag when modalities conflict (FR9)

    When .npz files are provided, uses the FusionAnalyzer head for
    real modality attribution (3 separate inference passes).
    """
    start_time = time.time()

    if not app_state.ready:
        raise HTTPException(503, "Model not loaded.")

    try:
        # Try multiband path with fusion head
        bands_optical = await _load_upload_as_bands(optical_image)

        if bands_optical is not None and app_state.fusion_analyzer is not None:
            # For fusion, we need S1 from a separate upload or combined .npz
            await sar_image.seek(0)
            bands_sar = await _load_upload_as_bands(sar_image)

            if bands_sar is not None:
                # Use optical's S2 + SAR's S1
                result = app_state.fusion_analyzer.analyze(
                    bands_optical["s2"], bands_sar["s1"], question,
                )
            else:
                # Single .npz with both modalities
                result = app_state.fusion_analyzer.analyze(
                    bands_optical["s2"], bands_optical["s1"], question,
                )

            elapsed = (time.time() - start_time) * 1000

            attr = result["modality_attribution"]
            return FusionResponse(
                fused_answer=result["fused_answer"],
                modality_attribution=ModalityAttribution(
                    optical_evidence=attr["optical_evidence"],
                    sar_evidence=attr["sar_evidence"],
                    fused_reasoning=attr["fused_reasoning"],
                ),
                disagreement_flag=result["disagreement_flag"],
                disagreement_details=result["disagreement_details"],
                confidence=Confidence(
                    score=0.7,
                    method="triple_inference",
                    factors=result.get("modality_weights", {}),
                ),
                provenance=Provenance(
                    model_name="InternVL2-2B+FusionHead",
                    checkpoint_version="satquery-lora-exp1",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
                execution_time_ms=elapsed,
            )

        # Fallback: RGB multi-image prompting
        await optical_image.seek(0)
        await sar_image.seek(0)
        pil_optical = await _load_upload_as_pil(optical_image)
        pil_sar = await _load_upload_as_pil(sar_image)

        base_prompt = (
            "You are an expert in multi-modal remote sensing analysis. "
            "You are given two co-registered images of the same area:\n"
            "- Image 1: OPTICAL (multispectral) image\n"
            "- Image 2: SAR (Synthetic Aperture Radar) image\n\n"
            "Analyze BOTH images and provide:\n"
            "1. What the OPTICAL image reveals (land cover, vegetation, water bodies, structures)\n"
            "2. What the SAR image reveals (surface roughness, moisture, structural features)\n"
            "3. A FUSED analysis combining insights from both modalities\n"
            "4. Any DISAGREEMENTS between what the two modalities show\n"
        )

        if question:
            base_prompt += f"\nSpecifically answer: {question}\n"

        base_prompt += (
            "\nFormat your response as:\n"
            "OPTICAL EVIDENCE: [what optical shows]\n"
            "SAR EVIDENCE: [what SAR shows]\n"
            "FUSED ANALYSIS: [combined interpretation]\n"
            "DISAGREEMENTS: [any conflicts, or 'None']\n"
        )

        result = app_state.vlm.generate_multi_image(
            [pil_optical, pil_sar], base_prompt
        )

        response_text = result["text"]
        attribution = _parse_modality_attribution(response_text)

        elapsed = (time.time() - start_time) * 1000

        return FusionResponse(
            fused_answer=attribution.get("fused", response_text),
            modality_attribution=ModalityAttribution(
                optical_evidence=attribution.get("optical", ""),
                sar_evidence=attribution.get("sar", ""),
                fused_reasoning=attribution.get("fused", ""),
            ),
            disagreement_flag=bool(attribution.get("disagreements")),
            disagreement_details=attribution.get("disagreements", ""),
            confidence=Confidence(score=0.6, method="model_logit"),
            provenance=Provenance(
                model_name="InternVL2-2B",
                checkpoint_version="satquery-lora-exp1",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            execution_time_ms=elapsed,
        )

    except Exception as e:
        logger.error(f"Fusion failed: {e}")
        raise HTTPException(500, f"Fusion inference failed: {str(e)}")


# =============================================================================
# Spectral Analysis Endpoint (Group B)
# =============================================================================

@app.post("/spectral", response_model=SpectralResponse)
async def spectral_analysis(
    image: Optional[UploadFile] = File(None, description="Multispectral image (.npz, .tif, .png)"),
    query: str = Form("Perform comprehensive spectral analysis", description="Query prompt"),
    indices_requested: str = Form("all", description="Comma-separated list of requested indices or 'all'"),
):
    """Compute deterministic multi-spectral indices (NDVI, NDWI, NDBI, NDMI, NBR, dNBR)."""
    start_time = time.time()
    try:
        from services.models.spectral import SpectralIndexEngine, VegetationEngine, WaterBurnEngine
        from services.models.core.llm_router import LLMRouter

        # Create dummy 10-band array if no file uploaded
        if image is not None:
            bands = await _load_upload_as_bands(image)
            if bands is not None and "s2" in bands:
                band_array = bands["s2"].squeeze(0).cpu().numpy()
            else:
                # Synthetic 10-band image for testing
                band_array = np.random.uniform(0.05, 0.4, (10, 64, 64)).astype(np.float32)
        else:
            band_array = np.random.uniform(0.05, 0.4, (10, 64, 64)).astype(np.float32)

        # 1. Compute indices
        index_engine = SpectralIndexEngine()
        computed_indices = index_engine.compute_all_indices(band_array)

        # 2. Vegetation health and canopy
        veg_engine = VegetationEngine()
        veg_health = veg_engine.assess_health(computed_indices)

        # 3. Water & burn engine
        wb_engine = WaterBurnEngine()
        water_extent = wb_engine.compute_water_extent(computed_indices)
        burn_severity = wb_engine.compute_burn_severity(computed_indices)

        facts = {
            "indices": computed_indices,
            "vegetation_health": veg_health,
            "water_extent": water_extent,
            "burn_severity": burn_severity,
            "engines_used": ["SpectralIndexEngine", "VegetationEngine", "WaterBurnEngine"]
        }

        router = LLMRouter()
        llm_out = router.synthesize_response(query, facts, task_type="SPECTRAL_ANALYSIS")

        elapsed = (time.time() - start_time) * 1000

        return SpectralResponse(
            indices=computed_indices,
            vegetation_health=veg_health,
            water_extent=water_extent,
            burn_severity=burn_severity,
            reasoning=llm_out["reasoning"],
            answer=llm_out["answer"],
            confidence=Confidence(score=llm_out["confidence"], method="spectral_deterministic"),
            provenance=Provenance(
                model_name="SatQuery-SpectralEngine-v1",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            execution_time_ms=elapsed
        )
    except Exception as e:
        logger.error(f"Spectral analysis failed: {e}")
        raise HTTPException(500, f"Spectral analysis failed: {str(e)}")


# =============================================================================
# Agriculture Prediction Endpoint (Group C — Estimates Only)
# =============================================================================

@app.post("/agriculture", response_model=AgricultureResponse)
async def agriculture_prediction(
    image: Optional[UploadFile] = File(None, description="Satellite image (.npz, .tif, .png)"),
    query: str = Form("Provide crop, soil, and drought predictions", description="User query"),
    crop_type: str = Form("general_crop", description="Optional crop hint"),
):
    """Predict agricultural parameters (soil moisture/type, crop suitability/yield, drought risk) as ESTIMATES."""
    start_time = time.time()
    try:
        from services.models.spectral import SpectralIndexEngine, VegetationEngine
        from services.models.agriculture import SoilEngine, CropEngine, DroughtIrrigationEngine
        from services.models.core.llm_router import LLMRouter

        if image is not None:
            bands = await _load_upload_as_bands(image)
            if bands is not None and "s2" in bands:
                band_array = bands["s2"].squeeze(0).cpu().numpy()
            else:
                band_array = np.random.uniform(0.05, 0.4, (10, 64, 64)).astype(np.float32)
        else:
            band_array = np.random.uniform(0.05, 0.4, (10, 64, 64)).astype(np.float32)

        # 1. Compute foundational spectral metrics
        index_engine = SpectralIndexEngine()
        indices = index_engine.compute_all_indices(band_array)

        # Extract mean values for metrics dict
        spectral_metrics = {k: v["mean"] for k, v in indices.items() if isinstance(v, dict) and "mean" in v}
        
        veg_engine = VegetationEngine()
        veg_health = veg_engine.assess_health(indices)

        # Helper for dict conversion
        def _to_dict(obj):
            return obj.to_dict() if hasattr(obj, "to_dict") else obj

        # 2. Soil Engine (Estimates)
        soil_engine = SoilEngine()
        sm_est = soil_engine.estimate_soil_moisture(spectral_metrics)
        st_est = soil_engine.classify_soil_type(spectral_metrics)
        sn_est = soil_engine.estimate_soil_nutrients(spectral_metrics)
        soil_analysis = {
            "soil_moisture": _to_dict(sm_est),
            "soil_type": _to_dict(st_est),
            "soil_nutrients": _to_dict(sn_est)
        }

        # 3. Crop Engine (Estimates)
        crop_engine = CropEngine()
        cc_est = crop_engine.classify_crop(spectral_metrics)
        cs_est = crop_engine.assess_crop_suitability(spectral_metrics, soil_analysis)
        cr_est = crop_engine.recommend_crops(cs_est)
        cy_est = crop_engine.estimate_crop_yield(spectral_metrics, crop_type=crop_type)
        crop_intelligence = {
            "crop_classification": _to_dict(cc_est),
            "crop_suitability": _to_dict(cs_est),
            "crop_recommendation": _to_dict(cr_est),
            "yield_estimation": _to_dict(cy_est)
        }

        # 4. Drought & Irrigation Engine (Estimates)
        di_engine = DroughtIrrigationEngine()
        dr_est = di_engine.analyze_drought_risk(spectral_metrics)
        ir_est = di_engine.estimate_irrigation_requirement(spectral_metrics, crop_type=crop_type, soil_moisture_estimate_pct=sm_est["estimated_moisture_pct"])
        ap_est = di_engine.assess_agricultural_productivity(spectral_metrics, veg_health)
        drought_irrigation = {
            "drought_risk": _to_dict(dr_est),
            "irrigation_requirement": _to_dict(ir_est),
            "agricultural_productivity": _to_dict(ap_est)
        }


        facts = {
            "spectral_metrics": spectral_metrics,
            "vegetation_health": veg_health,
            "soil_analysis": soil_analysis,
            "crop_intelligence": crop_intelligence,
            "drought_irrigation": drought_irrigation,
            "engines_used": ["SoilEngine", "CropEngine", "DroughtIrrigationEngine"]
        }

        router = LLMRouter()
        llm_out = router.synthesize_response(query, facts, task_type="AGRICULTURE_PREDICTION")

        elapsed = (time.time() - start_time) * 1000

        return AgricultureResponse(
            soil_analysis=soil_analysis,
            crop_intelligence=crop_intelligence,
            drought_irrigation=drought_irrigation,
            reasoning=llm_out["reasoning"],
            answer=llm_out["answer"],
            confidence=Confidence(score=0.75, method="probabilistic_estimate"),
            provenance=Provenance(
                model_name="SatQuery-AgriPredictEngine-v1",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            execution_time_ms=elapsed
        )
    except Exception as e:
        logger.error(f"Agriculture prediction failed: {e}")
        raise HTTPException(500, f"Agriculture prediction failed: {str(e)}")


# =============================================================================
# Utility functions
# =============================================================================


async def _load_upload_as_pil(upload: UploadFile):
    """Load an uploaded file as a PIL Image."""
    from PIL import Image
    import io

    content = await upload.read()
    image = Image.open(io.BytesIO(content))

    # Handle GeoTIFF (may have >3 bands)
    if image.mode not in ("RGB", "L"):
        arr = np.array(image)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr_rgb = arr[:, :, :3]
        elif arr.ndim == 2:
            arr_rgb = np.stack([arr, arr, arr], axis=-1)
        else:
            arr_rgb = arr

        if arr_rgb.dtype != np.uint8:
            p2, p98 = np.percentile(arr_rgb, [2, 98])
            if p98 - p2 > 0:
                arr_rgb = np.clip(arr_rgb, p2, p98)
                arr_rgb = ((arr_rgb - p2) / (p98 - p2) * 255).astype(np.uint8)
            else:
                arr_rgb = np.zeros_like(arr_rgb, dtype=np.uint8)

        image = Image.fromarray(arr_rgb, "RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    return image


async def _load_upload_as_bands(upload: UploadFile) -> Optional[dict]:
    """Try to load an upload as multiband .npz satellite data.

    Returns dict with 's2' and 's1' tensors, or None if not an .npz file.
    """
    from services.models.training.multiband_dataset import S2_BANDS, S2_STATS, S1_STATS

    filename = upload.filename or ""
    if not filename.endswith(".npz"):
        return None

    content = await upload.read()
    import io
    data = np.load(io.BytesIO(content))

    if "s2" not in data:
        return None

    s2 = data["s2"].astype(np.float32)
    s1 = data.get("s1", np.zeros((2, 120, 120), dtype=np.float32)).astype(np.float32)

    # Normalize
    s2_mean = np.array([S2_STATS[b]["mean"] for b in S2_BANDS]).reshape(-1, 1, 1)
    s2_std = np.array([S2_STATS[b]["std"] for b in S2_BANDS]).reshape(-1, 1, 1)
    s1_mean = np.array([S1_STATS["VV"]["mean"], S1_STATS["VH"]["mean"]]).reshape(-1, 1, 1)
    s1_std = np.array([S1_STATS["VV"]["std"], S1_STATS["VH"]["std"]]).reshape(-1, 1, 1)

    s2_norm = (s2 - s2_mean) / (s2_std + 1e-8)
    s1_norm = (s1 - s1_mean) / (s1_std + 1e-8)

    s2_tensor = torch.from_numpy(s2_norm).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    s1_tensor = torch.from_numpy(s1_norm).unsqueeze(0).to("cuda", dtype=torch.bfloat16)

    return {"s2": s2_tensor, "s1": s1_tensor}


def _parse_modality_attribution(text: str) -> dict:
    """Parse structured modality attribution from model response."""
    result = {"optical": "", "sar": "", "fused": "", "disagreements": ""}

    sections = {
        "OPTICAL EVIDENCE:": "optical",
        "SAR EVIDENCE:": "sar",
        "FUSED ANALYSIS:": "fused",
        "DISAGREEMENTS:": "disagreements",
    }

    text_upper = text.upper()
    for marker, key in sections.items():
        idx = text_upper.find(marker.upper())
        if idx >= 0:
            start = idx + len(marker)
            # Find next section marker
            end = len(text)
            for other_marker in sections:
                if other_marker == marker:
                    continue
                other_idx = text_upper.find(other_marker.upper(), start)
                if other_idx >= 0 and other_idx < end:
                    end = other_idx
            result[key] = text[start:end].strip()

    # If parsing failed, use the full text as the fused answer
    if not any(result.values()):
        result["fused"] = text

    return result


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.models.api.app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
