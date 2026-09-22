"""
Unit tests for SatQuery AI Spectral & Agriculture Intelligence Engines.
"""
import pytest
import numpy as np

from services.models.spectral.indices import SpectralIndexEngine
from services.models.spectral.vegetation_engine import VegetationEngine
from services.models.spectral.water_burn_engine import WaterBurnEngine
from services.models.agriculture.soil_engine import SoilEngine
from services.models.agriculture.crop_engine import CropEngine
from services.models.agriculture.drought_irrigation_engine import DroughtIrrigationEngine
from services.models.core.llm_router import LLMRouter
from controller.classifier import QueryClassifier, TaskType, InputScope


def test_spectral_index_engine():
    engine = SpectralIndexEngine()
    # 10 synthetic bands: (10, 32, 32)
    bands = np.random.uniform(0.05, 0.4, (10, 32, 32)).astype(np.float32)
    res = engine.compute_all_indices(bands)

    assert "ndvi" in res
    assert "ndwi" in res
    assert "ndmi" in res
    assert "ndbi" in res
    assert "nbr" in res
    assert res["ndvi"]["min"] >= -1.0
    assert res["ndvi"]["max"] <= 1.0


def test_vegetation_engine():
    indices = {
        "ndvi": {"mean": 0.65, "std": 0.05, "min": 0.1, "max": 0.8},
        "ndmi": {"mean": 0.35, "std": 0.04, "min": 0.0, "max": 0.5},
        "ndre": {"mean": 0.45, "std": 0.03, "min": 0.0, "max": 0.6}
    }
    engine = VegetationEngine()
    health = engine.assess_health(indices)

    assert health["status"] in ["EXCELLENT", "HEALTHY", "MODERATE", "STRESSED", "CRITICAL"]
    assert "composite_health_score" in health
    assert health["composite_health_score"] >= 0.0


def test_water_burn_engine():
    engine = WaterBurnEngine()
    indices = {
        "ndwi": {"mean": 0.4, "std": 0.1},
        "mndwi": {"mean": 0.5, "std": 0.1},
        "nbr": {"mean": 0.1, "std": 0.05},
        "dnbr": {"mean": 0.35, "std": 0.02}
    }
    w = engine.compute_water_extent(indices)
    assert "water_detected" in w
    assert "water_coverage_pct" in w

    b = engine.compute_burn_severity(indices)
    assert b["severity_class"] in ["UNBURNED", "LOW_SEVERITY", "MODERATE_SEVERITY", "HIGH_SEVERITY"]


def test_agriculture_soil_crop_drought_engines():
    metrics = {"ndvi_mean": 0.55, "ndmi_mean": 0.32, "ndwi_mean": -0.15, "ndbi_mean": -0.2}

    soil_eng = SoilEngine()
    sm = soil_eng.estimate_soil_moisture(metrics)
    assert sm["is_estimate"] is True
    assert sm["status"] == "ESTIMATE"

    st = soil_eng.classify_soil_type(metrics)
    assert st["is_estimate"] is True

    crop_eng = CropEngine()
    cc = crop_eng.classify_crop(metrics)
    assert cc["is_estimate"] is True

    cs = crop_eng.assess_crop_suitability(metrics)
    assert cs["is_estimate"] is True

    cy = crop_eng.estimate_crop_yield(metrics)
    assert cy["is_estimate"] is True

    di_eng = DroughtIrrigationEngine()
    dr = di_eng.analyze_drought_risk(metrics)
    assert dr["is_estimate"] is True
    assert dr["drought_risk_level"] in ["LOW", "MODERATE", "HIGH", "SEVERE"]


def test_llm_router():
    router = LLMRouter()
    facts = {
        "indices": {"ndvi": 0.62, "ndmi": 0.38},
        "vegetation_health": {"status": "HEALTHY", "composite_health_score": 78.5},
        "soil_analysis": {"soil_moisture": {"is_estimate": True, "estimated_moisture_pct": 24.5}}
    }
    out = router.synthesize_response("What is the health and soil moisture?", facts, task_type="AGRICULTURE_PREDICTION")

    assert "answer" in out
    assert "reasoning" in out
    assert "Estimate" in out["answer"] or "estimate" in out["answer"]


def test_query_classifier_rules():
    classifier = QueryClassifier()
    task_spec = classifier.classify("Calculate NDVI and vegetation stress", InputScope.SINGLE)
    assert task_spec == TaskType.SPECTRAL_ANALYSIS

    task_agri = classifier.classify("What is the estimated soil moisture and drought risk?", InputScope.SINGLE)
    assert task_agri == TaskType.AGRICULTURE_PREDICTION
