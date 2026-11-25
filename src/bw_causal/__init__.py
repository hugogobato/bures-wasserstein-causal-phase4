"""Prototype estimators for intrinsic causal effects on covariance outcomes."""

from .baselines import (
    BWFrechetGlobalTLearner, BWFrechetTLearner, BWRFWLFRTLearner, ElementwiseDRForest,
    LogCholeskyDRForest, LogEuclideanDRForest, SqrtEuclideanDRForest,
)
from .forest import BWCausalForest, ForestConfig, HonestSplit, MatrixPrediction, prepare_cohort
from .inference import simultaneous_regions, spectral_targets

__all__ = [
    "BWCausalForest", "BWFrechetGlobalTLearner", "BWFrechetTLearner", "BWRFWLFRTLearner",
    "ElementwiseDRForest", "ForestConfig", "HonestSplit", "LogCholeskyDRForest",
    "LogEuclideanDRForest", "MatrixPrediction", "SqrtEuclideanDRForest", "prepare_cohort",
    "simultaneous_regions", "spectral_targets",
]
