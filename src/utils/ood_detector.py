import torch
import torch.nn.functional as F
import numpy as np

class OODDetector:
    """
    Out-of-Distribution (OOD) & Non-Leaf Detection Module.
    Combines Energy-based Scores and Softmax Entropy to flag invalid, noisy,
    or non-leaf images to prevent false diagnostic predictions.
    """
    def __init__(self, energy_threshold=-8.5, entropy_threshold=2.2, temperature=1.0):
        self.energy_threshold = energy_threshold
        self.entropy_threshold = entropy_threshold
        self.temperature = temperature

    def compute_scores(self, logits):
        """
        Computes energy score and normalized entropy from model output logits.
        """
        # Energy Score: E(x) = -T * log(sum(exp(logits / T)))
        scaled_logits = logits / self.temperature
        energy = -self.temperature * torch.logsumexp(scaled_logits, dim=1).item()

        # Softmax Entropy: H(p) = -sum(p * log(p))
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1).item()

        max_prob = torch.max(probs).item()

        return {
            "energy": energy,
            "entropy": entropy,
            "max_confidence": max_prob
        }

    def is_ood(self, logits):
        """
        Determines if an input is Out-of-Distribution (Non-Leaf or Unknown Sample).
        Returns (is_ood: bool, reason: str, scores: dict).
        """
        scores = self.compute_scores(logits)

        reasons = []
        if scores["entropy"] > self.entropy_threshold:
            reasons.append("High Prediction Uncertainty (Entropy)")
        if scores["max_confidence"] < 0.40:
            reasons.append("Low Max Confidence (< 40%)")

        is_ood = len(reasons) > 0
        reason_str = " & ".join(reasons) if is_ood else "In-Distribution (Valid Sample)"

        return is_ood, reason_str, scores
