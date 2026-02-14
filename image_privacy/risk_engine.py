class RiskEngine:
    """
    Computes privacy risk score based on detection results.
    Logic: (face_count * 0.3) + (object_count * 0.4) + (sensitive_text_count * 0.3)
    """
    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.face_weight = 0.3
        self.object_weight = 0.4
        self.text_weight = 0.3

    def compute_risk(self, face_count, object_count, text_count):
        """
        Calculates risk score and determines safety.
        """
        risk_score = (face_count * self.face_weight) + \
                     (object_count * self.object_weight) + \
                     (text_count * self.text_weight)
        
        # Round for clean output
        risk_score = round(risk_score, 2)
        
        safe_to_forward = risk_score <= self.threshold
        
        return risk_score, safe_to_forward
