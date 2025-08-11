import boto3
import json
from typing import Annotated
from semantic_kernel.functions import kernel_function


REGION_MODIFIERS = {
    "gb": 2.2,
    "usa": 2.1,
    "eu": 2.3,
    "asia": 3.0,
    "africa": 4.5,
}

class MockInsurancePremiumEstimator:
    def __init__(self):
        self.runtime = boto3.client("sagemaker-runtime")
        self.endpoint_name = "claim-amount-linear-v2-endpoint"

    @kernel_function(description="Estimate the likely insurance premium range using model in GBP.")
    async def estimate_size(
        self,
        claim_data: Annotated[dict, "Structured company data."]
    ) -> dict:
        coverage_amount = claim_data.get("coverage_amount", "")
        coverage_amount = int(coverage_amount) // 100 if coverage_amount else 0

        region_of_operation = claim_data.get("region_of_operation", "").lower()
        modifier = REGION_MODIFIERS.get(region_of_operation, 1.5)
        premium = coverage_amount * modifier


        return {
            "estimated_insurance_premium": round(premium, 2),
            "currency": "GBP",
            "service_used": self.runtime,
            "model_used": self.endpoint_name 
        }
