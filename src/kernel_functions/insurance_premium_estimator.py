import boto3
import json
from typing import Annotated
from semantic_kernel.functions import kernel_function

class InsurancePremiumEstimator:
    def __init__(self):
        self.runtime = boto3.client("sagemaker-runtime")
        self.endpoint_name = "claim-amount-linear-v2-endpoint"

    @kernel_function(description="Estimate the likely insurance premium range using model in GBP.")
    async def estimate_size(
        self,
        claim_data: Annotated[dict, "Structured company data."]
    ) -> dict:
        coverage_amount = claim_data.get("coverage_amount", "")
        region_of_operation = claim_data.get("region_of_operation", "").lower()
        coverage_amount = int(coverage_amount) // 1000 if coverage_amount else 0
        if region_of_operation == "gb":
            region_value = 0
        elif region_of_operation == "usa":
            region_value = 1
        elif region_of_operation == "eu":
            region_value = 2
        elif region_of_operation == "asia":
            region_value = 3
        elif region_of_operation == "africa":
            region_value = 4
        else:
            region_value = 5
        payload = f"{coverage_amount},{region_value}"
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="text/csv",
            Body=payload
        )
        result = json.loads(response["Body"].read().decode())
        prediction = result["predictions"][0]["score"]
        return {
            "estimated_insurance_premium": round(prediction, 2),
            "currency": "GBP",
            "service_used": self.runtime,
            "model_used": self.endpoint_name 
        }
