import boto3
from typing import Annotated
from semantic_kernel.functions import kernel_function

class FailureScoreChecker:
    @kernel_function(description="Retrieve the failure score and failure score commentary for an organisation from the Dun & Bradstreed Database.")
    async def retrieve_failure_rating(
        self,
        claim_data: Annotated[dict, "Structured claim object containing organisation_name."]
    ) -> dict:
        dynamodb = boto3.resource('dynamodb', region_name="eu-west-2")
        dnb_table = dynamodb.Table("dnb_data")
        organisation_name = claim_data.get("organisation_name", "N/A")
        response = dnb_table.scan()
        items = response.get('Items', [])
        return items
