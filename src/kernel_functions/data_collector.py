import boto3
from typing import Annotated
from semantic_kernel.functions import kernel_function
from semantic_kernel import Kernel
from boto3.dynamodb.conditions import Attr

class DataCollector:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.dynamodb = boto3.resource('dynamodb', region_name="eu-north-1")
        self.customers_table = self.dynamodb.Table("customers")
        self.policies_table = self.dynamodb.Table("policies")
        self.claims_table = self.dynamodb.Table("historic_claims")

    @kernel_function(description="Validate a policy using policy number or claimant info against internal database.")
    async def get_user_policy(
        self,
        claim_data: Annotated[dict, "Structured claim object containing at least claimant_name or policy_number."]
    ) -> dict:
        policy_number = claim_data.get("policy_number", "").strip()
        claimant_name = claim_data.get("claimant_name", "").strip()
        customer_item = None
        policy_items = []
        if policy_number:
            policy_response = self.policies_table.scan(
                FilterExpression=Attr('policy_number').eq(policy_number)
            )
            policy_items = policy_response.get('Items', [])
            if policy_items:
                customer_id = policy_items[0]['customer_id']
                customer_response = self.customers_table.scan(
                    FilterExpression=Attr('customer_id').eq(customer_id)
                )
                customer_items = customer_response.get('Items', [])
                if customer_items:
                    customer_item = customer_items[0]
        elif claimant_name:
            customer_response = self.customers_table.scan(
                FilterExpression=Attr('name').eq(claimant_name)
            )
            customer_items = customer_response.get('Items', [])
            if not customer_items:
                return []
            customer_item = customer_items[0]
            customer_id = customer_item['customer_id']
            policy_response = self.policies_table.scan(
                FilterExpression=Attr('customer_id').eq(customer_id)
            )
            policy_items = policy_response.get('Items', [])
        else:
            return []
        results = []
        for policy_item in policy_items:
            if customer_item:
                result = {
                    "policy_number": policy_item["policy_number"],
                    "claimant_name": customer_item["name"],
                    "dob": customer_item.get("dob"),
                    "address": customer_item.get("address"),
                    "covered_incidents": policy_item["coverage"].split(","),
                    "coverage_limit": policy_item["coverage_limit"],
                    "deductible": policy_item["deductible"],
                    "policy_status": policy_item["status"]
                }
                results.append(result)
        return results

    @kernel_function(description="Retrieve claim history for a customer using policy number or claimant name.")
    async def get_claim_history(
        self,
        claim_data: Annotated[dict, "Structured claim object containing at least claimant_name or policy_number."]
    ) -> dict:
        policy_number = claim_data.get("policy_number", "").strip()
        claimant_name = claim_data.get("claimant_name", "").strip()
        claims = []
        customer_name = None
        if policy_number:
            policy_resp = self.policies_table.scan(FilterExpression=Attr('policy_number').eq(policy_number))
            if not policy_resp['Items']:
                return {"error": "Policy not found"}
            policy = policy_resp['Items'][0]
            customer_id = policy['customer_id']
            customer_resp = self.customers_table.scan(FilterExpression=Attr('customer_id').eq(customer_id))
            customer_name = customer_resp['Items'][0]['name'] if customer_resp['Items'] else "Unknown"
            claims_resp = self.claims_table.scan(FilterExpression=Attr('policy_number').eq(policy_number))
            for row in claims_resp['Items']:
                claims.append({
                    "claim_id": row.get("claim_id"),
                    "incident_type": row.get("incident_type"),
                    "incident_date": row.get("incident_date"),
                    "claim_amount": float(row.get("claim_amount", 0)),
                    "status": row.get("status"),
                    "decision_date": row.get("decision_date"),
                    "description": row.get("description"),
                    "claimant_name": customer_name,
                    "policy_number": policy_number
                })
        elif claimant_name:
            customer_resp = self.customers_table.scan(FilterExpression=Attr('name').eq(claimant_name))
            if not customer_resp['Items']:
                return {"error": "Customer not found"}
            customer = customer_resp['Items'][0]
            customer_id = customer['customer_id']
            customer_name = customer['name']
            policy_resp = self.policies_table.scan(FilterExpression=Attr('customer_id').eq(customer_id))
            for policy in policy_resp['Items']:
                policy_number = policy['policy_number']
                claims_resp = self.claims_table.scan(FilterExpression=Attr('policy_number').eq(policy_number))
                for row in claims_resp['Items']:
                    claims.append({
                        "claim_id": row.get("claim_id"),
                        "incident_type": row.get("incident_type"),
                        "incident_date": row.get("incident_date"),
                        "claim_amount": float(row.get("claim_amount", 0)),
                        "status": row.get("status"),
                        "decision_date": row.get("decision_date"),
                        "description": row.get("description"),
                        "claimant_name": customer_name,
                        "policy_number": policy_number
                    })
        else:
            return {"error": "No policy_number or claimant_name provided"}
        if not claims:
            return {"error": "No claim history found"}
        return {
            "claimant_name": customer_name,
            "policy_number": claims[0]["policy_number"] if claims else None,
            "total_claims": len(claims),
            "total_approved_amount": sum(c["claim_amount"] for c in claims if c["status"] == "approved"),
            "claims": claims
        }
