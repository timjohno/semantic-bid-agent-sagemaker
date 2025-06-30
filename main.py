import asyncio
import os
import json
import requests
import faiss
import numpy as np
import sqlite3
import streamlit as st
import random
import boto3
from boto3.dynamodb.conditions import Key, Attr


from dataclasses import dataclass, field, asdict
from typing import Annotated, Optional, TypedDict, List, Any, Dict

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.functions import KernelArguments, kernel_function
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent, FunctionResultContent, TextContent
from sentence_transformers import SentenceTransformer





import pandas as pd
#boto3.setup_default_session(profile_name='AWSAdministrator+Bedrock-613820096587')


from semantic_kernel.connectors.ai.bedrock.bedrock_prompt_execution_settings import BedrockChatPromptExecutionSettings
from semantic_kernel.connectors.ai.bedrock.services.bedrock_chat_completion import BedrockChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
#from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings


from semantic_kernel.agents import BedrockAgent, BedrockAgentThread


#api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]

# --- Instructions
AGENT_INSTRUCTIONS = """You are an expert insurance underwriting consultant. Your name, if asked, is 'IUA'.
 
Wait for specific instructions from the user before taking any action. Do not perform tasks unless they are explicitly requested.
 
You may be asked to:
- Assess the risk profile of an organisation based on model outputs. Please check the database first then run this
- Estimate the likely insurance premium using our model. Please check the database first then run this
- Reference insights from a database to assist underwriting decisions
 
If a large document has been pasted into the chat, use StructureClaimData to structure its contents and use the output for any function that takes a `claim_data` parameter.
 
Keep responses brief—no more than a few paragraphs—and always respond only to what the user has asked, when they ask it. 
For example 
- If the user only asks for risk rating only give the risk rating 
- If they only ask for insurance premium only give the insurance premium, do not run both models unless you are asked to in the prompt
- If they only ask for insights from the database do not give risk or insurance premium scores.
"""


@dataclass
class AgentMessage:
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    function_response: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResponse:
    messages: List[AgentMessage]
    thread: ChatHistoryAgentThread
    metrics: Dict[str, Any] = field(default_factory=dict)

#################################################################
#TODO: Adapt to use db after chat with Tania
class FailureScoreChecker:
    @kernel_function(description="Retrieve the failure score and failure score commentary for an organisation from the Dun & Bradstreed Database.")
    async def retrieve_failure_rating(
        self,
        claim_data: Annotated[dict, "Structured claim object containing organisation_name."]
    ) -> dict:
        dynamodb = boto3.resource('dynamodb', region_name="eu-north-1")
        dnb_table = dynamodb.Table("dnb_data")
        organisation_name = claim_data.get("organisation_name", "N/A")

        '''
        name = organisation_name
        df = pd.read_csv("data\\dnb.csv")
        risk_score = df.loc[df.organization_name==name, 'climate_risk_score'].values[0]
        return {
            "organisation_name": name,
            "climate_risk_score": risk_score
        }
        '''

        response = dnb_table.scan()
        items = response.get('Items', [])

        return items

        '''
        response = dnb_table.scan(
            FilterExpression='organization_name = :org',
            ExpressionAttributeValues={':org': {'S': organisation_name}},
            ProjectionExpression='organization_name, climate_risk_score'
        )

        item = response['Items'][0] if response['Items'] else None

        return {
            "organisation_name": organisation_name,
            "climate_risk_score": item['climate_risk_score']['S'] if item else None
        }'''

##################################################################

##################################################################
#TODO: Adapt to db data
class RiskEvaluator:
    def __init__(self):
        self.runtime = boto3.client("sagemaker-runtime")
        self.endpoint_name = "fraud-detection-xgb-v1-endpoint"

    @kernel_function(description="Determine the overall risk exposure rating of an organization based on our model to help support underwriters")
    async def assess_risk(
        self,
        claim_data: Annotated[dict, "Structured claim data with fields like coverage_amount and region_of_operation."]
    ) -> dict:
        
        return {
            "risk_score": 0.48,
            "model_used": "fraud-detection-xgb-v1-endpoint"
        }

        '''return {
            "risk_score": 0.48,
            "service_used": self.runtime,
            "model_used": self.endpoint_name 
        }'''
        
        '''coverage_amount = claim_data.get("coverage_amount", "")
        
        region_of_operation = claim_data.get("region_of_operation", "").lower()
        organisation_name =  claim_data.get("organisation_name", "").lower()
        
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
            region_value = 5'''
        '''
        #payload = f"{coverage_amount},{region_value}"
        payload = f"1,150000000"


        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="text/csv",
            Body=payload
        )
        prediction = json.loads(response["Body"].read().decode())

        return {
            "risk_score": prediction
        }

        rationale = f"Risk for organisation: {organisation_name} assessed as {prediction} based on our model."

        return {
            "risk_score": prediction,
            "explanation": rationale
        }
        '''



######################################################

import boto3
import json
import numpy as np
from typing import Annotated

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
        '''coverage_amount_str = claim_data.get("coverage_amount", "").lower()
        region_of_operation = claim_data.get("region_of_operation", "").lower()
        cleaned = coverage_amount_str.lower().replace(",", "")
        # Extract digits only
        digits = ''.join(filter(str.isdigit, cleaned))'''
        # Convert to integer and scale down
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

        #try:
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



######################################################

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
            print(f"Looking up policy by policy_number: {policy_number}")
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
            print(f"Looking up customer_id for: {claimant_name}")
            customer_response = self.customers_table.scan(
                FilterExpression=Attr('name').eq(claimant_name)
            )
            customer_items = customer_response.get('Items', [])
            if not customer_items:
                print("Customer not found.")
                return []
            customer_item = customer_items[0]
            customer_id = customer_item['customer_id']

            policy_response = self.policies_table.scan(
                FilterExpression=Attr('customer_id').eq(customer_id)
            )
            policy_items = policy_response.get('Items', [])

        else:
            print("No policy_number or claimant_name provided.")
            return []

        # Format the output
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
            # Get policy to find customer_id
            policy_resp = self.policies_table.scan(FilterExpression=Attr('policy_number').eq(policy_number))
            if not policy_resp['Items']:
                return {"error": "Policy not found"}
            policy = policy_resp['Items'][0]
            customer_id = policy['customer_id']

            # Get customer info
            customer_resp = self.customers_table.scan(FilterExpression=Attr('customer_id').eq(customer_id))
            customer_name = customer_resp['Items'][0]['name'] if customer_resp['Items'] else "Unknown"

            # Get claims
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
            # Get customer_id
            customer_resp = self.customers_table.scan(FilterExpression=Attr('name').eq(claimant_name))
            if not customer_resp['Items']:
                return {"error": "Customer not found"}
            customer = customer_resp['Items'][0]
            customer_id = customer['customer_id']
            customer_name = customer['name']

            # Get all policies for customer
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



#############################################################################
class VectorMemoryRAGPlugin:
    def __init__(self):
        self.text_chunks = []
        self.index = None
        self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")

    def add_document(self, doc_text: str, chunk_size: int = 500):
        self.text_chunks = [
            doc_text[i:i + chunk_size]
            for i in range(0, len(doc_text), chunk_size)
        ]
        vectors = self.embeddings.encode(self.text_chunks, convert_to_numpy=True)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    @kernel_function(description="retrieve relevant chunks from uploaded claim documents.")
    async def retrieve_chunks(self, query: Annotated[str, "Query to summmarise / retrieve relevant claim information"]) -> str:
        if not self.index:
            return "No documents indexed yet."
        query_vec = self.embeddings.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, k=3)
        relevant_chunks = [self.text_chunks[i] for i in I[0] if i < len(self.text_chunks)]
        return "\n---\n".join(relevant_chunks)

class ConsumerDutyChecker:
    def __init__(self, kernel: Optional[Kernel] = None):
        self.kernel = kernel
    
    @kernel_function(
        description="Evaluate if an insurance claim decision meets UK consumer duty requirements",
        name="check_consumer_duty"
    )
    async def check_consumer_duty(
        self,
        decision_text: Annotated[str, "The claim decision and explanation"],
    ) -> dict:
        prompt = f"""
System: You are a UK Consumer Duty compliance expert specializing in insurance claims evaluation.
Your task is to analyze the following insurance claim decision and provide a DETAILED evaluation 
of FCA Consumer Duty compliance requirements.

IMPORTANT: You MUST provide specific, detailed notes for each requirement explaining exactly why 
it passed or failed. Do not leave any fields empty.

For the decision text below:
\"\"\"{decision_text}\"\"\"

Evaluate each requirement and explain your reasoning:

1. Clear Communication
- Check for clear, simple language
- Identify any technical jargon
- Assess readability level

2. Fair Treatment
- Evaluate if the decision process was fair
- Check if all relevant factors were considered
- Assess if the outcome is justified

3. Transparent Reasoning
- Verify if decision rationale is clearly explained
- Check if policy terms are referenced correctly
- Ensure all key points are addressed

4. Consumer Understanding
- Assess if an average customer would understand
- Check if next steps are clearly explained
- Verify if important points are emphasized

5. Vulnerable Customer Considerations
- Check if potential vulnerabilities are considered
- Assess if additional support is offered
- Evaluate accessibility of communication

You MUST return a fully populated JSON response with detailed notes for each section:

{{
    "meets_requirements": true/false,
    "checklist": {{
        "clear_communication": {{
            "passed": true/false,
            "notes": "DETAILED explanation of why it passed or failed"
        }},
        "fair_treatment": {{
            "passed": true/false,
            "notes": "DETAILED explanation of why it passed or failed"
        }},
        "transparent_reasoning": {{
            "passed": true/false,
            "notes": "DETAILED explanation of why it passed or failed"
        }},
        "consumer_understanding": {{
            "passed": true/false,
            "notes": "DETAILED explanation of why it passed or failed"
        }},
        "vulnerable_customers": {{
            "passed": true/false,
            "notes": "DETAILED explanation of why it passed or failed"
        }}
    }},
    "improvement_suggestions": [
        "Specific actionable improvements needed"
    ],
    "risk_flags": [
        "Specific compliance risks that need addressing"
    ]
}}

Respond ONLY with a valid JSON object. Do not include any text before or after the JSON.
"""
        completion = await self.kernel.invoke_prompt(prompt)
        if hasattr(completion, "result"):
            return str(completion.result).strip()
        return str(completion).strip()

class StructureClaimData:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    @kernel_function(description="Return a JSON containing structured claim_data, use before calling other plugins")
    async def StructureClaimData(self, claim_text: Annotated[str, "The unstructured claim_text string input"]) -> str:
        prompt = f"""
Extract the following fields from the text below. If a field is not present, leave it blank. for coverage_amount, extract the number. For example USD 150,000,000 should be coverage_amount: 150000000

Required fields (as JSON):
{{
    "organisation_name": "",
    "region_of_operation": "",
    "coverage_amount": "",
    "premium": "",
    "export_destination": "",
    "client_priorities": ""
}}

Text:
\"\"\"{claim_text}\"\"\"

Respond ONLY with a valid JSON object. Do not include any text before or after the JSON.
"""
        completion = await self.kernel.invoke_prompt(prompt)
        if hasattr(completion, "result"):
            return str(completion.result).strip()
        return str(completion).strip()

# --- Main async entrypoint
async def main(
    user_input: str, 
    thread: Optional[ChatHistoryAgentThread] = None, 
    claim_text: Optional[str] = None
) -> AgentResponse:
    kernel = Kernel()
    kernel.add_service(BedrockChatCompletion(
        model_id="eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
    ))

    messages: List[AgentMessage] = []

    # 👉 Keep RAG setup for policy lookup
    vector_memory_rag = VectorMemoryRAGPlugin()
    if claim_text:
        vector_memory_rag.add_document(claim_text)

    # --- Register plugins
    kernel.add_plugin( FailureScoreChecker(), plugin_name="FailureScoreChecker")
    #kernel.add_plugin(DataCollector(kernel), plugin_name="collector")    
    kernel.add_plugin(vector_memory_rag, plugin_name="VectorMemoryRAG")
    kernel.add_plugin(RiskEvaluator(), plugin_name="RiskModel")
    kernel.add_plugin(InsurancePremiumEstimator(), plugin_name="PremiumEstimator")
    #kernel.add_plugin(ConsumerDutyChecker(kernel), plugin_name="ConsumerDuty")
    kernel.add_plugin(StructureClaimData(kernel), plugin_name="StructureClaimData")

    

    agent = ChatCompletionAgent(
        kernel=kernel,
        name="IUA",
        instructions=AGENT_INSTRUCTIONS,
        arguments=KernelArguments(
            settings=BedrockChatPromptExecutionSettings(
                temperature=0.5,
                top_p=0.95,
                function_choice_behavior=FunctionChoiceBehavior.Auto()
            )
        )
    )

    intermediate_steps: List[AgentMessage] = []
    metrics = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "steps": 0
    }

    messages.append(AgentMessage(role="user", content=user_input))

    async def handle_intermediate(message: ChatMessageContent):
        for item in message.items:
            if isinstance(item, FunctionCallContent):
                intermediate_steps.append(AgentMessage(
                    role="function_call",
                    name=item.name,
                    function_call={"name": item.name, "arguments": item.arguments},
                ))
            elif isinstance(item, FunctionResultContent):
                intermediate_steps.append(AgentMessage(
                    role="function_response",
                    name=item.name,
                    function_response=item.result,
                ))
            else:
                intermediate_steps.append(AgentMessage(
                    role=message.role.value,
                    content=message.content
                ))

    async for response in agent.invoke(
        messages=user_input,
        thread=thread,
        on_intermediate_message=handle_intermediate
    ):
        message = AgentMessage(
            role=response.role,
            content=response.content,
            name=getattr(response, "name", None)
        )

        if hasattr(response, "function_call") and response.function_call:
            message.function_call = {
                "name": response.function_call.name,
                "arguments": response.function_call.arguments
            }

        if hasattr(response, "function_response") and response.function_response:
            message.function_response = response.function_response

        if hasattr(response, "metadata") and response.metadata:
            message.metadata = response.metadata
            if "usage" in message.metadata:
                usage = message.metadata["usage"]
                metrics["prompt_tokens"] += usage.prompt_tokens
                metrics["completion_tokens"] += usage.completion_tokens
                metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]

        messages.extend(intermediate_steps)
        metrics["steps"] += len(intermediate_steps)
        intermediate_steps.clear()
        messages.append(message)
        metrics["steps"] += 1
        thread = response.thread

    return AgentResponse(
        messages=messages,
        thread=thread,
        metrics=metrics
    )

# --- Debug runner
if __name__ == "__main__":
    async def test():
        response = await main("produce a bid to sell a new AI note taking product to Microsoft")
        for msg in response.messages:
            print(f"\n[{msg.role}]")
            if msg.content:
                print(msg.content.strip())
            if msg.function_call:
                print(f"Function: {msg.function_call['name']}")
            if msg.function_response:
                print(f"Result: {str(msg.function_response)[:100]}...")
        print(f"\nMetrics: {response.metrics}")

    asyncio.run(test())
