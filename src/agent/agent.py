

import boto3
import streamlit as st
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.bedrock.bedrock_prompt_execution_settings import BedrockChatPromptExecutionSettings
from semantic_kernel.connectors.ai.bedrock.services.bedrock_chat_completion import BedrockChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

from src.kernel_functions.failure_score_checker import FailureScoreChecker
from src.kernel_functions.risk_evaluator import RiskEvaluator
from src.kernel_functions.insurance_premium_estimator import InsurancePremiumEstimator
from src.kernel_functions.data_collector import DataCollector
from src.kernel_functions.vector_memory_rag_plugin import VectorMemoryRAGPlugin
from src.kernel_functions.consumer_duty_checker import ConsumerDutyChecker
from src.kernel_functions.structure_claim_data import StructureClaimData

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

def make_agent(claim_text):
    kernel = Kernel()
    runtime_client=boto3.client(
        "bedrock-runtime",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_REGION"]
    )
    client=boto3.client(
        "bedrock",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_REGION"]
    )
    kernel.add_service(BedrockChatCompletion(
        model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
        runtime_client=runtime_client, client=client
    ))


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

    return agent
