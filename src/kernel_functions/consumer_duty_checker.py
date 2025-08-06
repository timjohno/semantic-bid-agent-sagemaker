from semantic_kernel.functions import kernel_function
from typing import Annotated, Optional
from semantic_kernel import Kernel

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
