from fastapi import FastAPI, HTTPException
import httpx
import asyncio

from schemas import UserRequest, ClassifierResponse
from defenses import apply_defense
from config import CLASSIFIER_URLS

app = FastAPI(title="LLM Security Gateway - Adaptive Defense")


async def call_classifier(client: httpx.AsyncClient, url: str, text: str) -> ClassifierResponse:
    try:
        response = await client.post(url, json={"text": text}, timeout=2.0)
        response.raise_for_status()
        return ClassifierResponse(**response.json())
    except httpx.RequestError as e:
        return ClassifierResponse(
            is_safe=False,
            reason=f"Network Error: {str(e)}",
            recommended_defense="isolation"  # A safe default if the classifier is down
        )


@app.post("/generate")
async def generate_secure_response(request: UserRequest):
    instruction_text = request.instruction
    data_text = request.data

    async with httpx.AsyncClient() as client:
        instruction_task = call_classifier(client, CLASSIFIER_URLS["instruction"], instruction_text)
        data_task = call_classifier(client, CLASSIFIER_URLS["data"], data_text)

        instruction_result, data_result = await asyncio.gather(instruction_task, data_task)

        if not instruction_result.is_safe:
            raise HTTPException(
                status_code=403,
                detail=f"Blocked instruction. Detected: {instruction_result.attack_type or 'Unknown Attack'}"
            )

        system_prompt = "Below is an instruction that describes a task, paired with an input..."
        applied_defense = "none"

        if not data_result.is_safe:
            target_defense = data_result.recommended_defense or request.fallback_defense
            applied_defense = target_defense

            print(f"Attack Detected: {data_result.attack_type}. Applying adaptive defense: '{target_defense}'")

            defense_payload = apply_defense(instruction_text, data_text, target_defense)

            instruction_text = defense_payload["instruction"]
            data_text = defense_payload["data"]

            if defense_payload["system_modifier"]:
                system_prompt += f" {defense_payload['system_modifier']}"

        llm_formatted_prompt = f"{system_prompt}\n\nInstruction: {instruction_text}\n\nInput: {data_text}"

        llm_response = "[Mock LLM Response]"

        return {
            "status": "success",
            "detected_attack": data_result.attack_type if not data_result.is_safe else None,
            "applied_defense": applied_defense,
            "final_prompt_sent_to_llm": llm_formatted_prompt,
            "response": llm_response
        }