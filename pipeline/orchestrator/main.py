from fastapi import FastAPI, HTTPException
import httpx
import asyncio
from contextlib import asynccontextmanager

from schemas import UserRequest, ClassifierResponse
from defenses import apply_defense
from config import CLASSIFIER_URLS

# Import your custom Chatbot class!
from chatbot import HuggingfaceChatbot

# Global variable to hold our target LLM
target_llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global target_llm
    print("Loading Target LLM: google/gemma-2-2b-it...")
    # This will use the 4-bit BitsAndBytes config from your chatbot.py
    target_llm = HuggingfaceChatbot("google/gemma-2-2b-it")
    print("Target LLM Ready.")
    yield
    print("Shutting down Target LLM...")
    target_llm = None

app = FastAPI(title="LLM Security Gateway - Adaptive Defense", lifespan=lifespan)

async def call_classifier(client: httpx.AsyncClient, url: str, text: str) -> ClassifierResponse:
    try:
        response = await client.post(url, json={"text": text}, timeout=2.0)
        response.raise_for_status()
        return ClassifierResponse(**response.json())
    except httpx.RequestError as e:
        return ClassifierResponse(
            is_safe=False,
            reason=f"Network Error: {str(e)}",
            recommended_defense="isolation"
        )

@app.post("/generate")
async def generate_secure_response(request: UserRequest):
    instruction_text = request.instruction
    data_text = request.data

    async with httpx.AsyncClient() as client:
        # 1. Run the filter layer concurrently
        instruction_task = call_classifier(client, CLASSIFIER_URLS["instruction"], instruction_text)
        data_task = call_classifier(client, CLASSIFIER_URLS["data"], data_text)

        instruction_result, data_result = await asyncio.gather(instruction_task, data_task)

        # 2. Hard Gate: Block malicious instructions
        if not instruction_result.is_safe:
            raise HTTPException(
                status_code=403,
                detail=f"Blocked instruction. Detected: {instruction_result.attack_type or 'Unknown Attack'}"
            )

        # 3. Soft Mitigation: Defend against data poisoning
        applied_defense = "none"

        if not data_result.is_safe:
            target_defense = data_result.recommended_defense or request.fallback_defense
            applied_defense = target_defense

            print(f"Attack Detected. Applying adaptive defense: '{target_defense}'")

            defense_payload = apply_defense(instruction_text, data_text, target_defense)
            instruction_text = defense_payload["instruction"]
            data_text = defense_payload["data"]

        # 4. Format for Gemma
        # Because your chatbot.py already injects ad_tools.SYS_INPUT, we just combine
        # the instruction and data here before passing it to the respond() method.
        final_prompt = f"Instruction: {instruction_text}\n\nInput: {data_text}"

        # 5. Call Gemma!
        print("Passing safe prompt to Target LLM...")
        llm_response = target_llm.respond(final_prompt, defense_cross_prompt=False)

        return {
            "status": "success",
            "detected_attack": data_result.attack_type if not data_result.is_safe else None,
            "applied_defense": applied_defense,
            "final_prompt_sent_to_llm": final_prompt,
            "response": llm_response
        }