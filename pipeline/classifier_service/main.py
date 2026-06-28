import os
from fastapi import FastAPI
from contextlib import asynccontextmanager

from schemas import EvaluationRequest, EvaluationResponse
from ml_engine import SafetyClassifier

SERVICE_TYPE = os.getenv("SERVICE_TYPE", "data")  # defaults to "data"

model_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_engine
    if SERVICE_TYPE == "instruction":
        model_engine = SafetyClassifier(
            # model_name="/home/rowan/Rowans_CSE/Term10/Team-Sherlock/models/bert_models/roberta-large/checkpoint-67200",  # Replace with your model path
            # model_name="/home/rowan/Rowans_CSE/Term10/Team-Sherlock/models/bert_models/roberta-base/checkpoint-168000",  # Replace with your model path
            # model_name = "/home/rowan/Rowans_CSE/Term10/Team-Sherlock/models/bert_models/roberta-base-xxz224/checkpoint-20241",
            # model_name = "/home/rowan/Rowans_CSE/Term10/Team-Sherlock/models/bert_models/roberta-large-xxz224/checkpoint-22490",
            model_name="/home/rowan/Rowans_CSE/Term10/Team-Sherlock/models/bert_models/roberta-base-xxz224-no-escape/checkpoint-18740",
            classifier_id="Instruction_Safety_v1"
        )
    else:
        model_engine = SafetyClassifier(
            # model_name="/home/rowan/Rowans_CSE/Term10/Team-Sherlock/models/bert_models/roberta-large/checkpoint-67200",  # Replace with your model path
            # model_name="/home/rowan/Rowans_CSE/Term10/Team-Sherlock/models/bert_models/roberta-base/checkpoint-168000",  # Replace with your model path
            # model_name = "/home/rowan/Rowans_CSE/Term10/Team-Sherlock/models/bert_models/roberta-base-xxz224/checkpoint-20241",
            # model_name = "/home/rowan/Rowans_CSE/Term10/Team-Sherlock/models/bert_models/roberta-large-xxz224/checkpoint-22490",
            model_name="/home/rowan/Rowans_CSE/Term10/Team-Sherlock/models/bert_models/roberta-base-xxz224-no-escape/checkpoint-18740",
            classifier_id="Data_Safety_v1"
        )
    yield
    print("Shutting down ML Engine...")
    model_engine = None


app = FastAPI(title=f"Classifier API ({SERVICE_TYPE.upper()})", lifespan=lifespan)


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_text(request: EvaluationRequest):
    result_dict = model_engine.evaluate(request.text)

    return EvaluationResponse(**result_dict)