import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SafetyClassifier:
    def __init__(self, model_name: str, classifier_id: str):
        self.classifier_id = classifier_id
        print(f"[{self.classifier_id}] Loading model: {model_name}...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        self.defense_mapping = {
            "ignore": "sandwich",
            "completion_real": "spotlight",
            "escape_deletion": "isolation",
            "escape_separation": "isolation",
            "naive": "instructional",
            "injection": "spotlight",
        }

        print(f"[{self.classifier_id}] Ready on {self.device}.")

    def evaluate(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        predicted_class_id = logits.argmax(dim=-1).item()

        predicted_label = self.model.config.id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")
        predicted_label = predicted_label.lower()

        if predicted_label in ["safe", "benign", "none", "label_0"]:
            return {
                "is_safe": True,
                "classifier_name": self.classifier_id
            }

        recommended_defense = self.defense_mapping.get(predicted_label, "spotlight")

        return {
            "is_safe": False,
            "reason": f"Model detected attack pattern: {predicted_label}",
            "classifier_name": self.classifier_id,
            "attack_type": predicted_label,
            "recommended_defense": recommended_defense
        }