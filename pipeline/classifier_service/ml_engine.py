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

        # open prompt injection dataset
        self.defense_mapping = {
            "naive": "instructional",
            "ignore": "adversarial_repeat",   # fight fire with fire
            "escape": "delimiter_flood",       # drown the escape attempt
            "combine": "completion_poison"     # pre-empt the completion stage
        }

        # # xxz224 dataset
        # self.defense_mapping = {
        #     "naive":           "instructional",
        #     "ignore":          "adversarial_repeat",
        #     "escape":          "delimiter_flood",
        #     "combine":         "completion_poison",
        #     "fake_completion": "completion_poison",   # pre-injected completion neutralises the fake one
        # }

        # # map your attack classes to defense strategies
        # self.defense_mapping = {
        #     "naive": "instructional",
        #     "ignore": "sandwich",
        #     "escape": "isolation",
        #     "combine": "spotlight"
        # }
        # self.defense_mapping = {
        #     "naive": "instructional",
        #     "ignore": "sandwich",
        #     "fake_completion": "isolation",
        #     "combine": "spotlight"
        # }
        self.safe_labels = ["benign"]

        print(f"[{self.classifier_id}] Ready on {self.device}.")

    def evaluate(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        predicted_class_id = logits.argmax(dim=-1).item()
        predicted_label = self.model.config.id2label[predicted_class_id].lower()
        print(f"[{self.classifier_id}] predicted = {predicted_label}")
        if predicted_label in self.safe_labels:
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