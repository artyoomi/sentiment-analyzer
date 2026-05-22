import torch

import torch.nn as nn
import numpy as np


from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    PreTrainedConfig,
    PreTrainedModel
)
from sklearn.metrics import accuracy_score, f1_score

from dataset import GeoReviewsDataset2023


class SentimentConfig(PreTrainedConfig):
    model_type = "sentiment_classifier"

    def __init__(
        self,
        base_model_name: str = "conintegrated/rubert-tiny2",
        num_labels: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        freeze_bert: bool = False,
        class_weights: list = [],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.freeze_bert = freeze_bert
        self.class_weights = class_weights  # stored as plain list, JSON-serialisable

class SentimentHead(nn.Module):
    """Simple classifier head."""

    def __init__(self, input_dim=312, hidden_dim=256, num_classes=6, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class SentimentClassifier(PreTrainedModel):
    """Combined model: RuBERT + classifier."""

    config_class = SentimentConfig

    def __init__(self, config: SentimentConfig):
        super().__init__(config)

        # Backbone - loaded from name stored in config
        bert_cfg = AutoConfig.from_pretrained(config.base_model_name)
        self.bert = AutoModel.from_config(bert_cfg)

        # Turn off BERT training if necessary
        if config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classification head
        input_dim = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_labels),
        )

        # Class weights as a buffer - moves to device automatically, saved with model
        if config.class_weights != []:
            self.register_buffer(
                "class_weights",
                torch.tensor(config.class_weights, dtype=torch.float),
            )
        else:
            self.class_weights = None

        # Required by PreTrainedModel
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Mean-pool over token dimension
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)

        # Trainer expects an object with .loss and .logits, or a plain dict
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)


def compute_metrics(eval_pred):
    """Compute metrics for model."""

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1":       float(f1_score(labels, preds, average="macro")),
    }

def train_model(
    base_model_name: str = "cointegrated/rubert-tiny2",
    output_dir: str = "./results",
    save_dir: str = "./pretrained",
    num_epochs: int = 3,
    freeze_bert: bool = False,
    use_cpu: bool = False,
):
    device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenized_train, tokenized_test, class_weights = GeoReviewsDataset2023.get_train_test_datasets(tokenizer)

    config = SentimentConfig(
        base_model_name=base_model_name,
        num_labels=6,
        hidden_dim=256,
        dropout=0.2,
        freeze_bert=freeze_bert,
        class_weights=class_weights,
    )
    model = SentimentClassifier(config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=num_epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # save_pretrained writes config.json + model weights - no manual JSON needed
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_sentiment_classifier(save_dir: str, map_location: str = "cpu") -> SentimentClassifier:
    """Rebuild model from a save_pretrained directory."""
    # model = SentimentClassifier.from_pretrained(save_dir, map_location=map_location)
    model = SentimentClassifier.from_pretrained(save_dir)

    model.eval()
    return model


def predict_sentiment(text: str, model: SentimentClassifier, tokenizer, device: str = "cpu") -> int:
    model.eval()
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return torch.argmax(outputs.logits, dim=-1).item()
