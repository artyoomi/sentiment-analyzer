import os

import torch

import torch.nn as nn
import numpy as np

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from datasets import Value, load_dataset
from sklearn.metrics import accuracy_score, f1_score


class GeoReviewsDataset2023:
    """Wrapper to wrap geo-reviews-dataset specifics."""

    _DATASET = None

    @classmethod
    def _load_dataset(cls):
        if cls._DATASET is None:
            cls._DATASET = load_dataset("d0rj/geo-reviews-dataset-2023")["train"]

            max_samples = 50000
            # Cut dataset size if specified
            if max_samples is not None and max_samples < len(cls._DATASET):
                cls._DATASET = cls._DATASET.select(range(max_samples))
        return cls._DATASET

    @classmethod
    def get_train_test_datasets(cls, tokenizer, test_size=0.2, seed=42, max_length=512):
        """Return tokenized train and test datasets."""

        # In initial dataset target column named `rating`, but Trainer by default waits `label` column.
        dataset = (
            cls._load_dataset()
            .rename_column("rating", "label")
            .cast_column("label", Value("int64"))
        )

        split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )

        tokenized_train = split_dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        tokenized_test = split_dataset["test"].map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        # Set PyTorch format
        tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        labels = tokenized_train["label"]

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )

        class_weights = torch.tensor(class_weights, dtype=torch.float)

        return tokenized_train, tokenized_test, class_weights


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

class SentimentClassifier(nn.Module):
    """Combined model: RuBERT + classifier."""

    _MODEL_NAME: str = "cointegrated/rubert-tiny2"

    def __init__(self, bert, num_classes=6, freeze_bert=False):
        super().__init__()

        self.bert = bert
        self.classifier = SentimentHead(
            input_dim=self.bert.config.hidden_size,
            hidden_dim=256,
            num_classes=num_classes,
            dropout=0.2
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    @classmethod
    def model_name(cls):
        return cls._MODEL_NAME

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        # Trainer (and BERT tokenizers) may pass token_type_ids; ignore unknown kwargs.
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Use [CLS] token (first token)
        # cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = outputs.last_hidden_state.mean(dim=1)

        logits = self.classifier(cls_embedding)

        # Calculate loss, if labels is present
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weights=)
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


def load_checkpoint_state_dict(save_dir: str, map_location="cpu"):
    """Loads full model weights saved by ``Trainer.save_model`` (``.bin`` or ``.safetensors``)."""
    bin_path = os.path.join(save_dir, "pytorch_model.bin")
    safe_path = os.path.join(save_dir, "model.safetensors")
    if os.path.isfile(bin_path):
        try:
            state = torch.load(bin_path, map_location=map_location, weights_only=True)
        except TypeError:
            state = torch.load(bin_path, map_location=map_location)
        return state, bin_path
    if os.path.isfile(safe_path):
        from safetensors.torch import load_file

        device = map_location if isinstance(map_location, str) else str(map_location)
        return load_file(safe_path, device=device), safe_path
    raise FileNotFoundError(
        f"No model weights in {save_dir!r}: expected pytorch_model.bin or model.safetensors"
    )


def load_sentiment_classifier(save_dir: str, map_location="cpu"):
    """Rebuild ``SentimentClassifier`` and load trained weights (deterministic inference).

    We do not call ``AutoModel.from_pretrained(save_dir)`` here because the training
    checkpoint saved by ``Trainer.save_model`` is a plain state_dict for the custom
    wrapper model and does not include a Hugging Face ``config.json``. Instead, we
    reconstruct the backbone from the original model name and then load the trained
    state dict on top of it.
    """
    model_meta_path = os.path.join(save_dir, "model_info.json")
    base_model_name = SentimentClassifier.model_name()
    num_classes = 6

    if os.path.isfile(model_meta_path):
        import json

        with open(model_meta_path, "r", encoding="utf-8") as f:
            model_meta = json.load(f)
        base_model_name = model_meta.get("base_model_name", base_model_name)
        num_classes = int(model_meta.get("num_classes", num_classes))

    bert_model = AutoModel.from_pretrained(base_model_name)
    model = SentimentClassifier(bert_model, num_classes=num_classes)
    state_dict, _ = load_checkpoint_state_dict(save_dir, map_location=map_location)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def compute_metrics(eval_pred):
    """Classification metrics for sentiment classification."""
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    accuracy = accuracy_score(labels, predictions)

    # Macro F1 gives equal importance to every class
    f1 = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def train_model(
    model_name='cointegrated/rubert-tiny2',
    output_dir='./results',
    save_dir='./pretrained',
    num_epochs=3,
    freeze_bert=False,
    use_cpu=False,
):
    """Model trainer."""

    # Detect main processor
    if use_cpu:
        device = torch.device("cpu")
        print("Forcibly use CPU")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)

    # Get tokenized datasets
    tokenized_train, tokenized_test, class_weights = GeoReviewsDataset2023.get_train_test_datasets(tokenizer)

    # Create model (BERT + classifier)
    model = SentimentClassifier(bert_model, num_classes=6, freeze_bert=freeze_bert)

    # Train setup
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
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    print("Start training...")
    trainer.train()

    # Save model
    save_directory = save_dir
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)

    # Save lightweight metadata so inference can rebuild the correct backbone.
    import json

    with open(os.path.join(save_directory, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model_name": model_name,
                "num_classes": 6,
                "freeze_bert": freeze_bert,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Model saved in {save_directory}")

    return model, tokenizer


def predict_sentiment(text, model, tokenizer, device='cpu'):
    """Predict sentiment for single text."""
    was_training = model.training
    model.eval()
    model.to(device)

    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        prediction = torch.argmax(logits, dim=-1).item()
    if was_training:
        model.train()
    return prediction
