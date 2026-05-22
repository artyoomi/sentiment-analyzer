import numpy as np

from datasets import Value, load_dataset
from sklearn.utils.class_weight import compute_class_weight


class GeoReviewsDataset2023:
    """Wrapper for geo-reviews-dataset-2023."""

    _DATASET = None

    @classmethod
    def _load_dataset(cls, max_samples: int = 0):
        if cls._DATASET is None:
            cls._DATASET = load_dataset("d0rj/geo-reviews-dataset-2023")["train"]
            if max_samples and max_samples < len(cls._DATASET):
                cls._DATASET = cls._DATASET.select(range(max_samples))
        return cls._DATASET

    @classmethod
    def get_train_test_datasets(
        cls,
        tokenizer,
        test_size: float = 0.2,
        seed: int = 42,
        max_length: int = 512,
    ):
        dataset = (
            cls._load_dataset()
            .rename_column("rating", "labels")          # Trainer looks for "labels"
            .cast_column("labels", Value("int64"))
        )

        split = dataset.train_test_split(test_size=test_size, seed=seed)

        def tokenize(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        cols_to_remove = ["text", "name_ru", "address", "rubrics"]  # drop non-tensor columns

        def safe_remove(ds):
            return [c for c in cols_to_remove if c in ds.column_names]

        tokenized_train = split["train"].map(tokenize, batched=True, remove_columns=safe_remove(split["train"]))
        tokenized_test  = split["test"].map(tokenize,  batched=True, remove_columns=safe_remove(split["test"]))

        tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        tokenized_test.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

        # Compute class weights from training labels
        labels = np.array(tokenized_train["labels"])
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels,
        )
        print(f"Class weights: {class_weights} for labels: {np.unique(labels)}")

        return tokenized_train, tokenized_test, class_weights.tolist()
