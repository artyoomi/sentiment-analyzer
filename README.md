## Description

Sentiment analyzer based on RuBERT model with classification head. Trained on
GeoReviews2023 dataset from Yandex Maps. Can't detect sarcasm.

## Usage
```
python3 -m venv .venv
uv pip install -r requirements.txt

python3 cli.py --help
```

## Examples
```
python3 cli.py predict --text "Ходили в это заведение с семьей всегда и все устраивало. Сегодня пришли с ребенком, заказали пиццу пиперони…через какое-то время принесли пиццу  Маргарита.Я пригласила официанта и попросила уточнить заказ. Девушка забрала пиццу,через минуты 4 вернули пиццу с «извинениями». Они просто накидали сверху колбасы на Маргариту и редели,что и так сойдет.Горе-повора. Больше мы не гости данного заведения."
...
Analysis result:
  Text: Ходили в это заведение с семьей всегда и все устраивало. Сегодня пришли с ребенком, заказали пиццу п...
  Predicted rating: 1/5, confidence: 88.26%
```
