from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

_FINBERT_MODEL = "ProsusAI/finbert"

class FinBertSentiment:
    def __init__(self, device: int = -1):
        self.tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL)
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )

    def score_headlines(self, headlines):
        if isinstance(headlines, str):
            headlines = [headlines]
        outputs = self.pipe(headlines)
        scores = []
        for o in outputs:
            label = o["label"].lower()
            score = o["score"]
            if label == "positive":
                scores.append(score)
            elif label == "negative":
                scores.append(-score)
            else:
                scores.append(0.0)
        return scores
