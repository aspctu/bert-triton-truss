from typing import Any, List
from transformers import BertTokenizer, BertModel
from pydantic import BaseModel, conlist


class Input(BaseModel):
    text: str


class Output(BaseModel):
    embedding: conlist(float, min_length=768, max_length=768)
    text: str


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None
        self._tokenizer = None

    def load(self):
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self._model = BertModel.from_pretrained("bert-base-uncased").to("cuda")

    def predict(self, model_input: List[Input]) -> List[Output]:
        model_output = []
        model_input_text = []
        print(f"Predict was invoked with batch of {len(model_input)}")
        for i in model_input:
            model_input_text.append(i.text)

        inputs_tok = self._tokenizer(
            model_input_text, return_tensors="pt", padding=True
        ).to("cuda")
        outputs = self._model(**inputs_tok)

        for output in outputs.pooler_output:
            model_output.append(
                Output(
                    embedding=output.detach().cpu().numpy(),
                    text=i.text,
                )
            )
        return model_output
