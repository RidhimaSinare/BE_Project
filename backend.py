from transformers import PreTrainedModel, PretrainedConfig, BertModel
import torch.nn as nn
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Define label mappings
label_mappings = [
    {0: "HATE", 1: "NOT", 2: "OFFN", 3: "PRFN"},
    {0: "Positive", 1: "Neutral", 2: "Negative"},
    {0: "Auto", 1: "Bhakti", 2: "Crime", 3: "Education", 4: "Fashion", 5: "Health", 6: "International", 7: "Manoranjan", 8: "Politics", 9: "Sports", 10: "Tech", 11: "Travel"},
    {0: "Auto", 1: "Bhakti", 2: "Crime", 3: "Education", 4: "Fashion", 5: "Health", 6: "International", 7: "Manoranjan", 8: "Politics", 9: "Sports", 10: "Tech", 11: "Travel"},
    {0: "HOF", 1: "NOT"}]

class BertMultiOutputConfig(PretrainedConfig):
    model_type = "bert-multi-output"
    
    def __init__(self, bert_model_name="l3cube-pune/marathi-bert-v2", hidden_size=16, output_sizes=None, **kwargs):
        super().__init__(**kwargs)
        self.bert_model_name = bert_model_name
        self.hidden_size = hidden_size
        self.output_sizes = output_sizes if output_sizes is not None else []
        
class BertMultiOutputModel(PreTrainedModel):
    config_class = BertMultiOutputConfig
    
    def __init__(self, config):
        super().__init__(config)
        # Load the underlying BERT model using the model name from the config
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        num_tasks = len(config.output_sizes)
        self.fc_branches = nn.ModuleList(
            [nn.Linear(self.bert.config.hidden_size, config.hidden_size) for _ in range(num_tasks)]
        )
        self.output_layers = nn.ModuleList(
            [nn.Linear(config.hidden_size, output_size) for output_size in config.output_sizes]
        )
        self.dropout = nn.Dropout(0.1)
        self.post_init()  # initializes weights following Hugging Face conventions

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        # Extract [CLS] token's hidden state
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_hidden_state)
        outs = []
        for fc_layer, out_layer in zip(self.fc_branches, self.output_layers):
            branch = fc_layer(x)
            out = out_layer(branch)
            outs.append(out)
        return outs


AutoConfig.register("bert-multi-output", BertMultiOutputConfig)
AutoModel.register(BertMultiOutputConfig, BertMultiOutputModel)

loaded_model = AutoModel.from_pretrained("amoghhf123/multitask_model")
tokenizer = AutoTokenizer.from_pretrained("amoghhf123/multitask_model")


# FastAPI app
app = FastAPI()

class TextInput(BaseModel):
    text: str


@app.post("/predict")
async def predict(input_data: TextInput):
    # Tokenize input text
    inputs = tokenizer(input_data.text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get model outputs
    with torch.no_grad():
        outputs = loaded_model(input_ids, attention_mask)

    # Process predictions correctly for each output head
    mapped_predictions = []
    for i, output in enumerate(outputs):
        probabilities = output.softmax(dim=-1)  # Convert logits to probabilities
        predicted_index = torch.argmax(probabilities, dim=-1).item()  # Get highest probability index
        mapped_label = label_mappings[i].get(predicted_index, "Unknown")  # Map index to label
        mapped_predictions.append(mapped_label)

    return {"predictions": mapped_predictions}


# Run FastAPI with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
