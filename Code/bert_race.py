import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


class RaceClassifier(nn.Module):

    def __init__(self, n_classes):
        super(RaceClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
        self.drop = nn.Dropout(p=0.3)  # can be changed in future
        self.out = nn.Linear(self.bert.config.hidden_size,
                             n_classes)  # linear layer for the output with the number of classes

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = bert_output[0]
        pooled_output = last_hidden_state[:, 0]
        output = self.drop(pooled_output)
        return self.out(output)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_classes = 4  # 4 races
model = RaceClassifier(n_classes)

model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.eval()
le = LabelEncoder()
le.fit_transform(['white', 'latin', 'asian', 'afr_amr'])


def text_to_loader(tokenizer, texts, max_len):
    encoding = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    return [{
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
    }]


def predict_probabilities(model, dataloader):
    model = model.eval()
    all_probs = []
    predictions = []
    with torch.no_grad():
        for item in dataloader:
            input_ids = item['input_ids'].to(device)
            attention_mask = item['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs)

    return all_probs, [le.inverse_transform([pred.item()])[0] for pred in predictions]


unseen_texts = ["i love maths"]
true_labels = ["afr_amr"]

MAX_LEN = 100
test_dataloader = DataLoader(text_to_loader(tokenizer, unseen_texts, MAX_LEN), batch_size=1)
probabilities = predict_probabilities(model, test_dataloader)


def return_results(unseen_texts, true_labels, probabilities):
    for text, true, probs in zip(unseen_texts, true_labels, probabilities[0]):
        label_probs = {le.inverse_transform([i])[0]: prob.item() for i, prob in enumerate(probs[0])}
        """print(f'Text: {text}')
        print(f'Label Probabilities: {label_probs}, Actual: {true}\n')"""
        return label_probs, probabilities[1]
