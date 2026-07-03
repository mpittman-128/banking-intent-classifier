import torch
from transformers import DistilBertTokenizer

def predict_intent(text, model, label_encoder, device, max_length=64):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model.eval()

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predicted = outputs.argmax(dim=1)

    return label_encoder.inverse_transform(predicted.cpu().numpy())[0]