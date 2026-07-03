import torch
import torch.nn as nn
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder

# Import from our own scripts
from model import IntentClassifier
from train import train_epoch, evaluate
from predict import predict_intent

# ── 1. Load data ───────────────────────────────────────────────
print("=" * 50)
print("Step 1: Loading data...")
print("=" * 50)

url_train = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
url_test = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

train_df = pd.read_csv(url_train)
test_df = pd.read_csv(url_test)

print(f"  Training examples : {len(train_df)}")
print(f"  Test examples     : {len(test_df)}")
print(f"  Number of intents : {train_df['category'].nunique()}")

# ── 2. Encode labels ───────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 2: Encoding labels...")
print("=" * 50)

le = LabelEncoder()
le.fit(train_df['category'])
train_df['label'] = le.transform(train_df['category'])
test_df['label'] = le.transform(test_df['category'])

print(f"  Total classes: {len(le.classes_)}")

# ── 3. Tokenizer + Dataset ─────────────────────────────────────
print("\n" + "=" * 50)
print("Step 3: Preparing tokenizer and datasets...")
print("=" * 50)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_loader = DataLoader(
    IntentDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer),
    batch_size=32, shuffle=True
)
test_loader = DataLoader(
    IntentDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer),
    batch_size=32, shuffle=False
)

print(f"  Training batches : {len(train_loader)}")
print(f"  Test batches     : {len(test_loader)}")

# ── 4. Model setup ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 4: Setting up model...")
print("=" * 50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Using device: {device}")

model = IntentClassifier(num_classes=77).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── 5. Training ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 5: Training...")
print("=" * 50)

NUM_EPOCHS = 3

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 40)

    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, criterion, device
    )
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
    print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.1f}%")

# ── 6. Save model ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 6: Saving model and results...")
print("=" * 50)

torch.save(model.state_dict(), '../model.pth')
print("  Model saved to model.pth")

results = {
    "train_accuracy": train_acc,
    "test_accuracy": test_acc,
    "test_loss": test_loss,
    "num_intents": 77,
    "training_examples": len(train_df),
    "test_examples": len(test_df)
}

with open('../results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("  Results saved to results.json")

# ── 7. Demo predictions ────────────────────────────────────────
print("\n" + "=" * 50)
print("Step 7: Demo predictions")
print("=" * 50)

test_sentences = [
    "I lost my card",
    "what is my account balance?",
    "I want to transfer money to my friend",
    "why do I need to verify my identity?",
    "my card payment was not recognised"
]

for sentence in test_sentences:
    intent = predict_intent(sentence, model, le, device)
    print(f"  '{sentence}'")
    print(f"   → {intent}\n")

print("=" * 50)
print("All done!")
print("=" * 50)