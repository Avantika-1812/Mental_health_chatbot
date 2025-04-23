from transformers import BertTokenizer, BertModel
import torch
import joblib
import pandas as pd

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load the dataset
df = pd.read_csv("model/20200325_counsel_chat.csv", encoding="utf-8")

# Generate question embeddings
# Generate question embeddings
questions = df['questionText'].tolist()  # Use the correct column name for questions
question_embeddings = []
for question in questions:
    inputs = tokenizer(question, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    question_embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
joblib.dump(question_embeddings, 'model/questionembedding.dump')

# Generate answer embeddings
answers = df['answerText'].tolist()  # Use the correct column name for answers
answer_embeddings = []
for answer in answers:
    inputs = tokenizer(answer, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    answer_embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
joblib.dump(answer_embeddings, 'model/ansembedding.dump')

print("Embeddings generated and saved successfully!")