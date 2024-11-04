
import pandas as pd

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, T5Tokenizer, RobertaTokenizerFast
from transformers import T5ForConditionalGeneration, TrainingArguments, Trainer, AutoModelForSeq2SeqLM, RobertaForSequenceClassification


import torch
from torch.utils.data import Dataset

from loguru import logger

class QADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.data[idx].items()}

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


def load_data():
    df = pd.read_csv("hf://datasets/CShorten/CDC-COVID-FAQ/CDC-COVID-FAQ.csv")

    # Remove duplicates
    df.drop_duplicates(subset=['question'], inplace=True)

    # Handle missing values
    df.dropna(subset=['question', 'answer'], inplace=True)

    # Lowercase the text
    df['question'] = df['question'].str.lower()
    df['answer'] = df['answer'].str.lower()

    # Split the DataFrame into train and validation sets (80/20 split)
    train_df, eval_df = train_test_split(df, test_size=0.2)

    # for idx, row in train_df.iterrows():
    #     logger.info(f"training ds: idx:{idx}, question={row.get('question')};\nanswer={row.get('answer')}")

    # for idx, row in eval_df.iterrows():
    #     logger.info(f"eval ds: idx:{idx}, question={row.get('question')};\nanswer={row.get('answer')}")

    return train_df, eval_df


def preprocess(df):
    tokenized_data = []
    for _, row in df.iterrows():
        question = f"question: {row['question']}"
        answer = row['answer']
        
        inputs = tokenizer(question, max_length=256, truncation=True, padding="max_length")
        labels = tokenizer(answer, max_length=512, truncation=True, padding="max_length")["input_ids"]
        # Set padding tokens in labels to -100, so they’re ignored in loss calculation
        labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]
        inputs["labels"] = labels
        #logger.debug(f"labels = {labels}")
        tokenized_data.append(inputs)
    return tokenized_data


def prepare_dataset():
    # Apply preprocessing
    train_df, eval_df = load_data()
    logger.debug('train and eval data loaded')
    train_tokenized_data = preprocess(train_df)
    eval_tokenized_data = preprocess(eval_df)
    logger.debug('data tokenized')

    # Create PyTorch Dataset objects for training and evaluation
    train_dataset = QADataset(train_tokenized_data)
    eval_dataset = QADataset(eval_tokenized_data)
    logger.debug(f'format as qa dataset; train: {len(train_dataset)}, eval:{len(eval_dataset)}')

    return train_dataset, eval_dataset


def fine_tune():
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    train_dataset, eval_dataset = prepare_dataset()

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Start training
    trainer.train()

    model.save_pretrained("./fine_tuned_roberta_qa")
    tokenizer.save_pretrained("./fine_tuned_roberta_qa")

def infer(question):
    model_dir = f"./fine_tuned_roberta_qa"

    input_text=f"question:{question}"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    output = model.generate(inputs["input_ids"], 
        max_length=256,  # Increase this if needed
        do_sample=True,
        num_beams=5,     # Adjust as needed
        early_stopping=False  # Set to False to avoid premature stopping
    )
    logger.debug(output)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

fine_tune()

question = "Should chest CT be used for diagnosis of COVID-19?"
answer = infer(question)
logger.info(answer)

# load_data()