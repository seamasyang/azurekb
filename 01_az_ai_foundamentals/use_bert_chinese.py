
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')
model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase')

def paraphrase(sentence):
    inputs = tokenizer([sentence], max_length=60, truncation=True, return_tensors="pt")
    outputs = model.generate(**inputs, num_beams=5, num_return_sequences=5)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

sentence = "How to enable Excel KPMG add-in"
result = paraphrase(sentence)
print(result)