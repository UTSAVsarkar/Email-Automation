from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_reply(text: str) -> bool:
    prompt = f"""
Classify the email as CUSTOMER_INQUIRY or OTHER.

Examples:
Email: Where is my order?
Label: CUSTOMER_INQUIRY

Email: I want a refund for my purchase
Label: CUSTOMER_INQUIRY

Email: Thanks for your support
Label: OTHER

Email: {text}
Label:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False
    )

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return prediction.strip().upper().startswith("CUSTOMER_INQUIRY")
