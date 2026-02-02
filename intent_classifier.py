from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def is_customer_inquiry(email_text: str) -> bool:
    prompt = f"""
You are an email intent classifier.

Classify the email into ONE of the following labels:
Customer Inquiry
Marketing / Promotion
Newsletter
System Notification

Examples:
Email: Where is my order?
Label: Customer Inquiry

Email: Renew now and get 30% off
Label: Marketing / Promotion

Email: Your OTP for login is 123456
Label: System Notification

Email: {email_text}
Label:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False
    )

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

    return prediction.startswith("customer")
