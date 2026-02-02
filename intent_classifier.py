from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

classifier = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

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

    output = classifier(
        prompt,
        max_new_tokens=8,
        do_sample=False,
        truncation=True
    )[0]["generated_text"]

    # Extract only the generated label (after "Label:")
    predicted_label = output.split("Label:")[-1].strip().lower()

    return predicted_label.startswith("customer")