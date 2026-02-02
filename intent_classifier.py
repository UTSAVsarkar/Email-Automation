from transformers import pipeline

classifier = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def is_customer_inquiry(email_text):
    prompt = f"""
You are an email intent classifier.

Classify the email into ONE of the following:
- Customer Inquiry
- Marketing / Promotion
- Newsletter
- System Notification

Examples:
Email: "Where is my order?"
Label: Customer Inquiry

Email: "Renew now and get 30% off"
Label: Marketing / Promotion

Email: "Your OTP for login is 123456"
Label: System Notification

Email:
{email_text}

Label:
"""

    result = classifier(prompt, max_length=10)[0]['generated_text'].lower()

    return "customer" in result