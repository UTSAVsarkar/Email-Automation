from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_reply(email_text):
    prompt = f"""
You are a professional customer support assistant.

Rules:
- Ignore marketing content.
- Focus only on user intent.
- Keep response concise and professional.

Example 1:
Customer: Where is my order?
Response: Your order is currently being processed and will be shipped soon.

Example 2:
Customer: I want a refund.
Response: Sure, please share your order ID to help us proceed.

Customer message:
{email_text}
"""

    response = generator(prompt, max_length=150)
    return response[0]['generated_text']