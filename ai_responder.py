from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_reply(text: str) -> str:
    prompt = f"""
You are a customer support assistant.

Write a polite and helpful reply to the following customer email.

Customer email:
{text}

Reply:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply.strip()
