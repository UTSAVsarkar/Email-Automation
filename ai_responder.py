from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

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
"""

    output = generator(
        prompt,
        max_new_tokens=5,
        do_sample=False
    )[0]["generated_text"]

    return "CUSTOMER_INQUIRY" in output
