import ollama

# Global system prompt template
SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful, empathetic, and professional customer support agent for {company_name}. "
    "Reply to the customer's message below. "
    "IMPORTANT: Your response must be strictly {maxwords} words or fewer. "
    "Maintain a consistent, polite tone regardless of whether the customer is happy, frustrated, or confused."
)

def generate_support_reply(customer_message, company_name, maxwords):
    """
    Generates a support reply to a customer message using a local LLM via Ollama.
    """
    model_name = "qwen2.5:0.5b"
    
    # Format the global system prompt with the provided inputs
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        company_name=company_name, 
        maxwords=maxwords
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": customer_message}
    ]
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error contacting Ollama: {e}"

if __name__ == "__main__":
    print("=" * 60)
    print("🛠️  Customer Support Reply Generator")
    print("=" * 60)
    
    # Get settings from user
    company_name = input("Enter your company name: ").strip()
    maxwords = input("Enter maximum words for the reply (e.g., 50): ").strip()
    
    # Fallbacks in case of empty input
    if not company_name:
        company_name = "Acme Corp"
    if not maxwords.isdigit():
        maxwords = "50"
        
    print(f"\nConfiguration saved! Company: {company_name} | Max Words: {maxwords}")
    print("Type your customer message below. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            customer_message = input("Customer: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if customer_message.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
            
        if not customer_message:
            continue
            
        print("\nGenerating reply...")
        reply = generate_support_reply(customer_message, company_name, maxwords)
        print(f"\nSupport Agent:\n{reply}\n")
