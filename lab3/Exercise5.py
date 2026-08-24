import ollama

# Global system prompt template
SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional translator. "
    "Your task is to translate the user's sentence into {target_language}. "
    "The translation must strictly use a {formality} tone. "
    "Provide ONLY the translated sentence, without any extra explanations, notes, or conversational filler."
)

def generate_translation(sentence, target_language, formality):
    """
    Translates a sentence into a target language at a given formality level using Ollama.
    """
    model_name = "qwen2.5:0.5b"
    
    # Format the global system prompt with the provided inputs
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        target_language=target_language, 
        formality=formality
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sentence}
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
    print("🌍 Multi-Language Translator")
    print("=" * 60)
    
    # Get settings from user
    target_language = input("Enter target language (e.g., Spanish, Japanese): ").strip()
    formality = input("Enter formality level (e.g., formal, informal, slang): ").strip()
    
    # Fallbacks in case of empty input
    if not target_language:
        target_language = "Spanish"
    if not formality:
        formality = "formal"
        
    print(f"\nConfiguration saved! Target Language: {target_language} | Formality: {formality}")
    print("Type your sentence to translate below. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            sentence = input("Sentence: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if sentence.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
            
        if not sentence:
            continue
            
        print("\nTranslating...")
        translation = generate_translation(sentence, target_language, formality)
        print(f"\nTranslation:\n{translation}\n")
