import ollama

def run_mini_chatbot():
    model_name = "qwen2.5:0.5b"
    
    # Store conversation state across turns
    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."}
    ]
    
    print(f"Mini Chatbot ({model_name}) ready! Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in {"exit", "quit"}:
            print("Session ended.")
            break
            
        if not user_input:
            continue
            
        # 1. Append user input to history
        messages.append({"role": "user", "content": user_input})
        
        # 2. Send complete conversation buffer to the local LLM
        response = ollama.chat(
            model=model_name,
            messages=messages
        )
        
        reply = response["message"]["content"]
        print(f"\nAssistant: {reply}\n")
        
        # 3. Append assistant response to preserve context for the next turn
        messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    run_mini_chatbot()