import ollama

def run_club_recommender():
    model_name = "qwen2.5:0.5b"

    # System prompt enforcing role, recommendation constraints, and output format
    system_prompt = (
        "You are an expert college student advisor. Your job is to recommend relevant "
        "campus clubs based on the student's stated interests, hobbies, or academic focus. "
        "For each input, recommend exactly one or two specific clubs by name with a concise, "
        "one-line reason explaining why it matches their interest. If the student provides "
        "additional preferences later in the conversation, adapt your recommendations accordingly."
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 60)
    print("🎓 Campus Club Recommendation Bot")
    print(f"Loaded Model: {model_name}")
    print("Type your interests below. Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting...")
            break

        # Termination condition
        if user_input.lower() in {"exit", "quit"}:
            print("Good luck finding the right club! Goodbye.")
            break

        if not user_input:
            continue

        # 1. Append the new student interest turn to history
        messages.append({"role": "user", "content": user_input})

        try:
            # 2. Call the local model with the entire conversational history
            response = ollama.chat(
                model=model_name,
                messages=messages
            )

            assistant_reply = response["message"]["content"]
            print(f"\nAdvisor: {assistant_reply}\n")

            # 3. Append advisor response to preserve context for follow-up turns
            messages.append({"role": "assistant", "content": assistant_reply})

        except Exception as e:
            print(f"\nError contacting Ollama: {e}\n")
            # Remove the last user message so the prompt state isn't broken
            messages.pop()

if __name__ == "__main__":
    run_club_recommender()