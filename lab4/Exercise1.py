# job posting to candidate outreach pipeline 
# build a 2 step pipeline. step 1 extract the key requirement from a raw job posting into structure from.
# step 2 using that structure form generate a personalized outreach message.

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_multiline_input(prompt_text):
    """Captures multi-line user input until an empty line or EOF is encountered."""
    print(prompt_text)
    print("(Paste text, then press Enter on an empty line to confirm):")
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                if lines:  # Finish on empty line if content already exists
                    break
            else:
                lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()

def get_completion(client, prompt):
    completion = client.chat.completions.create(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384},
        stream=True
    )

    result = ""
    for chunk in completion:
        if not chunk.choices:
            continue
        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
        if reasoning:
            print(reasoning, end="", flush=True)
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            result += content
    print()
    return result

def generate_outreach_pipeline(job_posting_text, candidate_profile):
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable is not set.")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

    print("\n--- Step 1: Extracting Structured Requirements ---")
    step1_prompt = f"""
    Extract the key requirements (skills, experience, tools, education) from the following job posting and present them in a structured bulleted format.

    Raw Job Posting:
    {job_posting_text}
    """
    structured_requirements = get_completion(client, step1_prompt)

    print("\n--- Step 2: Generating Personalized Outreach Message ---")
    step2_prompt = f"""
    You are an executive tech recruiter. Write a concise, personalized outreach message to the following candidate.
    Align their specific background with the structured requirements extracted below to explain why they are a strong fit.

    Structured Requirements:
    {structured_requirements}

    Candidate Profile:
    {candidate_profile}
    """
    outreach_message = get_completion(client, step2_prompt)

    return structured_requirements, outreach_message

if __name__ == "__main__":
    print("=" * 60)
    print("AI RECRUITER: JOB POSTING TO OUTREACH PIPELINE")
    print("=" * 60 + "\n")

    user_job_posting = get_multiline_input("Enter the Raw Job Posting:")
    if not user_job_posting:
        print("Error: Job posting cannot be empty.")
        sys.exit(1)

    print("\n" + "-" * 40 + "\n")

    user_candidate_profile = get_multiline_input("Enter the Candidate Profile / Resume:")
    if not user_candidate_profile:
        print("Error: Candidate profile cannot be empty.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PROCESSING PIPELINE...")
    print("=" * 60)

    structured_reqs, message = generate_outreach_pipeline(user_job_posting, user_candidate_profile)

    print("\n" + "=" * 25 + " FINAL OUTREACH MESSAGE " + "=" * 25 + "\n")
    print(message)