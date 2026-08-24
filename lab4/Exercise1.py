# job posting to candidate outreach pipeline 
# build a 2 step pipeline. step 1 extract the key requirement from a raw job posting into structure from.
# step 2 using that structure form generate a personalized outreach message.

import ollama

def generate_outreach_pipeline(job_posting_text, candidate_profile):
    model_name = "qwen2.5:0.5b" # Using the local Ollama model

    print("--- Step 1: Extracting structured requirements ---")
    step1_prompt = f"""
    Extract the key requirements (skills, experience, tools) from the following job posting and present them in a structured bulleted format.
    
    Raw Job Posting:
    {job_posting_text}
    """
    
    response_step1 = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": step1_prompt}]
    )
    
    structured_requirements = response_step1["message"]["content"]
    
    print("--- Step 2: Generating personalized outreach message ---")
    step2_prompt = f"""
    You are a recruiter. Write a short, personalized outreach message to the following candidate.
    Use ONLY the structured requirements provided below to explain why they are a good fit.
    
    Structured Requirements:
    {structured_requirements}
    
    Candidate Profile:
    {candidate_profile}
    """
    
    response_step2 = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": step2_prompt}]
    )
    
    outreach_message = response_step2["message"]["content"]
    
    return structured_requirements, outreach_message

if __name__ == "__main__":
    job_posting_test = """
    Senior Data Scientist
    We are looking for someone with at least 4 years of experience in Machine Learning and Python.
    Must have hands-on experience with deep learning frameworks like TensorFlow or PyTorch.
    Experience with cloud platforms like AWS or GCP is a big plus.
    """
    
    candidate_profile = """
    Name: John Doe
    Experience: 5 years in Data Science.
    Skills: Python, PyTorch, Scikit-learn, GCP (Google Cloud Platform), SQL.
    Projects: Built an image classification model using PyTorch deployed on GCP.
    """
    
    structured_reqs, message = generate_outreach_pipeline(job_posting_test, candidate_profile)
    
    print("\n================ STRUCTURED REQUIREMENTS (STEP 1) ================\n")
    print(structured_reqs)
    
    print("\n================ OUTREACH MESSAGE (STEP 2) ================\n")
    print(message)
