def generate_report(llm, context, query, mode="General"):

    # =========================
    # GENERAL REPORT
    # =========================
    if mode == "General":

        prompt = f"""
You are an AI research assistant.

Generate a clear, structured and informative research report using ONLY the provided context.

Avoid hallucination. Do not add information not present in context.

Topic: {query}

Context:
{context}

Report Structure:

1. Introduction  
- Brief overview of the topic  

2. Key Concepts  
- Explain main ideas clearly  

3. Applications  
- Real-world uses  

4. Challenges  
- Limitations or issues  

5. Future Scope  
- Possible advancements  

6. Conclusion  
- Summary of insights  

Ensure the report is concise, well-structured, and easy to understand.
"""

    # =========================
    # 🔥 CLINICAL REPORT MODE
    # =========================
    else:

        prompt = f"""
You are a clinical AI research assistant.

Generate a medically accurate, structured clinical report using ONLY the provided context.

Strictly avoid hallucination.
Base all statements on the given context.

Topic: {query}

Context:
{context}

Report Structure:

1. Clinical Overview  
- Brief description of the condition  

2. Symptoms and Presentation  
- Key clinical symptoms  

3. Diagnosis and Findings  
- Diagnostic methods and observations  

4. Treatment and Management  
- Medical treatment approaches  

5. Clinical Challenges  
- Complications or risks  

6. Future Scope in Healthcare  
- Emerging treatments or research  

7. Conclusion  
- Clinical summary  

Ensure:
- Clinical reasoning is clear  
- Information is precise  
- Language is professional and structured  
"""

    response = llm.invoke(prompt)

    return response.content
