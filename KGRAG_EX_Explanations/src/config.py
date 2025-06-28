from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable medical assistant.

Use the following medical paragraph to answer the multiple-choice question below.
Choose the best answer from the provided options based solely on the paragraph.
                                              
Medical Paragraph:
{paragraph}
                                              
Question:
{question}
                                              
Options:
{options}
                                              
Answer:
Provide only the letter (A, B, C or D) corresponding to the corrected choice.
Do not provide any additional explanation or commentary.
""")