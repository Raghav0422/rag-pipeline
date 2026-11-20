from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

def get_llm(query,relevant_docs):
    combined_input = f"""Instructions: You are a strict QA assistant. Based on the following documents please answer the questions {query}
    Documents: 
    {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

    Please provide the clear, helpful answer using only the information from these documets, if you didnot find any relevant answer return with message No relevant answer found
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                   temperature=0
                                   )

    messages = [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)

    print("\n ---- Generated Response ----")

    print("content only")
    print(result.content)