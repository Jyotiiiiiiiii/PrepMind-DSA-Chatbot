import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from vector_store import retrieve_top_k
from logging_utils import log_query_metrics, estimate_tokens, RequestLogger

load_dotenv()

def get_llm():
    """Initialize and return the Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env file")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

SYSTEM_INSTRUCTION = """You are PrepMind, a strict AI study assistant exclusively for computer science students.

YOU MUST FOLLOW THESE RULES WITHOUT EXCEPTION:
1. You ONLY answer questions related to: Computer Science subjects (DSA, DBMS, Operating Systems, Computer Networks, System Design, General Programming), exam preparation, placement preparation, coding interviews, and technical concepts.
2. If the user asks ANYTHING outside of these topics — including but not limited to general knowledge, personal advice, current events, creative writing, math unrelated to CS, or small talk — you MUST refuse with exactly this message: "I'm PrepMind, your CS study assistant. I can only help with computer science topics, exam preparation, and placement prep. Please ask a relevant question."
3. Do NOT engage with off-topic questions under any circumstances, even if asked politely or creatively.
4. Never break character or reveal that you are an LLM.
"""

def build_prompt_text(query: str, mode: str, context: str = "") -> str:
    base_prompt = SYSTEM_INSTRUCTION
    base_prompt += "\n---\n"

    if context:
        base_prompt += f"Use the following extracted knowledge context to ground your answer. If the context does not contain the answer, use your general CS knowledge but mention it was not in the provided documents.\n\nContext:\n{context}\n\n"
    else:
        base_prompt += "Answer based on your general computer science knowledge.\n\n"

    base_prompt += f"STUDENT QUERY: {query}\n"

    base_prompt += """
If this query is relevant to CS/exam/placement, you MUST format your response using this structure:
1. **Concise Explanation**: A brief, easy-to-understand explanation of the topic.
2. **Important Exam Points**: 3-5 key bullet points that professors look for.
3. **5 Probable Questions**: 5 likely exam questions regarding this topic.
"""

    if mode.lower() == "placement":
        base_prompt += "4. **2 Interview Questions**: 2 tricky or conceptual interview questions asked by top tech companies.\n"

    return base_prompt

def generate_pure_llm_response(query: str, mode: str) -> str:
    """Generates a response using ONLY the LLM's internal knowledge."""
    start_time = time.time()
    try:
        llm = get_llm()
        prompt_text = build_prompt_text(query=query, mode=mode, context="")
        response = llm.invoke([HumanMessage(content=prompt_text)])
        
        # Log metrics
        response_time = time.time() - start_time
        num_tokens = estimate_tokens(prompt_text + response.content)
        log_query_metrics(
            query=query,
            mode=mode,
            subject="Pure LLM (No RAG)",
            response_time=response_time,
            similarity_scores=[],
            num_tokens=num_tokens
        )
        
        return response.content
    except Exception as e:
        response_time = time.time() - start_time
        log_query_metrics(
            query=query,
            mode=mode,
            subject="Pure LLM (No RAG)",
            response_time=response_time,
            similarity_scores=[],
            error=str(e)
        )
        raise

def generate_rag_response(subject: str, query: str, mode: str):
    """Generates a response using the vector DB for context."""
    start_time = time.time()
    error_msg = None
    
    try:
        # 1. Retrieve top 5 chunks
        with RequestLogger(f"Retrieval for query: {query[:30]}..."):
            retrieved_docs_with_scores = retrieve_top_k(subject, query, k=5)
        
        # Extract text from docs and scores
        context_text = ""
        sources = []
        similarity_scores = []
        
        for doc, score in retrieved_docs_with_scores:
            context_text += f"---\n{doc.page_content}\n"
            sources.append({"source": doc.metadata.get("source", "Unknown"), "score": float(score)})
            similarity_scores.append(float(score))
        
        # 2. Generate Prompt and call LLM
        llm = get_llm()
        prompt_text = build_prompt_text(query=query, mode=mode, context=context_text)
        response = llm.invoke([HumanMessage(content=prompt_text)])
        
        # Calculate metrics
        response_time = time.time() - start_time
        num_tokens = estimate_tokens(prompt_text + response.content)
        
        # Log metrics
        log_query_metrics(
            query=query,
            mode=mode,
            subject=subject,
            response_time=response_time,
            similarity_scores=similarity_scores,
            num_tokens=num_tokens
        )
        
        return {
            "answer": response.content,
            "sources": sources,
            "metrics": {
                "response_time": response_time,
                "num_tokens": num_tokens,
                "sources_retrieved": len(similarity_scores),
                "avg_similarity": sum(similarity_scores)/len(similarity_scores) if similarity_scores else 0
            }
        }
    except Exception as e:
        response_time = time.time() - start_time
        log_query_metrics(
            query=query,
            mode=mode,
            subject=subject,
            response_time=response_time,
            similarity_scores=[],
            error=str(e)
        )
        raise
