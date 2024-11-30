import pinecone
from pinecone import Pinecone
from embeddings import get_huggingface_embeddings
import os
from dotenv import load_dotenv
from openai import Client
from PIL import Image
import pytesseract
import cv2
import numpy as np
import shutil

load_dotenv()
# Ensure Tesseract is available
if shutil.which("tesseract") is None:
    pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Get index configurations
INDICES_CONFIG = {
    "secure_agent": {
        "name": os.getenv("PINECONE_INDEX_SECURE"),  # 'codebase1-rag'
        "namespace": "https://github.com/CoderAgent/SecureAgent/functions",
    },
    "ai_chatbot": {
        "name": os.getenv("PINECONE_INDEX_CHATBOT"),  # 'codebase2-rag'
        "namespace": "https://github.com/coderYL2337/ai-chatbot",
    },
}

# Initialize Groq client
client = Client(api_key=os.getenv('GROQ_API_KEY'), base_url="https://api.groq.com/openai/v1")

def extract_text_from_image(image):
    """
    Extract text from an image using pytesseract.
    
    Args:
        image (PIL.Image.Image): The uploaded image object.
    
    Returns:
        str: Extracted text or None if extraction fails.
    """
    try:
        # Convert PIL Image to OpenCV format
        print("[INFO] Starting OCR processing...")
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Preprocess the image: Convert to grayscale
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        print("[DEBUG] Grayscale conversion completed.")

        # Apply thresholding to preprocess the image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print("[DEBUG] Thresholding applied.")
        # Remove noise using median blur
        denoised = cv2.medianBlur(thresh, 3)
        print("[DEBUG] Noise removal completed.")

        # OCR using pytesseract
        extracted_text = pytesseract.image_to_string(denoised)
        print(f"[DEBUG] Extracted text: {extracted_text}")

        return extracted_text.strip() if extracted_text else None

    except Exception as e:
        print(f"[ERROR] Error during text extraction with pytesseract: {e}")
        return None

def search_pinecone(query_embedding, indices=None):
    """Search specified Pinecone indices."""
    all_matches = []

    for index_info in indices:
        index = pc.Index(index_info["name"])
        try:
            top_matches = index.query(
                vector=query_embedding.tolist(),
                top_k=5,
                include_metadata=True,
                namespace=index_info["namespace"],
            )
            all_matches.extend([item["metadata"]["text"] for item in top_matches["matches"]])
        except Exception as e:
            print(f"[ERROR] Error querying Pinecone index {index_info['name']}: {e}")
            continue

    return all_matches

def perform_rag(query, image=None, parsed_text_from_frontend=None, use_secure_agent=False, use_ai_chatbot=False):
    """Perform RAG search across selected repositories."""
    # Validate repository selection
    if not (use_secure_agent or use_ai_chatbot):
        return "Error: Please select at least one codebase to search before submitting your query."

    screenshot_code = None
    screenshot_embedding = None
    has_screenshot_context = False

    # Process screenshot if provided and valid
    if image and parsed_text_from_frontend:
        try:
            print("[INFO] Processing uploaded image...")
            screenshot_code = parsed_text_from_frontend
            if screenshot_code and len(screenshot_code.strip()) > 0:
                screenshot_embedding = get_huggingface_embeddings(screenshot_code)
                has_screenshot_context = True
                print(f"[DEBUG] Screenshot Embedding generated successfully")
        except Exception as e:
            print(f"[ERROR] Error processing image: {e}")
            screenshot_code = None
            screenshot_embedding = None

    # Embed user query
    query_embedding = get_huggingface_embeddings(query)
    print(f"[DEBUG] Query embedding generated")

    # Combine embeddings only if screenshot is present and valid
    combined_embedding = query_embedding
    if has_screenshot_context and screenshot_embedding is not None:
        # Adjust weights based on query content
        query_weight = 0.8  # Increased weight for query
        screenshot_weight = 0.2  # Decreased weight for screenshot
        combined_embedding = (query_weight * query_embedding) + (screenshot_weight * screenshot_embedding)
        print(f"[DEBUG] Combined embedding with weights: query={query_weight}, screenshot={screenshot_weight}")

    # Build list of indices to search
    indices_to_search = []
    if use_secure_agent:
        indices_to_search.append(INDICES_CONFIG["secure_agent"])
    if use_ai_chatbot:
        indices_to_search.append(INDICES_CONFIG["ai_chatbot"])

    # Get contexts from all selected indices
    all_contexts = search_pinecone(combined_embedding, indices=indices_to_search)
    print(f"[DEBUG] Retrieved {len(all_contexts)} contexts")

    # Construct context text with the most relevant matches
    context_text = "\n\n-------\n\n".join(all_contexts[:5])  # Reduced from 10 to 5 for more focused context

    # Create augmented query based on presence of screenshot
    if has_screenshot_context:
        augmented_query = f"""<CONTEXT>\n{context_text}\n-------\n</CONTEXT>

Code snippet from screenshot:
{screenshot_code}

User query: {query}

Please address the user's query, giving priority to the specific code snippet if the query is directly related to it. Otherwise, focus on the general context and query."""
    else:
        augmented_query = f"""<CONTEXT>\n{context_text}\n-------\n</CONTEXT>

User query: {query}

Please provide a detailed response to the user's query based on the provided context."""

    print(f"[DEBUG] Query augmented with appropriate context")

    # Optimized system prompt
    system_prompt = """You are an expert Software Engineer with over 20 years of experience in TypeScript and Python.
    When analyzing code or responding to queries:
    1. If a code snippet is provided and the query is related to it, focus primarily on explaining that code
    2. If the query is not directly related to the provided code snippet, or no snippet is provided, focus on the context and query
    3. Use the provided context to enhance your explanations
    4. Be concise but thorough in your explanations"""

    # Get LLM response using Groq
    try:
        llm_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )
        return llm_response.choices[0].message.content

    except Exception as e:
        print(f"[ERROR] Error generating LLM response: {e}")
        return "An error occurred while generating the response. Please try again."