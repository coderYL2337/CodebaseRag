import streamlit as st
from rag_utils import perform_rag, extract_text_from_image
from PIL import Image

st.title("Codebase Chat")

st.markdown("### Ask questions about the following codebase(s) for better understanding.")

# Repository selection
col1, col2 = st.columns(2)
with col1:
    use_secure_agent = st.checkbox("SecureAgent")
    st.markdown("[View on GitHub](https://github.com/CoderAgent/SecureAgent)")
with col2:
    use_ai_chatbot = st.checkbox("AI-Chatbot")
    st.markdown("[View on GitHub](https://github.com/coderYL2337/ai-chatbot)")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "parsed_text_from_frontend" not in st.session_state:
    st.session_state.parsed_text_from_frontend = None
if "current_codebase" not in st.session_state:
    st.session_state.current_codebase = None
if "previous_uploaded_file" not in st.session_state:
    st.session_state.previous_uploaded_file = None

# Function to clear screenshot data
def clear_screenshot_data():
    st.session_state.uploaded_image = None
    st.session_state.parsed_text_from_frontend = None
    st.session_state.previous_uploaded_file = None
    print("[INFO] Screenshot data cleared.")

# Reset chat functionality
if st.button("Reset Chat"):
    st.session_state.messages = []
    clear_screenshot_data()
    st.success("Chat has been reset successfully!")

# Image upload
uploaded_file = st.file_uploader("Upload a code screenshot (optional)", type=["png", "jpg", "jpeg"])

# Handle image upload and removal
if uploaded_file is not None:
    # Check if this is a new upload
    if (st.session_state.previous_uploaded_file is None or 
        uploaded_file.name != st.session_state.previous_uploaded_file.name):
        st.session_state.uploaded_image = Image.open(uploaded_file)
        st.image(st.session_state.uploaded_image, caption="Uploaded code screenshot")
        st.session_state.parsed_text_from_frontend = extract_text_from_image(st.session_state.uploaded_image)
        st.session_state.previous_uploaded_file = uploaded_file
elif st.session_state.previous_uploaded_file is not None:
    # Image was removed
    clear_screenshot_data()

# Handle repository change - only trigger when there's an actual change
current_selection = (use_secure_agent, use_ai_chatbot)
if (st.session_state.current_codebase is not None and 
    st.session_state.current_codebase != current_selection and 
    (use_secure_agent or use_ai_chatbot)):  # Only if at least one repo is selected
    st.session_state.current_codebase = current_selection
    clear_screenshot_data()
    st.info("Codebase switched. Screenshot data cleared.")
elif st.session_state.current_codebase is None and (use_secure_agent or use_ai_chatbot):
    st.session_state.current_codebase = current_selection

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What does this codebase do?"):
    # Validate repository selection
    if not (use_secure_agent or use_ai_chatbot):
        st.error("Please select at least one codebase to search.")
        st.stop()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Perform RAG search and respond
    with st.chat_message("assistant"):
        with st.spinner("Searching codebase(s)..."):
            try:
                # Only pass screenshot data if it exists
                current_image = st.session_state.uploaded_image if uploaded_file is not None else None
                current_parsed_text = (st.session_state.parsed_text_from_frontend 
                                     if uploaded_file is not None else None)
                
                response = perform_rag(
                    prompt,
                    image=current_image,
                    parsed_text_from_frontend=current_parsed_text,
                    use_secure_agent=use_secure_agent,
                    use_ai_chatbot=use_ai_chatbot,
                )
                st.markdown(response)
                # Append the assistant's response to session state
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")





