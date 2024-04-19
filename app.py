from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from langchain_community.llms.together import Together
from langchain.chains.question_answering import load_qa_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
      
        # create embeddings
        model_name = "all-MiniLM-L6-v2"
        embeddings_model = SentenceTransformer(model_name)
        embeddings = embeddings_model.encode(chunks)  # Generate embeddings
        
        # create FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
      
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            # Encode user question
            user_question_embedding = embeddings_model.encode([user_question])[0]
            
            # Perform similarity search
            k = 5  # Number of similar documents to retrieve
            distances, indices = index.search(np.array([user_question_embedding]), k)
            
            # Retrieve relevant documents
            similar_docs = [chunks[i] for i in indices[0]]
            
            # Generate response using Together model
            llm = Together(model="meta-llama/Llama-2-7b-chat-hf", together_api_key="7b011acae38eaa5a2df735f5969d3df31762c4195f6a9043db48dfdd37beb5e4", max_tokens=1024)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=similar_docs, question=user_question)
            
            # Display response
            st.write("Response:")
            st.write(response)

if __name__ == '__main__':
    main()
