import pickle
import time
import uuid
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import config file
import config

# Load API keys
config.load_keys()


    

def load_chunks(file_name):
    """Loads chunk data from a pickle file."""
    file_path = f"chunks/{file_name}.pkl"
    with open(file_path, "rb") as f:
        return pickle.load(f)

def extract_texts_tables_images(chunks):
    """Extracts texts, tables, and images from chunks."""
    texts, tables, images_b64 = [], [], []
    
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
            for el in chunk.metadata.orig_elements:
                if "Table" in str(type(el)):
                    tables.append(el.metadata.text_as_html)
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    
    return texts, tables, images_b64

def summarize_texts(texts,tables):
    """Summarizes extracted texts using ChatGroq."""
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.
    Respond only with the summary, no additional comments.
    Table or text chunk: {element}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    
    text_summaries=[]
    for i in range(0, len(texts), 5):
            batch = texts[i : i + 5]
            summaries = summarize_chain.batch(batch, {"max_concurrency": 5})
            text_summaries.extend(summaries)

            if i + 5 < len(texts):  # Wait only if there are more batches left
                print(f"⏳ Waiting {90} seconds before sending next batch...")
                time.sleep(90)  # Respect the rate limit
    
    return text_summaries

def store_in_chroma_docstore(file_name, texts, text_summaries):
    """Creates and stores vectorstore and docstore with a given filename."""
    vectorstore_path = f"chroma/{file_name}"
    docstore_path = f"docstore/{file_name}.pkl"
    
    vectorstore = Chroma(persist_directory=vectorstore_path, collection_name=file_name, embedding_function=FastEmbedEmbeddings())
    store = InMemoryStore()
    
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [Document(page_content=summary, metadata={"doc_id": doc_ids[i]}) for i, summary in enumerate(text_summaries)]
    
    vectorstore.add_documents(summary_texts)
    store.mset(list(zip(doc_ids, texts)))
    
    with open(docstore_path, "wb") as f:
        pickle.dump(store, f)
    
    print(f"✅ Stored: {file_name} -> ChromaDB: {vectorstore_path}, Docstore: {docstore_path}")

def process_and_store(file_name):
    """Main function to process a pickle file and store vectorstore & docstore."""
    chunks = load_chunks(file_name)
    texts, tables, images_b64 = extract_texts_tables_images(chunks)
    text_summaries = summarize_texts(texts,tables)
    store_in_chroma_docstore(file_name, texts, text_summaries)

def retrieve_from_store(file_name):
    """Loads docstore & vectorstore for retrieval."""
    vectorstore_path = f"chroma/{file_name}"
    docstore_path = f"docstore/{file_name}.pkl"
    
    vectorstore = Chroma(persist_directory=vectorstore_path, collection_name=file_name, embedding_function=FastEmbedEmbeddings())
    
    with open(docstore_path, "rb") as f:
        store = pickle.load(f)
    
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
    return retriever

process_and_store("10social1")