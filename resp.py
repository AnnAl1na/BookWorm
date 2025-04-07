import pickle
import os
import base64
from base64 import b64decode
from io import BytesIO
from PIL import Image
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_together import ChatTogether
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage

os.environ["TOGETHER_API_KEY"] = "d33d08efd5ddb95de1277b4e69bd0ba7007b4528a023876cd2ca9e588bcf9e27"

def retrieve_from_store(file_name):
    """Loads docstore & vectorstore for retrieval."""
    vectorstore_path = f"chroma/{file_name}"
    docstore_path = f"docstore/{file_name}.pkl"
    
    if not os.path.exists(docstore_path):
        raise FileNotFoundError(f"Docstore file {docstore_path} not found.")

    vectorstore = Chroma(persist_directory=vectorstore_path, collection_name=file_name, embedding_function=FastEmbedEmbeddings())
    
    with open(docstore_path, "rb") as f:
        store = pickle.load(f)

    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
    return retriever

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = "".join(str(text_element) for text_element in docs_by_type["texts"])
    
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if docs_by_type["images"]:
        image = docs_by_type["images"][0]
        prompt_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
        )

    return ChatPromptTemplate.from_messages([
        HumanMessage(content=prompt_content),
    ])

def query_document(file_name, question):
    chain = (
        {
            "context": retrieve_from_store(file_name) | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough().assign(
            response=(
                RunnableLambda(build_prompt)
                | ChatTogether(model="meta-llama/Llama-Vision-Free")
                | StrOutputParser()
            )
        )
    )
    
    # response = chain.invoke(question)
    # print("Response:", response)
    response = chain.invoke(
        question
    )
    print("Response:", response['response'])

    images=[]
    for image in response['context']['images']:
        print(len(response['context']['images']))
        pimage = display_base64_image(image)
        images.append(pimage)
        
    return response['response'],images
    
def display_base64_image(base64_code):
    # Decode the base64 string to binary
    image_data = base64.b64decode(base64_code)
    # Display the image
    image = Image.open(BytesIO(image_data))    # Convert to image
    print(type(image))
    image.show() 
    return image


# Example usage
response,images = query_document("10social1", "What is french revolution .give an image?")
