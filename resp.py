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
    
    # Extract grade from filename (assumes it's passed via context)
    grade = str(kwargs.get("file_name", ""))[:2]
    # if not grade.isdigit():
    #     grade = "10"  # fallback if missing/invalid
    
    context_text = "".join(str(text_element) for text_element in docs_by_type["texts"])
    
    prompt_template = f"""
    You are an AI tutor helping a Grade {grade} student understand their school subject.

    Based on the following context, which may include textbook passages, tables, and relevant images,
    generate a clear, simple, and engaging answer to the student's question.
    If an image is given, analyze it carefully and relate it only if relevant.

    Context: {context_text}

    Question: {user_question}

    Your response should:
    - Be tailored for a Grade {grade} student
    - Use simple, grade-appropriate language
    - Avoid complex jargon

    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if docs_by_type["images"]:
        image = docs_by_type["images"][0]
        prompt_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        })

    return ChatPromptTemplate.from_messages([
        HumanMessage(content=prompt_content),
    ])


# def build_prompt(kwargs):
#     docs_by_type = kwargs["context"]
#     user_question = kwargs["question"]

#     context_text = "".join(str(text_element) for text_element in docs_by_type["texts"])
    
#     prompt_template = f"""
#     Generate and answer for the student query based only on the following context, which can include text, tables, and the image.
#     (images might or might be related to the question.. see to it)
#     provided context from textbooks.
#     Context: {context_text}
#     Question: {user_question}
#     """

#     prompt_content = [{"type": "text", "text": prompt_template}]

#     if docs_by_type["images"]:
#         image = docs_by_type["images"][0]
#         prompt_content.append(
#             {
#                 "type": "image_url",
#                 "image_url": {"url": f"data:image/jpeg;base64,{image}"},
#             }
#         )

#     return ChatPromptTemplate.from_messages([
#         HumanMessage(content=prompt_content),
#     ])

# def query_document(file_name, question):
#     chain = (
#         {
#             "context": retrieve_from_store(file_name) | RunnableLambda(parse_docs),
#             "question": RunnablePassthrough(),
#         }
#         | RunnablePassthrough().assign(
#             response=(
#                 RunnableLambda(build_prompt)
#                 | ChatTogether(model="meta-llama/Llama-Vision-Free")
#                 | StrOutputParser()
#             )
#         )
#     )
    
#     # response = chain.invoke(question)
#     # print("Response:", response)
#     response = chain.invoke(
#         question
#     )
#     print("Response:", response['response'])

#     images=[]
#     for image in response['context']['images']:
#         print(len(response['context']['images']))
#         pimage = display_base64_image(image)
#         images.append(pimage)
        
#     return response['response'],images

def query_document(file_name, question):
    chain = (
        {
            "context": retrieve_from_store(file_name) | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
            "file_name": RunnableLambda(lambda _: file_name),
        }
        | RunnablePassthrough().assign(
            response=(
                RunnableLambda(build_prompt)
                | ChatTogether(model="meta-llama/Llama-Vision-Free")
                | StrOutputParser()
            )
        )
    )
    
    response = chain.invoke(question)
    print("Response:", response['response'])

    images = []
    for image in response['context']['images']:
        print(len(response['context']['images']))
        pimage = display_base64_image(image)
        images.append(pimage)

    return response['response'], images

    
def display_base64_image(base64_code):
    # Decode the base64 string to binary
    image_data = base64.b64decode(base64_code)
    # Display the image
    image = Image.open(BytesIO(image_data))    # Convert to image
    print(type(image))
    image.show() 
    return image


def query_image(image_base64_url: str, question: str):
    def build_image_prompt(_):
        prompt_content = [
            {
                "type": "text",
                "text": f"""
You are an AI tutor. A student has a question based on the image provided.

Answer the question clearly and concisely, using simple, age-appropriate language.

Question: {question}
"""
            },
            {
                "type": "image_url",
                # "image_url": {"url": image_base64_url},
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64_url}"},
            }
        ]
        return ChatPromptTemplate.from_messages([
            HumanMessage(content=prompt_content)
        ])

    chain = (
        RunnableLambda(lambda _: {})
        | RunnableLambda(build_image_prompt)
        | ChatTogether(model="meta-llama/Llama-Vision-Free")
        | StrOutputParser()
    )

    return chain.invoke({})



# Example usage
response,images = query_document("5Science1", '''
what are animals?''')