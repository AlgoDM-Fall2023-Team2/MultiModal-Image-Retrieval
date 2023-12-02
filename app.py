import os
from PIL import Image
import streamlit as st 
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index import SimpleDirectoryReader
from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.vector_stores import QdrantVectorStore
from llama_index import SimpleDirectoryReader, StorageContext
from llama_index.response.notebook_utils import display_source_node
from llama_index.schema import ImageNode
import matplotlib.pyplot as plt
import qdrant_client
#from Home import *
from llama_index import StorageContext, load_index_from_storage
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index import SimpleDirectoryReader



OPENAI_API_TOKEN = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN

# load DB


st.title('MultiModal Image Retrieveal')

image = st.file_uploader("Upload Image", type = ['jpg'])
n = st.number_input('Choose the number of similar images to be displayed',min_value=1,max_value=10,value=3)
button_clicked = st.button('Submit', key=1002)
   
if button_clicked and image is not None and n is not None:
    # Create a folder to save uploaded images
    upload_folder = "./uploaded_images"
    os.makedirs(upload_folder, exist_ok=True)
    # Save the uploaded image to the specified folder
    image_path = os.path.join(upload_folder, image.name)
    with open(image_path, "wb") as f:
        f.write(image.read())
        #pil_image = Image.open(image)
    st.header("The Image you uploaded")
    
    st.image(image=image)
    
    # Load the Qdrant vector store and index
    client = qdrant_client.QdrantClient(path="qdrant_mm_db")
    text_store = QdrantVectorStore(client=client,collection_name="text_collection")
    image_store = QdrantVectorStore(client=client,collection_name="image_collection")
   

    # Save it
    # index.storage_context.persist(persist_dir="./storage")

    # # Load it
    

    storage_context = StorageContext.from_defaults(
         vector_store=text_store, persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context, image_store=image_store)
    
    
    index = load_index_from_storage(storage_context, image_store=image_store)
    openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=1500
)
    # put your local directore here
    image_documents = SimpleDirectoryReader("./uploaded_images").load_data()
    response_1 = openai_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,)
   
    MAX_TOKENS = 50
    retriever_engine = index.as_retriever(
                similarity_top_k=n, image_similarity_top_k=n
            )
            # retrieve more information from the GPT4V response
    retrieval_results = retriever_engine.retrieve(response_1.text[:MAX_TOKENS])

   
    st.write()
    retrieved_image = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
                    retrieved_image.append(res_node.node.metadata["file_path"])
        else:
                    display_source_node(res_node, source_length=200)

    st.write(retrieved_image)

    
    for i in retrieved_image:
                # Load image from file
        image = Image.open(i)

            # Display image
        st.image(image, use_column_width=True)###
   
                    
                