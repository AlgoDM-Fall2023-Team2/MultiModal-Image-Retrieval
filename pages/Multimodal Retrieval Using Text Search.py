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


st.title('MultiModal Retrieveal Using Text Search')
text_input = st.text_input("Enter text here")

n = st.number_input('Choose the number of similar images to be displayed',min_value=1,max_value=10,value=3)
button_clicked = st.button('Submit', key=1002)
   
if button_clicked and text_input is not None and n is not None:
    
    st.header("The Images as per your description are:")
    
    
    # Load the Qdrant vector store and index
    client = qdrant_client.QdrantClient(path="qdrant_mm_db")
    text_store = QdrantVectorStore(client=client,collection_name="text_collection")
    image_store = QdrantVectorStore(client=client,collection_name="image_collection")
   
    

    storage_context = StorageContext.from_defaults(vector_store=text_store)

# Create the MultiModal index
    documents = SimpleDirectoryReader("./mixed_wiki/").load_data()
    index = MultiModalVectorStoreIndex.from_documents(
    documents, storage_context=storage_context, image_vector_store=image_store
    )
    openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=1500
)
   
   
    MAX_TOKENS = 50
    retriever_engine = index.as_retriever(
                similarity_top_k=n, image_similarity_top_k=n
            )
            # retrieve more information from the GPT4V response
    
  
    retrieval_results = retriever_engine.retrieve(text_input)

   
    st.write()
    retrieved_image = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
                    retrieved_image.append(res_node.node.metadata["file_path"])
        else:
                    display_source_node(res_node, source_length=200)

    

    
    for i in retrieved_image:
                # Load image from file
        image = Image.open(i)

            # Display image
        st.image(image, use_column_width=True)###
    import shutil
    folder_path = "qdrant_mm_db"
    shutil.rmtree(folder_path)
    #shutil.rmtree("uploaded_images")
                    
                