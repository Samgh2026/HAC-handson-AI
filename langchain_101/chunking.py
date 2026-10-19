'''
text splitting : Chunking 
Doc objects --> Text Splitter --> Chunks
'''
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# create the loader object
loader = PyPDFLoader("/Volumes/Work/HAC_Langchain_Demo/langchain_101/gpt-1paper.pdf")

# create the document object
docs = loader.load()

print(f"Number of documents loaded: {len(docs)}")
print("="*50) # ======================
print(f"First document object content: {docs[0].page_content}")
print("Starting the chunking process...\n")

# create the text splitter object
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=140
)

# create the chunks
chunks = text_splitter.split_documents(docs)

print(f"Number of chunks created: {len(chunks)}")
print("="*50) # ======================
print(f"First chunk content: {chunks[0].page_content}")