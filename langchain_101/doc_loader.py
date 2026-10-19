'''
Document loader :
1. helps us work with the document types: pdf, docx, txt, csv, etc.
2. creates document objects aka chunks

'''
from langchain_community.document_loaders import PyPDFLoader

# create the loader object
loader = PyPDFLoader("/Volumes/Work/HAC_Langchain_Demo/langchain_101/gpt-1paper.pdf")

# create the document object
docs = loader.load()

print(f"Number of documents loaded: {len(docs)}")
print("="*50) # ======================
print(docs)
print("="*50)
print(f"First document object content: {docs[0].page_content}")
# the above is the page content on first page of the pdf


