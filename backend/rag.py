import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import warnings
from langchain.schema import Document

warnings.filterwarnings("ignore")


load_dotenv()


#load all the apis keys here
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
langchain_api_key = os.getenv("langchain_api_key")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


print("Keys are loaded")
 

#set the enviroment
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["langchain_api_key"] = langchain_api_key
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"


print("Enviroment is set")


Gemini = ChatGoogleGenerativeAI(
    model="gemini-vision",
    api_key=GOOGLE_API_KEY,
)

print("Gemini is set")

Groq = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama3-8b-8192",
)

print("Groq is set")

#load the pdf /scan pdf etc
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter

def load_document(doc_path):
    """
    Load a document and convert it to markdown format with images
    
    Args:
        doc_path (str): Path to the document file
        
    Returns:
        str: Markdown text with embedded images
    """
    try:
        # Get the MD text
        md_text_images = pymupdf4llm.to_markdown(
            doc=doc_path,
            page_chunks=True, 
            write_images=True,
            image_path="images",
            image_format="png",
            dpi=300
        )
        return md_text_images
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        return None

result = load_document(r"C:\Users\HP\Documents\Attention_is_all_you_need.pdf")
print("Text is loaded")  # Debugging step to check actual structure


texts = [] 
if isinstance(result, list) and len(result) > 0:  
    for item in result:  
        if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):  
            texts.append(item["text"])  


#Embeddings using langchain
print("Embeddings are loading")
from langchain_huggingface import HuggingFaceEmbeddings
def embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = {"device": "cpu"}
    )
    return embeddings

embedding = embeddings()
print(embeddings)
print("Till here embeddings are set")


#store the data into the vector store
print("Storing the data into the vector store")

from langchain.vectorstores import Chroma
def initialize_faiss_index(embeddings, texts):
    try:
        if texts:
            vectorstore = Chroma.from_texts(texts, embeddings)
            return vectorstore
        else:
            print("No texts provided to create FAISS index")
            return None
    except Exception as e:
        print(f"Error creating FAISS index: {str(e)}")

database = Chroma.from_texts(texts, embedding)
print("Database is created")

retriver = database.as_retriever(search_kwargs={"k": 2})

#define the retriveQa chain
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(
    llm=Groq,
    retriever=retriver,
    chain_type="stuff",
)

res = chain.invoke("what is the main idea of the document?")
print()
print()
print()
print()
print()
print("Resullt from the chain")
print(res)

#extracted the data from the youtube url
def extract_youtube_transcript(youtube_url):
    try:
        # Get video ID from URL
        from pytube import YouTube
        import whisper
        
        # Download audio from YouTube
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_file = audio_stream.download(filename="temp_audio")
        
        # Load Whisper model and transcribe
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        transcript_text = result["text"]
        
        # Clean up temporary audio file
        import os
        os.remove("audio_file")
        
        return transcript_text
        
    except Exception as e:
        print(f"Error extracting YouTube transcript: {str(e)}")
        return None
