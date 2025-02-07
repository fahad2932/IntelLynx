from pinecone import Pinecone, ServerlessSpec
import time

def initialize_pinecone_index(pinecone_api_key, index_name="test-databasee"):
    # Initialize Pinecone 
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists
    existing_indexes = pc.list_indexes()
       
    if index_name not in existing_indexes:
        try:
            pc.create_index(
                name=index_name,
                dimension=384,  
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                ),
                deletion_protection='enabled'  # Enable delete protection
            )
            time.sleep(20)
            
            index = pc.Index(index_name)
            print(f"Successfully created and connected to index: {index_name}")
            return index
            
        except Exception as e:
            print(f"Error creating/connecting to index: {str(e)}")
            return None
    
    # If index already exists, just connect to it
    return pc.Index(index_name)