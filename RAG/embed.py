import os
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

def create_property_description(row):
    """
    Create a comprehensive text description for each property
    """
    # Clean location data to remove quotes
    location = row['Location'].strip('"') if isinstance(row['Location'], str) else str(row['Location'])
    
    description = f"""
    Property: {row['Property Name']}
    Type: {row['Property Name'].split()[0]}
    Bedrooms: {row['Bedrooms']}
    View: {row['View']}
    Location: {location}
    Size: {row['Size (sqft)']} square feet
    Price: ${row['Price (USD)']:,}
    Description: {row['Description']}
    """
    return description.strip()

def generate_embedding(text):
    """
    Generate embedding using OpenAI's text-embedding-3-large model
    """
    try:
        response = client.embeddings.create(
            input=text,
            model='text-embedding-3-large'
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def create_pinecone_index(index_name='property-listings'):
    """
    Create a Pinecone index for property listings
    """
    print(f"Creating Pinecone index: {index_name}")
    
    # Check if the index already exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,  # Dimension for text-embedding-3-large
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # For free plan compatibility
            )
        )
        print(f"✅ Created index: {index_name}")
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        time.sleep(30)
    else:
        print(f"✅ Index {index_name} already exists")
    
    # Connect to the index
    index = pc.Index(index_name)
    return index

def process_property_listings(csv_file='property_listings_dummy.csv'):
    """
    Main function to process CSV file and create embeddings
    """
    print("="*60)
    print("PROCESSING PROPERTY LISTINGS")
    print("="*60)
    
    # Load the CSV file
    print("Loading property data...")
    df = pd.read_csv(csv_file, quotechar='"')
    print(f"✅ Loaded {len(df)} properties from {csv_file}")
    
    # Create property descriptions
    print("Creating property descriptions...")
    df['description'] = df.apply(create_property_description, axis=1)
    
    # Create Pinecone index
    index = create_pinecone_index()
    
    # Generate embeddings and prepare data for upsert
    print("Generating embeddings and preparing data...")
    vectors = []
    
    for idx, row in df.iterrows():
        print(f"Processing property {idx + 1}/{len(df)}: {row['Property Name']}")
        
        # Generate embedding
        embedding = generate_embedding(row['description'])
        
        if embedding:
            # Clean location data
            location = row['Location'].strip('"') if isinstance(row['Location'], str) else str(row['Location'])
            
            # Prepare vector data
            vector_data = {
                'id': str(idx),
                'values': embedding,
                'metadata': {
                    'property_name': row['Property Name'],
                    'bedrooms': int(row['Bedrooms']),
                    'view': row['View'],
                    'location': location,
                    'size_sqft': int(row['Size (sqft)']),
                    'price_usd': int(row['Price (USD)']),
                    'description': row['Description'],
                    'property_type': row['Property Name'].split()[0]
                }
            }
            vectors.append(vector_data)
            print(f"  ✅ Generated embedding for {row['Property Name']}")
        else:
            print(f"  ❌ Failed to generate embedding for {row['Property Name']}")
        
        # Add small delay to avoid rate limits
        time.sleep(0.15)
    
    # Upsert vectors into Pinecone
    print(f"\nUpserting {len(vectors)} vectors into Pinecone...")
    try:
        index.upsert(vectors=vectors)
        print(f"✅ Successfully stored {len(vectors)} vectors in Pinecone!")
    except Exception as e:
        print(f"❌ Error upserting vectors: {e}")
        return None
    
    return index

def check_index_statistics():
    """
    Check index statistics to verify data ingestion
    """
    try:
        index = pc.Index('property-listings')
        stats = index.describe_index_stats()
        
        print(f"\nIndex Statistics:")
        print(f"  Total Vector Count: {stats.get('total_vector_count', 0)}")
        print(f"  Namespaces: {stats.get('namespaces', {})}")
        print(f"  Dimension: {stats.get('dimension', 'Unknown')}")
        print(f"  Index Fullness: {stats.get('index_fullness', 'Unknown')}")
        
        return stats
    except Exception as e:
        print(f"❌ Error getting index statistics: {e}")
        return None

if __name__ == "__main__":
    # Process property listings and store in Pinecone
    index = process_property_listings()
    
    if index:
        # Check index statistics
        check_index_statistics()
        print("\n✅ Property listings successfully converted to embeddings and stored in Pinecone!")
    else:
        print("\n❌ Failed to process property listings")
