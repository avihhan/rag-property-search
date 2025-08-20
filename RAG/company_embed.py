import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()   

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

def create_company_description(company):
    """
    Create a comprehensive text description for each company
    """
    basic_info = company['basic_info']
    deal_analysis = company['deal_analysis']
    
    description = f"""
    Company: {company['company_name']}
    Industry: {basic_info['industry']}
    Headquarters: {basic_info['headquarters']}
    Revenue: {basic_info['revenue']}
    Employees: {basic_info['employees']}
    Business Model: {deal_analysis['business_model']}
    Strategic Priorities: {', '.join(deal_analysis['strategic_priorities'])}
    Ideal Operating Partner Profile:
    - Industry: {deal_analysis['ideal_op_profile']['industry']}
    - Functional Strengths: {', '.join(deal_analysis['ideal_op_profile']['functional'])}
    - Leadership Qualities: {', '.join(deal_analysis['ideal_op_profile']['leadership'])}
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

def create_pinecone_index(index_name='company-information-dummy'):
    """
    Create a Pinecone index for company information
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

def process_company_data(json_file='dummy_companies.json', index_name: str = 'company-information-dummy'):
    """
    Main function to process JSON file and create embeddings
    """
    print("="*60)
    print("PROCESSING COMPANY DATA")
    print("="*60)
    
    # Load the JSON file
    print("Loading company data...")
    with open(json_file, 'r') as f:
        companies = json.load(f)
    print(f"✅ Loaded {len(companies)} companies from {json_file}")
    
    # Create company descriptions
    print("Creating company descriptions...")
    for company in companies:
        company['description'] = create_company_description(company)
    
    # Create Pinecone index
    index = create_pinecone_index(index_name=index_name)
    
    # Generate embeddings and prepare data for upsert
    print("Generating embeddings and preparing data...")
    vectors = []
    
    for idx, company in enumerate(companies):
        print(f"Processing company {idx + 1}/{len(companies)}: {company['company_name']}")
        
        # Generate embedding
        embedding = generate_embedding(company['description'])
        
        if embedding:
            # Prepare vector data with basic_info as metadata for filtering
            vector_data = {
                'id': str(idx),
                'values': embedding,
                'metadata': {
                    'company_name': company['company_name'],
                    'industry': company['basic_info']['industry'],
                    'headquarters': company['basic_info']['headquarters'],
                    'revenue': company['basic_info']['revenue'],
                    'employees': company['basic_info']['employees'],
                    'business_model': company['deal_analysis']['business_model'],
                    'strategic_priorities': company['deal_analysis']['strategic_priorities'],
                    'ideal_op_industry': company['deal_analysis']['ideal_op_profile']['industry'],
                    'ideal_op_functional': company['deal_analysis']['ideal_op_profile']['functional'],
                    'ideal_op_leadership': company['deal_analysis']['ideal_op_profile']['leadership'],
                    'description': company['description']
                }
            }
            vectors.append(vector_data)
            print(f"  ✅ Generated embedding for {company['company_name']}")
        else:
            print(f"  ❌ Failed to generate embedding for {company['company_name']}")
        
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

def process_company_data_from_records(companies: List[Dict[str, Any]], index_name: str = 'company-information-dummy') -> Dict[str, Any]:
    """
    Ingest a list of company records into Pinecone.

    Each record should include keys compatible with create_company_description:
    {
      'company_name': str,
      'basic_info': {'industry': str, 'headquarters': str, 'revenue': str, 'employees': int},
      'deal_analysis': {
          'business_model': str,
          'strategic_priorities': List[str],
          'ideal_op_profile': {
              'industry': str,
              'functional': List[str],
              'leadership': List[str]
          }
      }
    }
    """
    print("="*60)
    print("INGESTING COMPANY RECORDS")
    print("="*60)

    # Build descriptions if missing
    for company in companies:
        if 'description' not in company or not company['description']:
            company['description'] = create_company_description(company)

    index = create_pinecone_index(index_name=index_name)

    print("Generating embeddings and preparing data...")
    vectors: List[Dict[str, Any]] = []

    for idx, company in enumerate(companies):
        print(f"Processing company {idx + 1}/{len(companies)}: {company.get('company_name', 'UNKNOWN')}")

        embedding = generate_embedding(company['description'])
        if embedding:
            vector_data = {
                'id': str(idx),
                'values': embedding,
                'metadata': {
                    'company_name': company['company_name'],
                    'industry': company['basic_info']['industry'],
                    'headquarters': company['basic_info']['headquarters'],
                    'revenue': company['basic_info']['revenue'],
                    'employees': company['basic_info']['employees'],
                    'business_model': company['deal_analysis']['business_model'],
                    'strategic_priorities': company['deal_analysis']['strategic_priorities'],
                    'ideal_op_industry': company['deal_analysis']['ideal_op_profile']['industry'],
                    'ideal_op_functional': company['deal_analysis']['ideal_op_profile']['functional'],
                    'ideal_op_leadership': company['deal_analysis']['ideal_op_profile']['leadership'],
                    'description': company['description']
                }
            }
            vectors.append(vector_data)
            print(f"  ✅ Generated embedding for {company['company_name']}")
        else:
            print(f"  ❌ Failed to generate embedding for {company.get('company_name', 'UNKNOWN')}")
        time.sleep(0.15)

    print(f"\nUpserting {len(vectors)} vectors into Pinecone (index: {index_name})...")
    try:
        index.upsert(vectors=vectors)
        print(f"✅ Successfully stored {len(vectors)} vectors in Pinecone!")
        return {
            'index_name': index_name,
            'upserted_count': len(vectors)
        }
    except Exception as e:
        print(f"❌ Error upserting vectors: {e}")
        return {
            'index_name': index_name,
            'error': str(e)
        }

def parse_companies_from_csv(file_like) -> List[Dict[str, Any]]:
    """
    Parse a CSV file-like object into the expected companies list structure.

    Expected columns:
      company_name, industry, headquarters, revenue, employees,
      business_model, strategic_priorities, ideal_op_industry,
      ideal_op_functional, ideal_op_leadership
    """
    df = pd.read_csv(file_like)

    required = [
        'company_name', 'industry', 'headquarters', 'revenue', 'employees',
        'business_model', 'strategic_priorities', 'ideal_op_industry',
        'ideal_op_functional', 'ideal_op_leadership'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    def to_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return value
        if pd.isna(value):
            return []
        return [s.strip() for s in str(value).split(',') if s.strip()]

    companies: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        companies.append({
            'company_name': row['company_name'],
            'basic_info': {
                'industry': row['industry'],
                'headquarters': row['headquarters'],
                'revenue': row['revenue'],
                'employees': int(row['employees']) if not pd.isna(row['employees']) else 0,
            },
            'deal_analysis': {
                'business_model': row['business_model'],
                'strategic_priorities': to_list(row['strategic_priorities']),
                'ideal_op_profile': {
                    'industry': row['ideal_op_industry'],
                    'functional': to_list(row['ideal_op_functional']),
                    'leadership': to_list(row['ideal_op_leadership'])
                }
            }
        })

    return companies

def get_index_details(index_name: str = 'company-information-dummy', sample_limit: int = 1) -> Dict[str, Any]:
    """
    Return details about the index including counts and sample structure when possible.

    Attempts to include:
      - total_vector_count, namespaces, dimension, index_fullness
      - creation_time / last_updated_time if available from describe_index
      - a sample record's metadata when possible
    """
    details: Dict[str, Any] = {'index_name': index_name}
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        details['stats'] = {
            'total_vector_count': stats.get('total_vector_count', 0),
            'namespaces': stats.get('namespaces', {}),
            'dimension': stats.get('dimension', 'Unknown'),
            'index_fullness': stats.get('index_fullness', 'Unknown')
        }
    except Exception as e:
        details['stats_error'] = str(e)

    # Index description (may include timestamps depending on Pinecone version)
    try:
        desc = pc.describe_index(index_name)
        # Try best-effort serialization
        desc_dict = getattr(desc, 'to_dict', lambda: desc)()
        details['info'] = desc_dict
    except Exception as e:
        details['info_error'] = str(e)

    # Try to fetch a sample vector's metadata
    try:
        sample_meta: Optional[Dict[str, Any]] = None
        # Some Pinecone clients support listing IDs
        ids = []
        try:
            ids = index.list(prefix=None, limit=max(1, sample_limit))  # type: ignore[attr-defined]
        except Exception:
            ids = []

        if ids:
            fetched = index.fetch(ids=ids if isinstance(ids, list) else list(ids))
            vectors = fetched.get('vectors', {}) if isinstance(fetched, dict) else {}
            if vectors:
                first_id = next(iter(vectors.keys()))
                vec = vectors[first_id]
                sample_meta = vec.get('metadata') if isinstance(vec, dict) else None

        if not sample_meta:
            # Fallback to schema we ingest if we cannot fetch
            sample_meta = {
                'company_name': 'string',
                'industry': 'string',
                'headquarters': 'string',
                'revenue': 'string',
                'employees': 'int',
                'business_model': 'string',
                'strategic_priorities': ['string'],
                'ideal_op_industry': 'string',
                'ideal_op_functional': ['string'],
                'ideal_op_leadership': ['string'],
                'description': 'string'
            }
        details['sample_structure'] = sample_meta
    except Exception as e:
        details['sample_error'] = str(e)

    return details

def clear_index(index_name: str, delete_index: bool = False) -> Dict[str, Any]:
    """
    Delete all vectors from the index. If delete_index is True, delete the index entirely.
    """
    try:
        if delete_index:
            pc.delete_index(index_name)
            return {'index_name': index_name, 'deleted_index': True}
        else:
            index = pc.Index(index_name)
            index.delete(delete_all=True)
            return {'index_name': index_name, 'deleted_index': False, 'deleted_all_vectors': True}
    except Exception as e:
        return {'index_name': index_name, 'error': str(e)}

def check_index_statistics():
    """
    Check index statistics to verify data ingestion
    """
    try:
        index = pc.Index('company-information-dummy')
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
    # Process company data and store in Pinecone
    index = process_company_data()
    
    if index:
        # Check index statistics
        check_index_statistics()
        print("\n✅ Company data successfully converted to embeddings and stored in Pinecone!")
    else:
        print("\n❌ Failed to process company data")
