import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

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

def get_top_k_companies(query, top_k=5, industry_filter=None, location_filter=None, revenue_filter=None, employees_filter=None, with_reasoning=False, index_name='company-information-dummy'):
    """
    Get top k companies for a query and return structured results
    
    Args:
        query (str): Search query text
        top_k (int): Number of results to return
        industry_filter (dict): Optional industry filter, e.g., {"$in": ["EdTech", "SaaS Data Analytics"]}
        location_filter (dict): Optional location filter, e.g., {"$in": ["California", "CA"]}
        revenue_filter (dict): Optional revenue filter, e.g., {"$gte": "$100M"} for $100M+
        employees_filter (dict): Optional employees filter, e.g., {"$gte": 100} for 100+ employees
        with_reasoning (bool): Whether to include reasoning for each company selection
        index_name (str): Name of the Pinecone index to search
    
    Returns:
        dict: Structured results containing query, companies, and optionally reasoning
    """
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        if not query_embedding:
            return {
                "query": query,
                "top_k": top_k,
                "filters_applied": {},
                "companies": [],
                "total_found": 0,
                "error": "Failed to generate query embedding",
                "search_summary": "Unable to process query"
            }
        
        # Search in Pinecone
        index = pc.Index(index_name)
        
        # Prepare query parameters
        query_params = {
            'vector': query_embedding,
            'top_k': top_k,
            'include_metadata': True
        }
        
        # Build metadata filter if any filters are provided
        filters_applied = {}
        if any([industry_filter, location_filter, revenue_filter, employees_filter]):
            filter_dict = {}
            if industry_filter:
                filter_dict['industry'] = industry_filter
                filters_applied['industry'] = industry_filter
            if location_filter:
                filter_dict['headquarters'] = location_filter
                filters_applied['location'] = location_filter
            if revenue_filter:
                filter_dict['revenue'] = revenue_filter
                filters_applied['revenue'] = revenue_filter
            if employees_filter:
                filter_dict['employees'] = employees_filter
                filters_applied['employees'] = employees_filter
            
            query_params['filter'] = filter_dict
        
        results = index.query(**query_params)
        
        # Structure the results with reasoning
        companies = []
        if results and results.get('matches'):
            for i, match in enumerate(results['matches'], 1):
                company_data = {
                    "rank": i,
                    "score": round(match['score'], 3),
                    "company_name": match['metadata']['company_name'],
                    "industry": match['metadata']['industry'],
                    "headquarters": match['metadata']['headquarters'],
                    "revenue": match['metadata']['revenue'],
                    "employees": match['metadata']['employees'],
                    "business_model": match['metadata']['business_model'],
                    "strategic_priorities": match['metadata']['strategic_priorities'],
                    "ideal_op_industry": match['metadata']['ideal_op_industry'],
                    "ideal_op_functional": match['metadata']['ideal_op_functional'],
                    "ideal_op_leadership": match['metadata']['ideal_op_leadership'],
                }
                
                # Add reasoning if requested
                if with_reasoning:
                    reasoning = generate_company_reasoning(query, match['metadata'], match['score'])
                    company_data["reasoning"] = reasoning
                
                companies.append(company_data)
        
        # Generate search summary
        search_summary = generate_search_summary(query, companies, filters_applied)
        
        return {
            "query": query,
            "top_k": top_k,
            "filters_applied": filters_applied,
            "companies": companies,
            "total_found": len(companies),
            "search_summary": search_summary
        }
        
    except Exception as e:
        return {
            "query": query,
            "top_k": top_k,
            "filters_applied": {},
            "companies": [],
            "total_found": 0,
            "error": f"Search error: {str(e)}",
            "search_summary": "Search encountered an error"
        }

def get_top_k_companies_with_reasoning(query, top_k=5, industry_filter=None, location_filter=None, revenue_filter=None, employees_filter=None, index_name='company-information-dummy'):
    """
    Get top k companies for a query and return structured results with reasoning
    
    Args:
        query (str): Search query text
        top_k (int): Number of results to return
        industry_filter (dict): Optional industry filter, e.g., {"$in": ["EdTech", "SaaS Data Analytics"]}
        location_filter (dict): Optional location filter, e.g., {"$in": ["California", "CA"]}
        revenue_filter (dict): Optional revenue filter, e.g., {"$gte": "$100M"} for $100M+
        employees_filter (dict): Optional employees filter, e.g., {"$gte": 100} for 100+ employees
        index_name (str): Name of the Pinecone index to search
    
    Returns:
        dict: Structured results containing query, companies, and reasoning
    """
    return get_top_k_companies(
        query=query,
        top_k=top_k,
        industry_filter=industry_filter,
        location_filter=location_filter,
        revenue_filter=revenue_filter,
        employees_filter=employees_filter,
        with_reasoning=True,
        index_name=index_name
    )

def parse_filter_params(industry_list=None, location_list=None, revenue_min=None, revenue_max=None, employees_min=None, employees_max=None):
    """
    Parse filter parameters from API query strings into Pinecone filter format
    
    Args:
        industry_list (str): Comma-separated list of industries
        location_list (str): Comma-separated list of locations
        revenue_min (str): Minimum revenue (e.g., "$100M")
        revenue_max (str): Maximum revenue (e.g., "$500M")
        employees_min (str): Minimum number of employees
        employees_max (str): Maximum number of employees
    
    Returns:
        tuple: (industry_filter, location_filter, revenue_filter, employees_filter)
    """
    industry_filter = None
    location_filter = None
    revenue_filter = None
    employees_filter = None
    
    # Parse industry filter
    if industry_list:
        industries = [i.strip() for i in industry_list.split(',') if i.strip()]
        if industries:
            industry_filter = {"$in": industries}
    
    # Parse location filter
    if location_list:
        locations = [l.strip() for l in location_list.split(',') if l.strip()]
        if locations:
            location_filter = {"$in": locations}
    
    # Parse revenue filter
    if revenue_min or revenue_max:
        revenue_conditions = []
        if revenue_min:
            revenue_conditions.append({"$gte": revenue_min})
        if revenue_max:
            revenue_conditions.append({"$lte": revenue_max})
        
        if len(revenue_conditions) == 1:
            revenue_filter = revenue_conditions[0]
        elif len(revenue_conditions) == 2:
            revenue_filter = {"$and": revenue_conditions}
    
    # Parse employees filter
    if employees_min or employees_max:
        try:
            employees_conditions = []
            if employees_min:
                employees_conditions.append({"$gte": int(employees_min)})
            if employees_max:
                employees_conditions.append({"$lte": int(employees_max)})
            
            if len(employees_conditions) == 1:
                employees_filter = employees_conditions[0]
            elif len(employees_conditions) == 2:
                employees_filter = {"$and": employees_conditions}
        except ValueError:
            # Invalid number format, ignore employees filter
            pass
    
    return industry_filter, location_filter, revenue_filter, employees_filter

def generate_company_reasoning(query, company_metadata, score):
    """
    Generate reasoning for why a company was selected using GPT-4o
    """
    try:
        # Prepare company information for the prompt
        company_info = f"""
Company Information:
- Name: {company_metadata['company_name']}
- Industry: {company_metadata['industry']}
- Headquarters: {company_metadata['headquarters']}
- Revenue: {company_metadata['revenue']}
- Employees: {company_metadata['employees']}
- Business Model: {company_metadata['business_model']}
- Strategic Priorities: {', '.join(company_metadata['strategic_priorities'])}
- Ideal Operating Partner Industry: {company_metadata['ideal_op_industry']}
- Ideal Operating Partner Functional Strengths: {', '.join(company_metadata['ideal_op_functional'])}
- Ideal Operating Partner Leadership Qualities: {', '.join(company_metadata['ideal_op_leadership'])}
- Semantic Similarity Score: {score:.3f}
"""

        system_prompt = """You are an expert business analyst. Your task is to explain why a specific company was selected as a match for a search query.

IMPORTANT: Only use the provided company information and query. If something is not explicitly mentioned in the provided documents, say "Not found" rather than making assumptions.

Analyze the company's characteristics against the search query and provide a concise, factual explanation (400-600 tokens) of why this company is a good match. Focus on:
1. Industry alignment
2. Location relevance (if applicable)
3. Business model fit
4. Strategic priorities alignment
5. Size/revenue characteristics
6. Semantic similarity score interpretation

Be specific and reference actual data from the company information provided."""

        user_prompt = f"""Search Query: "{query}"

{company_info}

Explain why this company was selected as a match for the search query. Use only the information provided above."""

        # Call OpenAI GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=600,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        reasoning = response.choices[0].message.content.strip()
        return reasoning
        
    except Exception as e:
        # Fallback to basic reasoning if GPT-4o fails
        return f"Selected based on semantic similarity (score: {score:.3f}). Error generating detailed reasoning: {str(e)}"

def generate_search_summary(query, companies, filters_applied):
    """
    Generate a summary of the search results
    """
    if not companies:
        return f"No companies found matching '{query}'"
    
    total_found = len(companies)
    industries = list(set(c['industry'] for c in companies))
    locations = list(set(c['headquarters'] for c in companies))
    revenue_range = f"{min(c['revenue'] for c in companies)} - {max(c['revenue'] for c in companies)}"
    
    summary_parts = [f"Found {total_found} companies matching '{query}'"]
    
    if filters_applied:
        filter_desc = []
        if 'industry' in filters_applied:
            filter_desc.append(f"in {filters_applied['industry']['$in'][0] if '$in' in filters_applied['industry'] else filters_applied['industry']}")
        if 'location' in filters_applied:
            filter_desc.append(f"in {filters_applied['location']['$in'][0] if '$in' in filters_applied['location'] else filters_applied['location']}")
        if 'revenue' in filters_applied:
            if '$gte' in filters_applied['revenue']:
                filter_desc.append(f"revenue {filters_applied['revenue']['$gte']}+")
        if 'employees' in filters_applied:
            if '$gte' in filters_applied['employees']:
                filter_desc.append(f"{filters_applied['employees']['$gte']}+ employees")
        
        if filter_desc:
            summary_parts.append(f"with filters: {', '.join(filter_desc)}")
    
    summary_parts.append(f"Industries: {', '.join(industries[:3])}{'...' if len(industries) > 3 else ''}")
    summary_parts.append(f"Locations: {', '.join(locations[:3])}{'...' if len(locations) > 3 else ''}")
    summary_parts.append(f"Revenue range: {revenue_range}")
    
    return ". ".join(summary_parts) + "."



def search_companies(query, top_k=5, industry_filter=None, location_filter=None, revenue_filter=None, employees_filter=None):
    """
    Search for companies using semantic search with optional metadata filtering
    
    Args:
        query (str): Search query text
        top_k (int): Number of results to return
        industry_filter (dict): Optional industry filter
        location_filter (dict): Optional location filter
        revenue_filter (dict): Optional revenue filter
        employees_filter (dict): Optional employees filter
    """
    print(f"\nSearching for: '{query}'")
    
    # Display active filters
    filters = []
    if industry_filter:
        filters.append(f"Industry: {industry_filter}")
    if location_filter:
        filters.append(f"Location: {location_filter}")
    if revenue_filter:
        filters.append(f"Revenue: {revenue_filter}")
    if employees_filter:
        filters.append(f"Employees: {employees_filter}")
    
    if filters:
        print(f"Active filters: {', '.join(filters)}")
    
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return None
        
        # Search in Pinecone
        index = pc.Index('company-information-dummy')
        
        # Prepare query parameters
        query_params = {
            'vector': query_embedding,
            'top_k': top_k,
            'include_metadata': True
        }
        
        # Build metadata filter if any filters are provided
        if any([industry_filter, location_filter, revenue_filter, employees_filter]):
            filter_dict = {}
            if industry_filter:
                filter_dict['industry'] = industry_filter
            if location_filter:
                filter_dict['headquarters'] = location_filter
            if revenue_filter:
                filter_dict['revenue'] = revenue_filter
            if employees_filter:
                filter_dict['employees'] = employees_filter
            
            query_params['filter'] = filter_dict
        
        results = index.query(**query_params)
        
        return results
    except Exception as e:
        print(f"❌ Error searching companies: {e}")
        return None

def display_search_results(results):
    """
    Display search results in a readable format
    """
    if not results or not results.get('matches'):
        print("No results found")
        return
    
    print(f"\nFound {len(results['matches'])} similar companies:")
    print("-" * 80)
    
    for i, match in enumerate(results['matches'], 1):
        print(f"{i}. Score: {match['score']:.3f}")
        print(f"   Company: {match['metadata']['company_name']}")
        print(f"   Industry: {match['metadata']['industry']}")
        print(f"   Location: {match['metadata']['headquarters']}")
        print(f"   Revenue: {match['metadata']['revenue']}")
        print(f"   Employees: {match['metadata']['employees']}")
        print(f"   Business Model: {match['metadata']['business_model'][:100]}...")
        print()

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

def interactive_search():
    """
    Interactive search interface for users
    """
    print("="*80)
    print("COMPANY SEARCH INTERFACE")
    print("="*80)
    
    while True:
        print("\nOptions:")
        print("1. Search by description")
        print("2. Search with industry filter")
        print("3. Search with location filter")
        print("4. Search with revenue filter")
        print("5. Search with multiple filters")
        print("6. Check index statistics")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            query = input("Enter your search query: ").strip()
            top_k = int(input("Number of results (default 5): ") or "5")
            results = search_companies(query, top_k=top_k)
            display_search_results(results)
            
        elif choice == '2':
            query = input("Enter your search query: ").strip()
            industry = input("Industry to filter by: ").strip()
            top_k = int(input("Number of results (default 5): ") or "5")
            
            if industry:
                results = search_companies(
                    query, 
                    top_k=top_k,
                    industry_filter={"$in": [industry]}
                )
            else:
                results = search_companies(query, top_k=top_k)
            display_search_results(results)
            
        elif choice == '3':
            query = input("Enter your search query: ").strip()
            location = input("Location to filter by: ").strip()
            top_k = int(input("Number of results (default 5): ") or "5")
            
            if location:
                results = search_companies(
                    query, 
                    top_k=top_k,
                    location_filter={"$in": [location]}
                )
            else:
                results = search_companies(query, top_k=top_k)
            display_search_results(results)
            
        elif choice == '4':
            query = input("Enter your search query: ").strip()
            min_revenue = input("Minimum revenue (e.g., $100M): ").strip()
            top_k = int(input("Number of results (default 5): ") or "5")
            
            if min_revenue:
                results = search_companies(
                    query, 
                    top_k=top_k,
                    revenue_filter={"$gte": min_revenue}
                )
            else:
                results = search_companies(query, top_k=top_k)
            display_search_results(results)
            
        elif choice == '5':
            query = input("Enter your search query: ").strip()
            industry = input("Industry (optional): ").strip()
            location = input("Location (optional): ").strip()
            min_revenue = input("Minimum revenue (optional): ").strip()
            min_employees = input("Minimum employees (optional): ").strip()
            top_k = int(input("Number of results (default 5): ") or "5")
            
            filters = {}
            if industry:
                filters['industry_filter'] = {"$in": [industry]}
            if location:
                filters['location_filter'] = {"$in": [location]}
            if min_revenue:
                filters['revenue_filter'] = {"$gte": min_revenue}
            if min_employees:
                filters['employees_filter'] = {"$gte": int(min_employees)}
            
            results = search_companies(query, top_k=top_k, **filters)
            display_search_results(results)
            
        elif choice == '6':
            check_index_statistics()
            
        elif choice == '7':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    # Check if index exists and has data
    stats = check_index_statistics()
    
    if stats and stats.get('total_vector_count', 0) > 0:
        print("✅ Index found with data.")
        
        # Demonstrate the new function
        demonstrate_get_top_k_companies()
        
        print("\n" + "="*80)
        print("INTERACTIVE SEARCH INTERFACE")
        print("="*80)
        interactive_search()
    else:
        print("❌ No data found in index. Please run company_embed.py first to populate the database.")
        
        # Run some example searches anyway
        print("\n" + "="*80)
        print("EXAMPLE SEARCHES")
        print("="*80)
        
        # Search 1: EdTech companies
        print("\n1. Searching for EdTech companies...")
        results1 = search_companies("edtech companies with digital transformation focus")
        display_search_results(results1)
        
        # Search 2: Location-based search
        print("\n2. Searching by location...")
        results2 = search_companies("companies in California")
        display_search_results(results2)
        
        # Search 3: Industry-based search WITH filter
        print("\n3. Searching for SaaS companies...")
        results3 = search_companies(
            "data analytics companies",
            industry_filter={"$in": ["SaaS Data Analytics"]}
        )
        display_search_results(results3)
        
        # Search 4: Revenue-based search with filter
        print("\n4. Searching for high-revenue companies...")
        results4 = search_companies(
            "successful companies",
            revenue_filter={"$gte": "$100M"}
        )
        display_search_results(results4)
        
        # Search 5: Multiple filters
        print("\n5. Searching for large technology companies in California...")
        results5 = search_companies(
            "technology companies",
            location_filter={"$in": ["California", "CA"]},
            employees_filter={"$gte": 200}
        )
        display_search_results(results5)
