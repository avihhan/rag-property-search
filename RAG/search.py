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

def get_top_k_properties_with_reasoning(query, top_k=5, price_filter=None, location_filter=None, bedrooms_filter=None):
    """
    Get top k properties for a query and return structured results with reasoning
    
    Args:
        query (str): Search query text
        top_k (int): Number of results to return
        price_filter (dict): Optional price filter, e.g., {"$lte": 1000000} for under $1M
        location_filter (dict): Optional location filter, e.g., {"$in": ["California", "CA"]}
        bedrooms_filter (dict): Optional bedrooms filter, e.g., {"$gte": 3} for 3+ bedrooms
    
    Returns:
        dict: Structured results containing query, properties, and reasoning
        {
            "query": "user's original query",
            "top_k": 5,
            "filters_applied": {...},
            "properties": [
                {
                    "rank": 1,
                    "score": 0.85,
                    "property_name": "Penthouse #23",
                    "location": "Palm Springs, CA",
                    "bedrooms": 5,
                    "view": "lake view",
                    "price_usd": 2206633,
                    "size_sqft": 2349,
                    "description": "Original description from CSV",
                    "property_type": "Penthouse",
                    "reasoning": "Selected because it's a penthouse with lake view, matching luxury criteria"
                },
                ...
            ],
            "total_found": 5,
            "search_summary": "Found 5 luxury properties matching your criteria"
        }
    """
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        if not query_embedding:
            return {
                "query": query,
                "top_k": top_k,
                "filters_applied": {},
                "properties": [],
                "total_found": 0,
                "error": "Failed to generate query embedding",
                "search_summary": "Unable to process query"
            }
        
        # Search in Pinecone
        index = pc.Index('property-listings')
        
        # Prepare query parameters
        query_params = {
            'vector': query_embedding,
            'top_k': top_k,
            'include_metadata': True
        }
        
        # Build metadata filter if any filters are provided
        filters_applied = {}
        if any([price_filter, location_filter, bedrooms_filter]):
            filter_dict = {}
            if price_filter:
                filter_dict['price_usd'] = price_filter
                filters_applied['price'] = price_filter
            if location_filter:
                filter_dict['location'] = location_filter
                filters_applied['location'] = location_filter
            if bedrooms_filter:
                filter_dict['bedrooms'] = bedrooms_filter
                filters_applied['bedrooms'] = bedrooms_filter
            
            query_params['filter'] = filter_dict
        
        results = index.query(**query_params)
        
        # Structure the results with reasoning
        properties = []
        if results and results.get('matches'):
            for i, match in enumerate(results['matches'], 1):
                # Generate reasoning for why this property was selected
                reasoning = generate_property_reasoning(query, match['metadata'], match['score'])
                
                property_data = {
                    "rank": i,
                    "score": round(match['score'], 3),
                    "property_name": match['metadata']['property_name'],
                    "location": match['metadata']['location'],
                    "bedrooms": match['metadata']['bedrooms'],
                    "view": match['metadata']['view'],
                    "price_usd": match['metadata']['price_usd'],
                    "size_sqft": match['metadata']['size_sqft'],
                    "description": match['metadata']['description'],
                    "property_type": match['metadata']['property_type'],
                    "reasoning": reasoning
                }
                properties.append(property_data)
        
        # Generate search summary
        search_summary = generate_search_summary(query, properties, filters_applied)
        
        return {
            "query": query,
            "top_k": top_k,
            "filters_applied": filters_applied,
            "properties": properties,
            "total_found": len(properties),
            "search_summary": search_summary
        }
        
    except Exception as e:
        return {
            "query": query,
            "top_k": top_k,
            "filters_applied": {},
            "properties": [],
            "total_found": 0,
            "error": f"Search error: {str(e)}",
            "search_summary": "Search encountered an error"
        }

def generate_property_reasoning(query, property_metadata, score):
    """
    Generate reasoning for why a property was selected based on query and metadata
    """
    query_lower = query.lower()
    property_name = property_metadata['property_name']
    property_type = property_metadata['property_type']
    location = property_metadata['location']
    bedrooms = property_metadata['bedrooms']
    view = property_metadata['view']
    price = property_metadata['price_usd']
    
    reasons = []
    
    # Check for property type matches
    if any(word in query_lower for word in ['penthouse', 'luxury', 'high-end']):
        if property_type.lower() in ['penthouse', 'villa']:
            reasons.append(f"matches luxury property type ({property_type})")
    
    if any(word in query_lower for word in ['house', 'home', 'family']):
        if property_type.lower() in ['house', 'townhouse']:
            reasons.append(f"matches family home type ({property_type})")
    
    if any(word in query_lower for word in ['apartment', 'condo', 'studio']):
        if property_type.lower() in ['apartment', 'condo', 'studio']:
            reasons.append(f"matches residential type ({property_type})")
    
    # Check for view matches
    if any(word in query_lower for word in ['ocean', 'water', 'beach']):
        if 'ocean' in view.lower() or 'lake' in view.lower() or 'river' in view.lower():
            reasons.append(f"has water view ({view})")
    
    if any(word in query_lower for word in ['mountain', 'forest', 'nature']):
        if 'mountain' in view.lower() or 'forest' in view.lower() or 'garden' in view.lower():
            reasons.append(f"has nature view ({view})")
    
    if any(word in query_lower for word in ['city', 'urban', 'skyline']):
        if 'city' in view.lower() or 'skyline' in view.lower():
            reasons.append(f"has urban view ({view})")
    
    # Check for location matches
    if any(word in query_lower for word in ['california', 'ca', 'cali']):
        if 'california' in location.lower() or ', ca' in location.lower():
            reasons.append(f"located in California ({location})")
    
    if any(word in query_lower for word in ['new york', 'ny', 'brooklyn', 'manhattan']):
        if 'new york' in location.lower() or ', ny' in location.lower():
            reasons.append(f"located in New York area ({location})")
    
    # Check for bedroom requirements
    if any(word in query_lower for word in ['bedroom', 'bedrooms', 'br']):
        if '3' in query_lower or 'three' in query_lower:
            if bedrooms >= 3:
                reasons.append(f"has {bedrooms} bedrooms (meets 3+ requirement)")
        elif '2' in query_lower or 'two' in query_lower:
            if bedrooms >= 2:
                reasons.append(f"has {bedrooms} bedrooms (meets 2+ requirement)")
        elif '4' in query_lower or 'four' in query_lower:
            if bedrooms >= 4:
                reasons.append(f"has {bedrooms} bedrooms (meets 4+ requirement)")
    
    # Check for price-related reasoning
    if any(word in query_lower for word in ['affordable', 'budget', 'cheap', 'low price']):
        if price <= 1000000:
            reasons.append(f"affordable price (${price:,})")
    
    if any(word in query_lower for word in ['luxury', 'expensive', 'high-end', 'premium']):
        if price >= 2000000:
            reasons.append(f"luxury price point (${price:,})")
    
    # Add similarity score reasoning
    if score >= 0.6:
        reasons.append("high semantic similarity to query")
    elif score >= 0.5:
        reasons.append("good semantic similarity to query")
    else:
        reasons.append("moderate semantic similarity to query")
    
    # Combine reasons
    if reasons:
        return f"Selected because: {', '.join(reasons)}"
    else:
        return f"Selected based on semantic similarity (score: {score:.3f})"

def generate_search_summary(query, properties, filters_applied):
    """
    Generate a summary of the search results
    """
    if not properties:
        return f"No properties found matching '{query}'"
    
    total_found = len(properties)
    price_range = f"${min(p['price_usd'] for p in properties):,} - ${max(p['price_usd'] for p in properties):,}"
    locations = list(set(p['location'] for p in properties))
    property_types = list(set(p['property_type'] for p in properties))
    
    summary_parts = [f"Found {total_found} properties matching '{query}'"]
    
    if filters_applied:
        filter_desc = []
        if 'price' in filters_applied:
            if '$lte' in filters_applied['price']:
                filter_desc.append(f"under ${filters_applied['price']['$lte']:,}")
            elif '$gte' in filters_applied['price']:
                filter_desc.append(f"over ${filters_applied['price']['$gte']:,}")
        if 'bedrooms' in filters_applied:
            if '$gte' in filters_applied['bedrooms']:
                filter_desc.append(f"{filters_applied['bedrooms']['$gte']}+ bedrooms")
        if 'location' in filters_applied:
            filter_desc.append(f"in {filters_applied['location']['$in'][0]}")
        
        if filter_desc:
            summary_parts.append(f"with filters: {', '.join(filter_desc)}")
    
    summary_parts.append(f"Price range: {price_range}")
    summary_parts.append(f"Locations: {', '.join(locations[:3])}{'...' if len(locations) > 3 else ''}")
    summary_parts.append(f"Types: {', '.join(property_types)}")
    
    return ". ".join(summary_parts) + "."

def get_top_k_properties(query, top_k=5, price_filter=None, location_filter=None, bedrooms_filter=None):
    """
    Get top k properties for a query and return structured results
    
    Args:
        query (str): Search query text
        top_k (int): Number of results to return
        price_filter (dict): Optional price filter, e.g., {"$lte": 1000000} for under $1M
        location_filter (dict): Optional location filter, e.g., {"$in": ["California", "CA"]}
        bedrooms_filter (dict): Optional bedrooms filter, e.g., {"$gte": 3} for 3+ bedrooms
    
    Returns:
        dict: Structured results containing query and properties
        {
            "query": "user's original query",
            "top_k": 5,
            "filters_applied": {...},
            "properties": [
                {
                    "rank": 1,
                    "score": 0.85,
                    "property_name": "Penthouse #23",
                    "location": "Palm Springs, CA",
                    "bedrooms": 5,
                    "view": "lake view",
                    "price_usd": 2206633,
                    "size_sqft": 2349,
                    "description": "Original description from CSV"
                },
                ...
            ],
            "total_found": 5
        }
    """
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        if not query_embedding:
            return {
                "query": query,
                "top_k": top_k,
                "filters_applied": {},
                "properties": [],
                "total_found": 0,
                "error": "Failed to generate query embedding"
            }
        
        # Search in Pinecone
        index = pc.Index('property-listings')
        
        # Prepare query parameters
        query_params = {
            'vector': query_embedding,
            'top_k': top_k,
            'include_metadata': True
        }
        
        # Build metadata filter if any filters are provided
        filters_applied = {}
        if any([price_filter, location_filter, bedrooms_filter]):
            filter_dict = {}
            if price_filter:
                filter_dict['price_usd'] = price_filter
                filters_applied['price'] = price_filter
            if location_filter:
                filter_dict['location'] = location_filter
                filters_applied['location'] = location_filter
            if bedrooms_filter:
                filter_dict['bedrooms'] = bedrooms_filter
                filters_applied['bedrooms'] = bedrooms_filter
            
            query_params['filter'] = filter_dict
        
        results = index.query(**query_params)
        
        # Structure the results
        properties = []
        if results and results.get('matches'):
            for i, match in enumerate(results['matches'], 1):
                property_data = {
                    "rank": i,
                    "score": round(match['score'], 3),
                    "property_name": match['metadata']['property_name'],
                    "location": match['metadata']['location'],
                    "bedrooms": match['metadata']['bedrooms'],
                    "view": match['metadata']['view'],
                    "price_usd": match['metadata']['price_usd'],
                    "size_sqft": match['metadata']['size_sqft'],
                    "description": match['metadata']['description'],
                    "property_type": match['metadata']['property_type']
                }
                properties.append(property_data)
        
        return {
            "query": query,
            "top_k": top_k,
            "filters_applied": filters_applied,
            "properties": properties,
            "total_found": len(properties)
        }
        
    except Exception as e:
        return {
            "query": query,
            "top_k": top_k,
            "filters_applied": {},
            "properties": [],
            "total_found": 0,
            "error": f"Search error: {str(e)}"
        }

def search_properties(query, top_k=5, price_filter=None, location_filter=None, bedrooms_filter=None):
    """
    Search for properties using semantic search with optional metadata filtering
    
    Args:
        query (str): Search query text
        top_k (int): Number of results to return
        price_filter (dict): Optional price filter, e.g., {"$lte": 1000000} for under $1M
        location_filter (dict): Optional location filter, e.g., {"$in": ["California", "CA"]}
        bedrooms_filter (dict): Optional bedrooms filter, e.g., {"$gte": 3} for 3+ bedrooms
    """
    print(f"\nSearching for: '{query}'")
    
    # Display active filters
    filters = []
    if price_filter:
        filters.append(f"Price: {price_filter}")
    if location_filter:
        filters.append(f"Location: {location_filter}")
    if bedrooms_filter:
        filters.append(f"Bedrooms: {bedrooms_filter}")
    
    if filters:
        print(f"Active filters: {', '.join(filters)}")
    
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return None
        
        # Search in Pinecone
        index = pc.Index('property-listings')
        
        # Prepare query parameters
        query_params = {
            'vector': query_embedding,
            'top_k': top_k,
            'include_metadata': True
        }
        
        # Build metadata filter if any filters are provided
        if any([price_filter, location_filter, bedrooms_filter]):
            filter_dict = {}
            if price_filter:
                filter_dict['price_usd'] = price_filter
            if location_filter:
                filter_dict['location'] = location_filter
            if bedrooms_filter:
                filter_dict['bedrooms'] = bedrooms_filter
            
            query_params['filter'] = filter_dict
        
        results = index.query(**query_params)
        
        return results
    except Exception as e:
        print(f"❌ Error searching properties: {e}")
        return None

def display_search_results(results):
    """
    Display search results in a readable format
    """
    if not results or not results.get('matches'):
        print("No results found")
        return
    
    print(f"\nFound {len(results['matches'])} similar properties:")
    print("-" * 60)
    
    for i, match in enumerate(results['matches'], 1):
        print(f"{i}. Score: {match['score']:.3f}")
        print(f"   Property: {match['metadata']['property_name']}")
        print(f"   Location: {match['metadata']['location']}")
        print(f"   Bedrooms: {match['metadata']['bedrooms']}")
        print(f"   View: {match['metadata']['view']}")
        print(f"   Price: ${match['metadata']['price_usd']:,}")
        print(f"   Size: {match['metadata']['size_sqft']} sqft")
        print()

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

def interactive_search():
    """
    Interactive search interface for users
    """
    print("="*60)
    print("PROPERTY SEARCH INTERFACE")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Search by description")
        print("2. Search with price filter")
        print("3. Search with location filter")
        print("4. Search with multiple filters")
        print("5. Check index statistics")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            query = input("Enter your search query: ").strip()
            top_k = int(input("Number of results (default 5): ") or "5")
            results = search_properties(query, top_k=top_k)
            display_search_results(results)
            
        elif choice == '2':
            query = input("Enter your search query: ").strip()
            max_price = input("Maximum price (e.g., 1000000 for $1M): ").strip()
            top_k = int(input("Number of results (default 5): ") or "5")
            
            if max_price:
                results = search_properties(
                    query, 
                    top_k=top_k,
                    price_filter={"$lte": int(max_price)}
                )
            else:
                results = search_properties(query, top_k=top_k)
            display_search_results(results)
            
        elif choice == '3':
            query = input("Enter your search query: ").strip()
            location = input("Location to filter by: ").strip()
            top_k = int(input("Number of results (default 5): ") or "5")
            
            if location:
                results = search_properties(
                    query, 
                    top_k=top_k,
                    location_filter={"$in": [location]}
                )
            else:
                results = search_properties(query, top_k=top_k)
            display_search_results(results)
            
        elif choice == '4':
            query = input("Enter your search query: ").strip()
            max_price = input("Maximum price (optional): ").strip()
            location = input("Location (optional): ").strip()
            min_bedrooms = input("Minimum bedrooms (optional): ").strip()
            top_k = int(input("Number of results (default 5): ") or "5")
            
            filters = {}
            if max_price:
                filters['price_filter'] = {"$lte": int(max_price)}
            if location:
                filters['location_filter'] = {"$in": [location]}
            if min_bedrooms:
                filters['bedrooms_filter'] = {"$gte": int(min_bedrooms)}
            
            results = search_properties(query, top_k=top_k, **filters)
            display_search_results(results)
            
        elif choice == '5':
            check_index_statistics()
            
        elif choice == '6':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

def demonstrate_get_top_k_properties():
    """
    Demonstrate the get_top_k_properties function with various examples
    """
    print("="*60)
    print("DEMONSTRATING get_top_k_properties FUNCTION")
    print("="*60)
    
    # Example 1: Basic search
    print("\n1. Basic search for luxury properties:")
    results1 = get_top_k_properties("luxury penthouse with ocean view", top_k=3)
    print(f"Query: {results1['query']}")
    print(f"Found: {results1['total_found']} properties")
    for prop in results1['properties']:
        print(f"  {prop['rank']}. {prop['property_name']} - ${prop['price_usd']:,} - Score: {prop['score']}")
    
    # Example 2: Search with price filter
    print("\n2. Search with price filter (under $1M):")
    results2 = get_top_k_properties(
        "affordable properties", 
        top_k=3,
        price_filter={"$lte": 1000000}
    )
    print(f"Query: {results2['query']}")
    print(f"Filters: {results2['filters_applied']}")
    print(f"Found: {results2['total_found']} properties")
    for prop in results2['properties']:
        print(f"  {prop['rank']}. {prop['property_name']} - ${prop['price_usd']:,} - Score: {prop['score']}")
    
    # Example 3: Search with multiple filters
    print("\n3. Search with multiple filters (3+ bedrooms, under $2M):")
    results3 = get_top_k_properties(
        "family homes",
        top_k=3,
        price_filter={"$lte": 2000000},
        bedrooms_filter={"$gte": 3}
    )
    print(f"Query: {results3['query']}")
    print(f"Filters: {results3['filters_applied']}")
    print(f"Found: {results3['total_found']} properties")
    for prop in results3['properties']:
        print(f"  {prop['rank']}. {prop['property_name']} - {prop['bedrooms']}BR - ${prop['price_usd']:,} - Score: {prop['score']}")
    
    # Example 4: Return full structured data
    print("\n4. Full structured data for one property:")
    results4 = get_top_k_properties("luxury villa", top_k=1)
    if results4['properties']:
        prop = results4['properties'][0]
        print(f"Query: {results4['query']}")
        print(f"Property: {prop['property_name']}")
        print(f"Location: {prop['location']}")
        print(f"Bedrooms: {prop['bedrooms']}")
        print(f"View: {prop['view']}")
        print(f"Price: ${prop['price_usd']:,}")
        print(f"Size: {prop['size_sqft']} sqft")
        print(f"Score: {prop['score']}")
        print(f"Type: {prop['property_type']}")

if __name__ == "__main__":
    # Check if index exists and has data
    stats = check_index_statistics()
    
    if stats and stats.get('total_vector_count', 0) > 0:
        print("✅ Index found with data.")
        
        # Demonstrate the new function
        demonstrate_get_top_k_properties()
        
        print("\n" + "="*60)
        print("INTERACTIVE SEARCH INTERFACE")
        print("="*60)
        interactive_search()
    else:
        print("❌ No data found in index. Please run embed.py first to populate the database.")
        
        # Run some example searches anyway
        print("\n" + "="*60)
        print("EXAMPLE SEARCHES")
        print("="*60)
        
        # Search 1: Luxury properties (no price filter)
        print("\n1. Searching for luxury properties...")
        results1 = search_properties("luxury penthouse with ocean view")
        display_search_results(results1)
        
        # Search 2: Location-based search (no price filter)
        print("\n2. Searching by location...")
        results2 = search_properties("properties in California")
        display_search_results(results2)
        
        # Search 3: Price-based search WITH filter
        print("\n3. Searching for affordable properties under $1M...")
        results3 = search_properties(
            "affordable properties under 1 million",
            price_filter={"$lte": 1000000}
        )
        display_search_results(results3)
        
        # Search 4: Budget properties with filter
        print("\n4. Searching for budget properties for low prices.")
        results4 = search_properties(
            "budget friendly properties",
            price_filter={"$lte": 500000}
        )
        display_search_results(results4)
        
        # Search 5: High-end properties with filter
        print("\n5. Searching for high-end properties over $2M...")
        results5 = search_properties(
            "luxury high-end properties",
            price_filter={"$gte": 2000000}
        )
        display_search_results(results5)
