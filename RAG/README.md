# RAG Property Search System

This folder contains a modular RAG (Retrieval-Augmented Generation) system for property search, separated into embedding and searching functionalities.

## 📁 Project Structure

```
RAG/
├── embed.py              # Embedding module - converts properties to vectors
├── search.py             # Searching module - queries the vector database
├── test_get_top_k.py     # Test script for the get_top_k_properties function
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Set Up Environment Variables

Create a `.env` file in the root directory with your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 2. Install Dependencies

```bash
pip install openai pinecone-client python-dotenv pandas
```

### 3. Run the Embedding Process

First, convert your property listings to vector embeddings:

```bash
python RAG/embed.py
```

This will:
- ✅ Load your `property_listings_dummy.csv` file
- ✅ Create comprehensive property descriptions
- ✅ Generate embeddings using OpenAI's `text-embedding-3-large` model
- ✅ Store vectors in Pinecone with full metadata
- ✅ Display index statistics

### 4. Run the Search Interface

After embedding, start the interactive search:

```bash
python RAG/search.py
```

This provides:
- 🔍 Interactive search interface
- 🎯 Semantic property search
- 💰 Price filtering capabilities
- 📍 Location filtering
- 🏠 Bedroom filtering
- 📊 Index statistics

### 5. Test the get_top_k_properties Function

Test the new structured search function:

```bash
python RAG/test_get_top_k.py
```

This demonstrates:
- ✅ Basic semantic search
- ✅ Price filtering
- ✅ Multiple filter combinations
- ✅ Full property details

## 📋 Module Details

### `embed.py` - Embedding Module

**Purpose**: Converts property listings from CSV to vector embeddings

**Key Functions**:
- `create_property_description()`: Creates comprehensive text descriptions
- `generate_embedding()`: Uses OpenAI's text-embedding-3-large model
- `create_pinecone_index()`: Sets up Pinecone index with proper configuration
- `process_property_listings()`: Main function that orchestrates the embedding process

**Usage**:
```bash
python RAG/embed.py
```

### `search.py` - Searching Module

**Purpose**: Provides semantic search capabilities with metadata filtering

**Key Functions**:
- `get_top_k_properties()`: **NEW** - Returns structured results with query and properties
- `search_properties()`: Main search function with filtering options
- `display_search_results()`: Formats and displays search results
- `interactive_search()`: User-friendly interactive interface
- `check_index_statistics()`: Verifies data in the index

**Usage**:
```bash
python RAG/search.py
```

## 🔍 Search Features

### NEW: get_top_k_properties() Function

This is the main function you'll want to use for programmatic access:

```python
from search import get_top_k_properties

# Basic search
results = get_top_k_properties("luxury penthouse with ocean view", top_k=5)

# Search with price filter
results = get_top_k_properties(
    "affordable properties", 
    top_k=3,
    price_filter={"$lte": 1000000}
)

# Search with multiple filters
results = get_top_k_properties(
    "family homes",
    top_k=5,
    price_filter={"$lte": 2000000},
    bedrooms_filter={"$gte": 3},
    location_filter={"$in": ["California", "CA"]}
)
```

**Returns structured data**:
```python
{
    "query": "user's original query",
    "top_k": 5,
    "filters_applied": {
        "price": {"$lte": 1000000},
        "bedrooms": {"$gte": 3}
    },
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
            "property_type": "Penthouse"
        },
        # ... more properties
    ],
    "total_found": 5
}
```

### Basic Search
```python
# Simple semantic search
results = search_properties("luxury penthouse with ocean view")
```

### Search with Price Filter
```python
# Properties under $1M
results = search_properties(
    "affordable properties",
    price_filter={"$lte": 1000000}
)
```

### Search with Location Filter
```python
# Properties in California
results = search_properties(
    "properties in California",
    location_filter={"$in": ["California", "CA"]}
)
```

### Search with Multiple Filters
```python
# 3+ bedroom properties under $1M in California
results = search_properties(
    "family homes",
    price_filter={"$lte": 1000000},
    location_filter={"$in": ["California", "CA"]},
    bedrooms_filter={"$gte": 3}
)
```

## 🎯 Available Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$lte` | Less than or equal to | `{"$lte": 1000000}` |
| `$gte` | Greater than or equal to | `{"$gte": 2000000}` |
| `$lt` | Less than | `{"$lt": 500000}` |
| `$gt` | Greater than | `{"$gt": 1000000}` |
| `$in` | In list of values | `{"$in": ["California", "CA"]}` |

## 📊 Interactive Search Options

When running `search.py`, you'll get these options:

1. **Search by description** - Basic semantic search
2. **Search with price filter** - Add maximum price constraint
3. **Search with location filter** - Filter by specific location
4. **Search with multiple filters** - Combine price, location, and bedroom filters
5. **Check index statistics** - View database statistics
6. **Exit** - Close the interface

## 🔧 Configuration

### Pinecone Settings
- **Index Name**: `property-listings`
- **Dimension**: 3072 (for text-embedding-3-large)
- **Metric**: Cosine similarity
- **Region**: us-east-1 (free plan compatible)

### OpenAI Settings
- **Model**: text-embedding-3-large
- **Rate Limiting**: 0.15 second delay between requests

## 🚨 Troubleshooting

### Common Issues

1. **"No data found in index"**
   - Run `embed.py` first to populate the database

2. **API Key Errors**
   - Ensure your `.env` file has correct API keys
   - Check that keys are valid and have sufficient credits

3. **Rate Limit Errors**
   - The system includes built-in delays to avoid rate limits
   - If you still hit limits, increase the delay in `embed.py`

4. **Pinecone Region Errors**
   - Free plan only supports `us-east-1` region
   - Paid plans can use other regions

## 📈 Performance Notes

- **Embedding Generation**: ~0.15 seconds per property (with rate limiting)
- **Search Response**: ~1-2 seconds per query
- **Index Size**: ~100 properties = ~300KB of vector data
- **Memory Usage**: Minimal (vectors stored in Pinecone cloud)

## 🔄 Workflow

1. **Initial Setup**: Run `embed.py` once to populate the database
2. **Regular Usage**: Use `get_top_k_properties()` for programmatic access
3. **Interactive Search**: Run `search.py` for interactive interface
4. **Data Updates**: Re-run `embed.py` when you have new property data
5. **Testing**: Use `test_get_top_k.py` to verify functionality

## 💡 Usage Examples

### For Web Applications
```python
from search import get_top_k_properties

def search_properties_api(query, max_price=None, min_bedrooms=None):
    filters = {}
    if max_price:
        filters['price_filter'] = {"$lte": max_price}
    if min_bedrooms:
        filters['bedrooms_filter'] = {"$gte": min_bedrooms}
    
    return get_top_k_properties(query, top_k=10, **filters)
```

### For Data Analysis
```python
from search import get_top_k_properties

# Get all luxury properties
luxury_results = get_top_k_properties("luxury properties", top_k=50)

# Analyze price distribution
prices = [prop['price_usd'] for prop in luxury_results['properties']]
avg_price = sum(prices) / len(prices)
print(f"Average luxury property price: ${avg_price:,.0f}")
```

This modular approach provides clear separation of concerns and makes the system easy to maintain and extend!
