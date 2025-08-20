# Company Information RAG System

This folder contains a specialized RAG (Retrieval-Augmented Generation) system for company information, designed to search and analyze company data from the `dummy_companies.json` file.

## 📁 Project Structure

```
RAG/
├── company_embed.py          # Company embedding module - converts companies to vectors
├── company_search.py         # Company searching module - queries the vector database
├── test_company_search.py    # Test script for company search functionality
├── COMPANIES_README.md       # This file
├── embed.py                  # Property embedding module (original)
├── search.py                 # Property searching module (original)
├── test_get_top_k.py         # Test script for property search (original)
└── README.md                 # Original property RAG documentation
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

### 3. Run the Company Embedding Process

First, convert your company data to vector embeddings:

```bash
python RAG/company_embed.py
```

This will:
- ✅ Load your `dummy_companies.json` file
- ✅ Create comprehensive company descriptions
- ✅ Generate embeddings using OpenAI's `text-embedding-3-large` model
- ✅ Store vectors in Pinecone index `company-information-dummy` with full metadata
- ✅ Display index statistics

### 4. Run the Company Search Interface

After embedding, start the interactive search:

```bash
python RAG/company_search.py
```

This provides:
- 🔍 Interactive search interface
- 🎯 Semantic company search
- 🏭 Industry filtering capabilities
- 📍 Location filtering
- 💰 Revenue filtering
- 👥 Employee count filtering
- 📊 Index statistics

### 5. Test the Company Search Functionality

Test the company search functions:

```bash
python RAG/test_company_search.py
```

This demonstrates:
- ✅ Basic semantic search
- ✅ Industry filtering
- ✅ Location filtering
- ✅ Revenue filtering
- ✅ Employee filtering
- ✅ Multiple filter combinations
- ✅ Reasoning functionality

## 📋 Module Details

### `company_embed.py` - Company Embedding Module

**Purpose**: Converts company data from JSON to vector embeddings

**Key Functions**:
- `create_company_description()`: Creates comprehensive text descriptions
- `generate_embedding()`: Uses OpenAI's text-embedding-3-large model
- `create_pinecone_index()`: Sets up Pinecone index `company-information-dummy`
- `process_company_data()`: Main function that orchestrates the embedding process

**Usage**:
```bash
python RAG/company_embed.py
```

### `company_search.py` - Company Searching Module

**Purpose**: Provides semantic search capabilities with metadata filtering

**Key Functions**:
- `get_top_k_companies()`: Returns structured results with query and companies
- `get_top_k_companies_with_reasoning()`: Returns results with AI-generated reasoning
- `search_companies()`: Main search function with filtering options
- `display_search_results()`: Formats and displays search results
- `interactive_search()`: User-friendly interactive interface
- `check_index_statistics()`: Verifies data in the index

**Usage**:
```bash
python RAG/company_search.py
```

## 🔍 Search Features

### NEW: get_top_k_companies() Function

This is the main function you'll want to use for programmatic access:

```python
from company_search import get_top_k_companies

# Basic search
results = get_top_k_companies("edtech companies with digital transformation focus", top_k=5)

# Search with industry filter
results = get_top_k_companies(
    "data analytics companies", 
    top_k=3,
    industry_filter={"$in": ["SaaS Data Analytics"]}
)

# Search with multiple filters
results = get_top_k_companies(
    "technology companies",
    top_k=5,
    location_filter={"$in": ["California", "CA"]},
    employees_filter={"$gte": 200},
    revenue_filter={"$gte": "$100M"}
)
```

**Returns structured data**:
```python
{
    "query": "user's original query",
    "top_k": 5,
    "filters_applied": {
        "industry": {"$in": ["SaaS Data Analytics"]},
        "location": {"$in": ["California", "CA"]}
    },
    "companies": [
        {
            "rank": 1,
            "score": 0.85,
            "company_name": "Company 14",
            "industry": "SaaS Data Analytics",
            "headquarters": "Atlanta, GA",
            "revenue": "$106M",
            "employees": 242,
            "business_model": "Operates in the saas data analytics sector...",
            "strategic_priorities": ["Enhance digital transformation", ...],
            "ideal_op_industry": "SaaS Data Analytics",
            "ideal_op_functional": ["GTM strategy", "FDA compliance"],
            "ideal_op_leadership": ["Execution-focused", "Health-conscious branding"]
        },
        # ... more companies
    ],
    "total_found": 5
}
```

### Basic Search
```python
# Simple semantic search
results = search_companies("edtech companies with digital transformation focus")
```

### Search with Industry Filter
```python
# Companies in specific industry
results = search_companies(
    "data analytics companies",
    industry_filter={"$in": ["SaaS Data Analytics"]}
)
```

### Search with Location Filter
```python
# Companies in specific location
results = search_companies(
    "technology companies",
    location_filter={"$in": ["California", "CA"]}
)
```

### Search with Revenue Filter
```python
# Companies with minimum revenue
results = search_companies(
    "successful companies",
    revenue_filter={"$gte": "$100M"}
)
```

### Search with Employee Filter
```python
# Companies with minimum employees
results = search_companies(
    "large companies",
    employees_filter={"$gte": 300}
)
```

### Search with Multiple Filters
```python
# Combine multiple filters
results = search_companies(
    "technology companies",
    industry_filter={"$in": ["SaaS Data Analytics"]},
    location_filter={"$in": ["California", "CA"]},
    revenue_filter={"$gte": "$50M"},
    employees_filter={"$gte": 200}
)
```

## 🎯 Available Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$lte` | Less than or equal to | `{"$lte": 500}` |
| `$gte` | Greater than or equal to | `{"$gte": "$100M"}` |
| `$lt` | Less than | `{"$lt": 200}` |
| `$gt` | Greater than | `{"$gt": "$50M"}` |
| `$in` | In list of values | `{"$in": ["EdTech", "SaaS Data Analytics"]}` |

## 📊 Interactive Search Options

When running `company_search.py`, you'll get these options:

1. **Search by description** - Basic semantic search
2. **Search with industry filter** - Filter by specific industry
3. **Search with location filter** - Filter by specific location
4. **Search with revenue filter** - Filter by minimum revenue
5. **Search with multiple filters** - Combine industry, location, revenue, and employee filters
6. **Check index statistics** - View database statistics
7. **Exit** - Close the interface

## 🔧 Configuration

### Pinecone Settings
- **Index Name**: `company-information-dummy`
- **Dimension**: 3072 (for text-embedding-3-large)
- **Metric**: Cosine similarity
- **Region**: us-east-1 (free plan compatible)

### OpenAI Settings
- **Model**: text-embedding-3-large
- **Rate Limiting**: 0.15 second delay between requests

## 🏭 Available Industries

The system supports these industries from the dummy data:
- **EdTech** - Education Technology
- **SaaS Data Analytics** - Software as a Service Data Analytics
- **Medical Equipment Distribution** - Healthcare Equipment
- **Freight & Logistics** - Transportation and Logistics
- **Organic Packaged Foods** - Food and Beverage
- **Water Filtration Technology** - Environmental Technology
- **Nutraceutical Manufacturing** - Health Supplements
- **Commercial Landscaping Services** - Landscaping
- **Precision Metal Fabrication** - Manufacturing
- **Renewable Energy Infrastructure** - Energy

## 📍 Available Locations

Major locations in the dataset:
- **California**: San Francisco, CA
- **Texas**: Austin, TX
- **Massachusetts**: Boston, MA
- **Florida**: Miami, FL
- **Illinois**: Chicago, IL
- **Colorado**: Denver, CO
- **Oregon**: Portland, OR
- **Michigan**: Detroit, MI
- **Georgia**: Atlanta, GA
- **Arizona**: Phoenix, AZ

## 🚨 Troubleshooting

### Common Issues

1. **"No data found in index"**
   - Run `company_embed.py` first to populate the database

2. **API Key Errors**
   - Ensure your `.env` file has correct API keys
   - Check that keys are valid and have sufficient credits

3. **Rate Limit Errors**
   - The system includes built-in delays to avoid rate limits
   - If you still hit limits, increase the delay in `company_embed.py`

4. **Pinecone Region Errors**
   - Free plan only supports `us-east-1` region
   - Paid plans can use other regions

## 📈 Performance Notes

- **Embedding Generation**: ~0.15 seconds per company (with rate limiting)
- **Search Response**: ~1-2 seconds per query
- **Index Size**: ~100 companies = ~300KB of vector data
- **Memory Usage**: Minimal (vectors stored in Pinecone cloud)

## 🔄 Workflow

1. **Initial Setup**: Run `company_embed.py` once to populate the database
2. **Regular Usage**: Use `get_top_k_companies()` for programmatic access
3. **Interactive Search**: Run `company_search.py` for interactive interface
4. **Data Updates**: Re-run `company_embed.py` when you have new company data
5. **Testing**: Use `test_company_search.py` to verify functionality

## 💡 Usage Examples

### For Web Applications
```python
from company_search import get_top_k_companies

def search_companies_api(query, industry=None, location=None, min_revenue=None, min_employees=None):
    filters = {}
    if industry:
        filters['industry_filter'] = {"$in": [industry]}
    if location:
        filters['location_filter'] = {"$in": [location]}
    if min_revenue:
        filters['revenue_filter'] = {"$gte": min_revenue}
    if min_employees:
        filters['employees_filter'] = {"$gte": min_employees}
    
    return get_top_k_companies(query, top_k=10, **filters)
```

### For Data Analysis
```python
from company_search import get_top_k_companies

# Get all EdTech companies
edtech_results = get_top_k_companies("edtech companies", top_k=50)

# Analyze revenue distribution
revenues = [company['revenue'] for company in edtech_results['companies']]
print(f"EdTech companies found: {len(edtech_results['companies'])}")

# Get companies by location
california_results = get_top_k_companies(
    "technology companies", 
    top_k=50,
    location_filter={"$in": ["California", "CA"]}
)
print(f"California tech companies: {len(california_results['companies'])}")
```

### For Investment Analysis
```python
from company_search import get_top_k_companies_with_reasoning

# Find high-revenue companies with specific characteristics
results = get_top_k_companies_with_reasoning(
    "successful companies with digital transformation focus",
    top_k=10,
    revenue_filter={"$gte": "$100M"},
    employees_filter={"$gte": 200}
)

for company in results['companies']:
    print(f"{company['company_name']}: {company['reasoning']}")
```

## 🔍 Advanced Search Examples

### Healthcare Companies
```python
results = get_top_k_companies(
    "healthcare companies with FDA compliance",
    industry_filter={"$in": ["Medical Equipment Distribution", "Nutraceutical Manufacturing"]}
)
```

### Food Industry Companies
```python
results = get_top_k_companies(
    "organic food companies with supply chain optimization",
    industry_filter={"$in": ["Organic Packaged Foods"]}
)
```

### Logistics Companies
```python
results = get_top_k_companies(
    "logistics companies with route optimization",
    industry_filter={"$in": ["Freight & Logistics"]}
)
```

### Technology Companies in California
```python
results = get_top_k_companies(
    "technology companies",
    location_filter={"$in": ["California", "CA"]},
    employees_filter={"$gte": 200}
)
```

This specialized company RAG system provides powerful search and analysis capabilities for company data, with comprehensive filtering options and structured results!
