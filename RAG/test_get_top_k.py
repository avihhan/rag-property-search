#!/usr/bin/env python3
"""
Test script for the get_top_k_properties_with_reasoning function
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from search import get_top_k_properties_with_reasoning, check_index_statistics

def test_get_top_k_properties_with_reasoning():
    """
    Test the get_top_k_properties_with_reasoning function with various scenarios
    """
    print("Testing get_top_k_properties_with_reasoning function...")
    
    # Check if index has data
    stats = check_index_statistics()
    if not stats or stats.get('total_vector_count', 0) == 0:
        print("❌ No data in index. Please run embed.py first.")
        return
    
    print(f"✅ Found {stats.get('total_vector_count', 0)} properties in index")
    
    # Test 1: Basic search with reasoning
    print("\n" + "="*60)
    print("TEST 1: Basic search for luxury properties with reasoning")
    print("="*60)
    
    results = get_top_k_properties_with_reasoning("luxury penthouse with ocean view", top_k=3)
    
    print(f"Query: {results['query']}")
    print(f"Top K: {results['top_k']}")
    print(f"Total Found: {results['total_found']}")
    print(f"Filters Applied: {results['filters_applied']}")
    print(f"Search Summary: {results['search_summary']}")
    
    if results['properties']:
        print("\nTop Properties with Reasoning:")
        for prop in results['properties']:
            print(f"  {prop['rank']}. {prop['property_name']}")
            print(f"     Location: {prop['location']}")
            print(f"     Price: ${prop['price_usd']:,}")
            print(f"     Bedrooms: {prop['bedrooms']}")
            print(f"     Score: {prop['score']}")
            print(f"     Reasoning: {prop['reasoning']}")
            print()
    else:
        print("No properties found.")
    
    # Test 2: Search with price filter and reasoning
    print("\n" + "="*60)
    print("TEST 2: Search with price filter (under $1M) and reasoning")
    print("="*60)
    
    results = get_top_k_properties_with_reasoning(
        "affordable properties", 
        top_k=3,
        price_filter={"$lte": 1000000}
    )
    
    print(f"Query: {results['query']}")
    print(f"Filters: {results['filters_applied']}")
    print(f"Total Found: {results['total_found']}")
    print(f"Search Summary: {results['search_summary']}")
    
    if results['properties']:
        print("\nAffordable Properties with Reasoning:")
        for prop in results['properties']:
            print(f"  {prop['rank']}. {prop['property_name']} - ${prop['price_usd']:,}")
            print(f"     Reasoning: {prop['reasoning']}")
    else:
        print("No affordable properties found.")
    
    # Test 3: Search with multiple filters and reasoning
    print("\n" + "="*60)
    print("TEST 3: Search with multiple filters and reasoning")
    print("="*60)
    
    results = get_top_k_properties_with_reasoning(
        "family homes with 3 bedrooms",
        top_k=3,
        price_filter={"$lte": 2000000},
        bedrooms_filter={"$gte": 3}
    )
    
    print(f"Query: {results['query']}")
    print(f"Filters: {results['filters_applied']}")
    print(f"Total Found: {results['total_found']}")
    print(f"Search Summary: {results['search_summary']}")
    
    if results['properties']:
        print("\nFamily Homes (3+ BR, under $2M) with Reasoning:")
        for prop in results['properties']:
            print(f"  {prop['rank']}. {prop['property_name']}")
            print(f"     {prop['bedrooms']} bedrooms, ${prop['price_usd']:,}")
            print(f"     Reasoning: {prop['reasoning']}")
    else:
        print("No family homes found matching criteria.")
    
    # Test 4: Location-based search with reasoning
    print("\n" + "="*60)
    print("TEST 4: Location-based search with reasoning")
    print("="*60)
    
    results = get_top_k_properties_with_reasoning(
        "properties in California",
        top_k=3
    )
    
    print(f"Query: {results['query']}")
    print(f"Total Found: {results['total_found']}")
    print(f"Search Summary: {results['search_summary']}")
    
    if results['properties']:
        print("\nCalifornia Properties with Reasoning:")
        for prop in results['properties']:
            print(f"  {prop['rank']}. {prop['property_name']} - {prop['location']}")
            print(f"     Reasoning: {prop['reasoning']}")
    else:
        print("No California properties found.")
    
    # Test 5: Return full property details with reasoning
    print("\n" + "="*60)
    print("TEST 5: Full property details with reasoning")
    print("="*60)
    
    results = get_top_k_properties_with_reasoning("luxury villa", top_k=1)
    
    if results['properties']:
        prop = results['properties'][0]
        print(f"Query: {results['query']}")
        print(f"Best Match:")
        print(f"  Name: {prop['property_name']}")
        print(f"  Type: {prop['property_type']}")
        print(f"  Location: {prop['location']}")
        print(f"  Bedrooms: {prop['bedrooms']}")
        print(f"  View: {prop['view']}")
        print(f"  Price: ${prop['price_usd']:,}")
        print(f"  Size: {prop['size_sqft']} sqft")
        print(f"  Score: {prop['score']}")
        print(f"  Reasoning: {prop['reasoning']}")
        print(f"  Description: {prop['description'][:100]}...")
    else:
        print("No luxury villa found.")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)

if __name__ == "__main__":
    test_get_top_k_properties_with_reasoning()
