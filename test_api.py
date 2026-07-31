"""
Test script for the Search Relevancy API.
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    """Test health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Health check passed: {data}")
    return data

def test_info():
    """Test service info endpoint."""
    print("\nTesting service info...")
    response = requests.get(f"{API_URL}/info")
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Service info: {json.dumps(data, indent=2)}")
    return data

def test_search(query: str = "climate change", num_results: int = 5):
    """Test search endpoint."""
    print(f"\nTesting search for query: '{query}'...")
    payload = {
        "query": query,
        "num_results": num_results
    }
    response = requests.post(f"{API_URL}/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Search returned {data['num_results']} results")
    
    # Print top result
    if data['results']:
        top = data['results'][0]
        print(f"  Top result: {top['title']} (score: {top['relevance_score']:.2%})")
    
    return data

def test_batch_search(queries: list = None):
    """Test batch search endpoint."""
    if queries is None:
        queries = ["artificial intelligence", "climate change", "renewable energy"]
    
    print(f"\nTesting batch search for {len(queries)} queries...")
    payload = {
        "queries": queries,
        "num_results": 3
    }
    response = requests.post(f"{API_URL}/search/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    print(f"✓ Batch search completed: {len(data['batch_results'])} results")
    
    return data

def test_error_handling():
    """Test error handling."""
    print("\nTesting error handling...")
    
    # Test missing query
    response = requests.post(f"{API_URL}/search", json={})
    assert response.status_code == 400
    print("✓ Empty query handled correctly")
    
    # Test short query
    response = requests.post(f"{API_URL}/search", json={"query": "a"})
    assert response.status_code == 400
    print("✓ Short query handled correctly")
    
    # Test invalid endpoint
    response = requests.get(f"{API_URL}/invalid")
    assert response.status_code == 404
    print("✓ Invalid endpoint handled correctly")

if __name__ == "__main__":
    try:
        print("=" * 50)
        print("Search Relevancy API - Test Suite")
        print("=" * 50)
        
        # Run tests
        test_health()
        test_info()
        test_search("climate change", 5)
        test_search("technology", 3)
        test_batch_search()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)
    
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API")
        print("  Make sure the Flask server is running at http://localhost:5000")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
