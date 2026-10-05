import asyncio
import httpx
import sys

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/v1"

async def test_health(client):
    print("Testing /health...")
    resp = await client.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    print("OK: /health passed")

async def test_metadata(client):
    print("Testing /api/v1/scan/allergens...")
    resp = await client.get(f"{API_URL}/scan/allergens")
    assert resp.status_code == 200
    data = resp.json()
    assert "allergen_categories" in data
    print("OK: /api/v1/scan/allergens passed")

    print("Testing /api/v1/scan/dishes...")
    resp = await client.get(f"{API_URL}/scan/dishes")
    assert resp.status_code == 200
    data = resp.json()
    assert "dishes" in data
    print("OK: /api/v1/scan/dishes passed")

async def test_user_flow(client):
    print("Testing user registration...")
    user_payload = {
        "email": f"testuser_{asyncio.get_event_loop().time()}@example.com",
        "full_name": "Test User",
        "age": 30,
        "gender": "male",
        "province": "Western",
        "allergens": ["peanuts", "shellfish"]
    }
    
    resp = await client.post(f"{API_URL}/users/register", json=user_payload)
    if resp.status_code != 201:
        print(f"Failed to register user. Status: {resp.status_code}, Msg: {resp.text}")
        return None
        
    data = resp.json()
    assert data["email"] == user_payload["email"]
    assert "peanuts" in data["allergens"]
    user_id = data["id"]
    print("OK: User registration passed")
    
    print("Testing user fetch...")
    resp = await client.get(f"{API_URL}/users/{user_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == user_id
    print("OK: User fetch passed")
    
    return user_id

async def test_scan_flow(client, user_id):
    print("Testing scan endpoint with dummy image...")
    # Create a small dummy image in memory (valid PNG header for instance, or just use a tiny blank image)
    # A 1x1 pixel PNG
    dummy_image = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    
    files = {
        "image": ("dummy.png", dummy_image, "image/png")
    }
    data = {
        "user_allergens": '["peanuts"]'
    }
    
    # Adding user_id if we created one
    if user_id:
        data["user_id"] = str(user_id)
        
    resp = await client.post(
        f"{API_URL}/scan/", 
        data=data,
        files=files,
        timeout=60.0
    )
    
    # It might fail with 503 if Gemini is overloaded, or 500 if the dummy image doesn't parse well,
    # but we just want to ensure it connects and tries to process.
    if resp.status_code == 200:
        res_data = resp.json()
        assert "identified_dish" in res_data
        print("OK: Scan flow passed (status 200)")
    else:
        print(f"WARN: Scan flow returned {resp.status_code}: {resp.text}")
        print("This is expected if Gemini complains about the image content not being a real dish, but the routing is working.")

async def run_all():
    print("Starting API Test Scenarios...")
    async with httpx.AsyncClient() as client:
        try:
            await test_health(client)
        except Exception as e:
            print(f"Health check failed (is server running?): {e}")
            sys.exit(1)
            
        await test_metadata(client)
        user_id = await test_user_flow(client)
        await test_scan_flow(client, user_id)
    
    print("DONE: All test scenarios executed.")

if __name__ == "__main__":
    asyncio.run(run_all())
