from together import Together
from config_loader import load_api_key

# Load API Key
TOGETHER_API_KEY = load_api_key()
client = Together(api_key=TOGETHER_API_KEY)

def test_together_api():
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        try:
            response = client.completions.create(
                model="mistralai/Mixtral-8x7B-v0.1",
                prompt=f"{query}",  # ✅ Just use the raw query
                max_tokens=50,  # ✅ Allow enough space for a full response
                temperature=0.2  # ✅ Keep responses stable
            )
            print("\n--- API Response ---")
            print(response.choices[0].text.strip() if response.choices else "[No response received]")
        except Exception as e:
            print(f"\nAPI Error: {e}")

if __name__ == "__main__":
    print(f"DEBUG: API Key Loaded = {TOGETHER_API_KEY[:5]}****** (Length: {len(TOGETHER_API_KEY)})")
    test_together_api()
