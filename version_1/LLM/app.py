import requests

url = "https://d6b3-34-133-156-153.ngrok-free.app/api/query"
data = {"query": "What is Endocarditis?"}

response = requests.post(url, json=data)
response_json = response.json()

print(response_json.get("answer"))

# curl -X POST https://d6b3-34-133-156-153.ngrok-free.app/api/query -H "Content-Type: application/json" -d '{"query": "What is Endocarditis?"}'