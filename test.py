import requests

data = {
    "url": 'https://plus.unsplash.com/premium_photo-1663952767362-e95f11d98acd?q=80&w=1550&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'

}

url = "http://localhost:8080/2015-03-31/functions/function/invocations"

result = requests.post(url, json=data).json() ## get the server response

print(result)