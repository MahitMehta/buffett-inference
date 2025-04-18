from dataclasses import dataclass
from flask import Flask, request

from main_pipeline_basic import BuffettInference, Post

@dataclass 
class Output:
    account_id: str
    analysis: str

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    instances = data.get('instances', [])
    if not instances:
        return {"outputs": []}, 400

    buffett_inference = BuffettInference()
    outputs = []

    for instance in instances:
        post = Post(instance["handle"], instance["content"])
        output = buffett_inference.call_agent(post)
        outputs.append(Output(instance["account_id"], output))
        

@app.route('/health', methods=['GET'])
def health():
    return {"status": "healthy"}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)