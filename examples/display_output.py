import json

with open('./results.json') as f:
    data = json.load(f)

print(data['simple-orq-j44fn-3618796786']['result']['predictions'])