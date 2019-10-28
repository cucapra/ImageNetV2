import json
with open("imagenetv2-a-44.json") as json_file:
    data = json_file.read().replace('}{','}\n{')
for i,l in enumerate(data.split('\n')) :
    data = json.loads(l)
print(i+1)
