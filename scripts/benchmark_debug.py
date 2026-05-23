from bench.accuracy import bench_needle, _compress_messages
import json

samples = bench_needle('gpt-4o-mini', 1)
item = samples[0]

messages = []
messages.append({'role': 'system', 'content': f"Context:\n{item['context']}"})
messages.append({'role': 'user', 'content': item['question']})

compressed = _compress_messages(messages, 50000, query=item['question'])
print(f"Original sys length: {len(messages[0]['content'])}")
print(f"Compressed sys length: {len(compressed[0]['content'])}")
print(f"Compressed content:\n{compressed[0]['content']}")
