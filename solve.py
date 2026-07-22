import json, re, math

base = "/Users/amanchandrah/Documents/TDS W2/Week 4/semantic_cache_query_augmentation"
rules = json.load(open(f"{base}/cache_rules.json"))
expansion_map = json.load(open(f"{base}/expansion_map.json"))
requests = json.load(open(f"{base}/requests.json"))
entries = [json.loads(l) for l in open(f"{base}/cache_entries.jsonl")]

ttl = rules["ttl_minutes"]
threshold = rules["similarity_threshold"]

TOKEN_RE = re.compile(r"\b[a-z0-9]+\b")

def tokenize(s):
    return TOKEN_RE.findall(s.lower())

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

result = {}
for r in requests:
    tokens = tokenize(r["query"])
    added = set()
    for t in tokens:
        if t in expansion_map:
            for term in expansion_map[t]:
                added.add(term)
    added_terms = sorted(added)

    candidates = []
    for e in entries:
        if e["tenant"] != r["tenant"]:
            continue
        if e["channel"] != r["channel"]:
            continue
        if e["language"] != r["language"]:
            continue
        age = r["at_minute"] - e["created_minute"]
        if age < 0 or age > ttl:
            continue
        candidates.append(e)

    best_sim = -1.0
    best_id = None
    for e in candidates:
        sim = cosine(r["embedding"], e["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_id = e["cache_id"]

    if best_id is not None and best_sim >= threshold:
        decision = "HIT"
        cache_id = best_id
        nearest = best_sim
    else:
        decision = "MISS"
        cache_id = None
        nearest = best_sim if best_id is not None else 0.0

    result[r["request_id"]] = {
        "decision": decision,
        "cache_id": cache_id,
        "nearest_similarity": round(nearest, 4),
        "added_terms": added_terms,
    }

with open(f"{base}/answer.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(dict(list(result.items())[:3]), indent=2))
print("total:", len(result))
print("hits:", sum(1 for v in result.values() if v["decision"]=="HIT"))
