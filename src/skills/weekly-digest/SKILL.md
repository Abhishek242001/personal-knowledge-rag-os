# Weekly Digest Skill
## Surface forgotten ideas from the knowledge base

Every week, or when the user asks, surface interesting ideas they haven't
revisited recently from their knowledge base.

### Trigger phrases
- "Weekly digest"
- "What have I forgotten?"
- "Surface forgotten ideas"
- "What haven't I revisited?"
- "Give me my digest"
- Every Monday morning (scheduled via OpenClaw heartbeat)

### How to use

```bash
cd ~/personal-knowledge-rag-os
python src/rag/rag_query.py --digest
```

### Response format
Present the 5 ideas in a friendly, engaging way. For each idea:
1. The idea/concept title
2. Which source it came from
3. Why it might be worth revisiting
4. A suggested action (read more, expand on it, connect it to current work)

### Scheduling
To run this automatically every Monday, add to OpenClaw's HEARTBEAT.md:
```
Every Monday at 9am IST, run the weekly digest and send to Telegram.
```
