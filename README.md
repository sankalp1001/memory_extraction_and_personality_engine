## Module 1: Memory Extractor (`memory_extractor.py`)

### Design Choices

#### 1. Chunked Extraction Strategy

**Assumption:** The 30 messages represent a conversation between user and agent (15 from each side).

**Chunked extraction** was chosen over alternatives:

| Approach | Pros | Cons |
|----------|------|------|
| **Fine-grained (per-message)** | Maximum detail | Noisy, overfits transient states, expensive |
| **One-shot (full conversation)** | Simple | Flattens temporal structure, loses context, no incremental updates |
| **Chunked (implemented)** | Preserves time, evidence-backed, scalable | Slightly more complex |

**Implementation:** 10-turn chunks (configurable via `chunk_size`)

```python
extractor = MemoryExtractor(chunk_size=10)
```

#### 2. Memory Type 

Three distinct memory types are extracted:

| Type | Description | Example |
|------|-------------|---------|
| `preference` | Stable likes/dislikes that shape interaction | "prefers running over gym" |
| `emotional_pattern` | Recurring emotional responses or causal relationships | "sleep disruption leads to anxiety" |
| `long_term_fact` | Biographical details, life circumstances | "software engineer", "lives alone" |

**Rationale:**
- **Preferences** inform how to interact (tone, topics, approach)
- **Emotional patterns** enable proactive support and trigger awareness
- **Facts** provide grounding context for personalization

#### 3. Confidence Scoring

Each memory includes a confidence score (0.0–1.0):

| Score Range | Meaning | Example |
|-------------|---------|---------|
| 0.9–1.0 | Explicit, direct statement | "I prefer Python" |
| 0.7–0.8 | Strongly implied, clear from context | Repeated behavior patterns |
| 0.5–0.6 | Inferred from indirect statement | Implied preferences |
| < 0.5 | Uncertain, speculative | Not extracted |

#### 4. Aggregation & Deduplication

Memories are deduplicated by `(type, key)` with the following rules:
- **Highest confidence wins** — if the same memory appears in multiple chunks, the most confident version is kept
- **Evidence merged** — all supporting turn numbers are combined
- **Source chunks tracked** — provenance is maintained for debugging

### Architecture

```
Conversation → Chunking → LLM Extraction → Aggregation → File Output
     │              │            │              │              │
  30 turns    10-turn chunks   Per-chunk     Dedupe &      user_memory.json
                               JSON output   bucket by type
```

### Structured Output Parsing

The LLM is prompted to return structured JSON with a defined schema. The response is parsed using `json.loads()` with error handling for malformed outputs:

```python
try:
    return json.loads(raw)
except json.JSONDecodeError:
    raise ValueError(f"Invalid JSON returned for chunk {chunk_id}")
```

### Prompt Engineering

**System Prompt Design:**
1. Clear role definition ("memory extraction system for companion AI")
2. Explicit extraction criteria (what TO and what NOT TO extract)
3. Step-by-step reasoning process
4. Confidence scoring rubric
5. Output format specification

**User Prompt Design:**
1. Structured 4-step extraction process
2. Concrete examples with expected outputs
3. Strict JSON-only output requirement

### Usage

```python
from memory_extractor import MemoryExtractor

extractor = MemoryExtractor(
    model="openai/gpt-oss-20b",
    chunk_size=10,
    temperature=0.0  # Deterministic for consistency
)

# Extract and save
memories = extractor.extract(
    conversation=conversation_data,
    output_path="user_memory.json"
)
```

### Output Format

```json
{
  "preferences": [
    {
      "id": "mem_1",
      "type": "preference",
      "key": "exercise_preference",
      "value": "prefers running over gym workouts",
      "confidence": 0.95,
      "evidence": {
        "quote": "I prefer running over going to the gym",
        "turns": [11]
      },
      "source_chunks": [2]
    }
  ],
  "emotional_patterns": [...],
  "long_term_facts": [...]
}
```

---

## Module 2: Personality Engine (`personality_engine.py`)

The Personality Engine transforms generic agent responses into personality-specific responses by designing system prompts based on predefined personality configurations.

### Available Personalities

Four personalities are defined, each suited for different user needs:

| Personality | Purpose | Key Traits |
|-------------|---------|------------|
| `supportive_listener` | Emotional validation without guidance | Calm, empathetic, no advice, minimal questions |
| `cbt_reflective_guide` | Gentle cognitive reflection | Structured, reflects thought patterns, one open question |
| `grounding_presence` | De-escalation and grounding | Very brief, present-focused, no analysis |
| `practical_mentor` | Action-oriented support | Concise, offers one small concrete step |

### Personality Parameters

Each personality is defined by structured parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `warmth` | 0.0–1.0 | Emotional warmth (clinical ↔ warm) |
| `directness` | 0.0–1.0 | Communication style (gentle ↔ direct) |
| `verbosity` | 0.0–1.0 | Response length (brief ↔ elaborate) |
| `gives_advice` | bool | Whether to offer suggestions |
| `asks_questions` | str | "none", "minimal", or "one_open" |
| `focus` | str | "emotions", "thoughts", "present", or "action" |

### Usage

```python
from personality_engine import PersonalityEngine

engine = PersonalityEngine()

# Build prompt for a specific personality
prompt = engine.build_prompt(personality="supportive_listener")

# Use with base agent
response = agent.respond(user_message, system_prompt=prompt)
```

### Before/After Demonstration

The demo shows how the same user message receives different responses:

**User:** "I had a really stressful day today."

| Personality | Response Style |
|-------------|----------------|
| **Neutral (before)** | Generic, may offer unsolicited advice |
| **Supportive Listener** | Validates feelings, invites sharing |
| **Grounding Presence** | Very brief, present-focused, calming |
| **Practical Mentor** | One concrete action step |

---

## Design Assumptions

1. **30 messages = 15 user + 15 assistant turns** — alternating conversation format assumed
2. **Conversations are coherent** — messages belong to a single session/context
3. **User messages contain extractable information** — not all conversations yield memories
4. **Memories are stable** — extracted preferences persist across sessions
5. **LLM returns valid JSON** — error handling for malformed responses is included
6. **Single user per conversation** — no multi-user scenarios

### Extension: Memory-Based Personality Selection

**Note:** The assignment requires a memory extraction module and a separate personality engine. The connection between them (automatically selecting personality based on extracted memory) is implemented as an **optional extension** to demonstrate how user memory can inform agent behavior. This is not explicitly required by the assignment but showcases "working with user memory" in a practical way.

The `select_personality(memory)` method and memory context injection can be used when tighter integration is desired:

```python
# Optional: Auto-select personality based on memory
memory = engine.load_memory("user_memory.json")
selected = engine.select_personality(memory)
prompt = engine.build_prompt(memory=memory)
```

---

## Project Structure

```
memory_extraction_and_persona_build/
├── memory_extractor.py     # Memory extraction with aggregation
├── personality_engine.py   # Personality prompt design + demo
├── base_agent.py           # Base LLM agent
├── data/
│   └── sample_conversation.json
    ├── user_memory.json (output)  # Extracted memories
└── README.md
```

---

## Running the System

```bash
# Step 1: Extract memories from sample conversation
python memory_extractor.py

# Step 2: Run personality demo (before/after comparison)
python personality_engine.py
```

---

