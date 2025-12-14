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

### Prompt Design

Each personality is defined by a complete system prompt with specific guidelines:

- **Role and purpose** — what the personality aims to achieve
- **Communication style** — tone, length, approach
- **Behavioral constraints** — what to do and what to avoid
- **Example responses** — concrete illustrations of the expected style

This approach was chosen over numeric parameters (warmth: 0.8, directness: 0.3) because direct prompt text is more expressive and easier to tune.

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

The Streamlit demo (`app.py`) demonstrates how personality instructions transform responses when combined with user memory.

#### Demo Process

1. **Select a Test User** - Choose from 4 test users, each representing a different personality type:
   - Supportive Listener - Wants to be heard, values emotional connection
   - CBT-Reflective Guide - Introspective, notices thought patterns
   - Grounding Presence - High anxiety, dislikes cheerful advice, needs calming presence
   - Practical Mentor - Action-oriented, wants concrete steps

2. **Memory Extraction** - The system extracts structured memory from the user's conversation:
   - **Preferences**: Communication style, likes/dislikes
   - **Emotional Patterns**: Recurring emotional responses
   - **Facts**: Biographical/contextual information

3. **Generate Two Responses** to the same test message:
   
   **BEFORE (Baseline)**: 
   - System Prompt: `"You are a helpful assistant."` + user memory context
   - The LLM has access to user memory but no special communication instructions
   - Response style: Generic, helpful, may give advice
   
   **AFTER (Personality-Adjusted)**: 
   - System Prompt: Personality-specific instructions + user memory context
   - The LLM follows detailed personality guidelines while respecting user memory
   - Response style: Tailored to personality (e.g., no advice, brief, specific tone)

#### Example Comparison

**User:** "I've been working past midnight again and the anxiety is building up."

| Response Type | Example Style |
|---------------|---------------|
| **BEFORE (Baseline + Memory)** | "It sounds like you're dealing with a lot. Working late can definitely contribute to anxiety. Have you considered setting a stricter bedtime routine? Creating boundaries around work hours might help..." |
| **AFTER (Grounding Presence + Memory)** | "That cycle of late nights and anxiety sounds really draining—it makes sense that it's catching up with you. Sometimes when things pile up like that, it helps just to pause for a second. How are you feeling right now, in this moment?" |

**Key Insight**: Both responses use the same user memory (e.g., "sleep disruption leads to anxiety", "prefers calm presence over advice"), but the personality instructions dramatically change how the AI responds to that information.

---

## Design Assumptions

1. **30 messages = 15 user + 15 assistant turns** — alternating conversation format assumed
2. **Conversations are coherent** — messages belong to a single session/context
3. **User messages contain extractable information** — not all conversations yield memories
4. **Memories are stable** — extracted preferences persist across sessions
5. **LLM returns valid JSON** — error handling for malformed responses is included
6. **Single user per conversation** — no multi-user scenarios

### Memory + Personality Integration

The personality engine integrates with extracted memory in two ways:

1. **Memory Context Injection**: User memory is automatically appended to any system prompt when memory is provided:
   ```python
   # Build prompt with personality + memory
   prompt = engine.build_prompt(
       personality="supportive_listener",
       memory=user_memory  # Memory context automatically added
   )
   ```

2. **Baseline Prompt with Memory**: For comparison purposes, a baseline prompt includes memory but no personality instructions:
   ```python
   # Baseline prompt: neutral + memory (no personality)
   baseline = engine.build_baseline_prompt(memory=user_memory)
   ```

3. **Automatic Personality Selection** (Optional): The engine can automatically select a personality based on extracted memory:
   ```python
   # Auto-select personality from memory
   memory = engine.load_memory("user_memory.json")
   selected = engine.select_personality(memory)
   prompt = engine.build_prompt(memory=memory)  # Uses selected personality
   ```

This demonstrates how user memory and personality instructions work together to create personalized responses.

---

## Project Structure

```
memory_extraction_and_persona_build/
├── memory_extractor.py     # Memory extraction with aggregation
├── personality_engine.py   # Personality prompt design + baseline prompt builder
├── base_agent.py           # Base LLM agent (Groq API wrapper)
├── app.py                  # Streamlit demo app
├── data/
│   ├── sample_conversation.json  # Sample conversation for extraction
│   ├── test_users.json           # Test users for demo (4 personalities)
│   └── user_memory.json (output) # Extracted memories
└── README.md
```

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| **LLM** | `openai/gpt-oss-20b` via Groq API |
| **Framework** | Python with Streamlit for UI |


---

## Running the System

### Option 1: Streamlit Demo (Recommended)

The interactive Streamlit app demonstrates the full system with test users:

```bash
# Run the interactive demo
streamlit run app.py
```

**What the demo shows:**
1. Test users with different personality types
2. Memory extraction from conversations
3. Before/After comparison showing how personality instructions transform responses
4. Both responses use the same memory, but different communication instructions

### Option 2: Command Line Scripts

```bash
# Step 1: Extract memories from sample conversation
python memory_extractor.py
# Output: data/user_memory.json

# Step 2: Run personality demo (before/after comparison)
python personality_engine.py
# Shows a single example with the extracted memory

# Step 3: Run Streamlit app (interactive demo)
streamlit run app.py
```

---

