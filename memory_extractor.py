from groq import Groq
from dotenv import load_dotenv
import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass

load_dotenv()

@dataclass
class Turn:
    turn: int
    role: str
    content: str

class MemoryExtractor:
    """
    MemoryExtractor identifies structured, long-term user memory from conversation
    transcripts for use in a companion AI system.

    The module accepts a full conversation, internally segments it into coherent
    chunks, and extracts memory candidates from each chunk using a language model.
    """
    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        chunk_size: int = 10,
        temperature: float = 0.0
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found in environment")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        self.chunk_size = chunk_size
        self.temperature = temperature

    def extract(
        self,
        conversation: List[Dict[str, Any]],
        output_path: str = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract memory candidates from a full conversation.

        Input:
            conversation: A list of conversation turns in chronological order.
                Each turn must contain:
                - turn (int): monotonically increasing turn number
                - role (str): "user" or "assistant"
                - content (str): raw message text
            output_path: If provided, saves aggregated memories to this JSON file.

        Output:
            Dict with keys: preferences, emotional_patterns, long_term_facts
            Each containing a list of deduplicated memory objects.
        """
        turns = [Turn(**t) for t in conversation]
        chunks = self._chunk_conversation(turns)

        all_candidates = []

        for chunk_id, chunk_turns in enumerate(chunks, start=1):
            chunk_text = self._format_chunk(chunk_turns)
            response = self._call_llm(chunk_text, chunk_id)
            all_candidates.extend(response["memory_candidates"])

        return self.aggregate_memories(all_candidates, output_path)

    def aggregate_memories(
        self,
        candidates: List[Dict[str, Any]],
        output_path: str = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Deduplicate, merge, and bucket memory candidates by type.

        Rules:
        - Group candidates by (type, key)
        - Keep the highest-confidence value for each group
        - Merge all turn numbers from evidence across duplicates
        - Assign unique IDs to each memory
        - Bucket into preferences, emotional_patterns, long_term_facts

        Input:
            candidates: Raw list of memory candidates (may contain duplicates)
            output_path: If provided, saves the result to this JSON file

        Output:
            Dict with keys: preferences, emotional_patterns, long_term_facts
        """
        from collections import defaultdict

        grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

        for mem in candidates:
            group_key = (mem["type"], mem["key"])
            grouped[group_key].append(mem)

        # Buckets for each memory type
        buckets = {
            "preferences": [],
            "emotional_patterns": [],
            "long_term_facts": []
        }
        
        id_counter = 1

        for (mem_type, key), mems in grouped.items():
            # Find the memory with highest confidence
            best = max(mems, key=lambda m: m.get("confidence", 0.0))

            # Merge all turns from all duplicates
            all_turns = set()
            all_chunks = set()
            for m in mems:
                all_turns.update(m.get("evidence", {}).get("turns", []))
                chunk = m.get("source_chunk")
                if chunk is not None:
                    all_chunks.add(chunk)

            # Build the aggregated memory with ID
            result = {
                "id": f"mem_{id_counter}",
                "type": mem_type,
                "key": key,
                "value": best["value"],
                "confidence": best.get("confidence", 0.0),
                "evidence": {
                    "quote": best.get("evidence", {}).get("quote", ""),
                    "turns": sorted(all_turns)
                },
                "source_chunks": sorted(all_chunks)
            }
            id_counter += 1

            # Add to appropriate bucket
            if mem_type == "preference":
                buckets["preferences"].append(result)
            elif mem_type == "emotional_pattern":
                buckets["emotional_patterns"].append(result)
            elif mem_type == "long_term_fact":
                buckets["long_term_facts"].append(result)

        # Save to file if path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(buckets, f, indent=2)
            print(f"Memories saved to {output_path}")

        return buckets

    def _chunk_conversation(self, turns: List[Turn]) -> List[List[Turn]]:
        """
        Split a conversation into non-overlapping chunks.

        Input:
            turns: List of Turn objects in chronological order.

        Output:
            A list of chunks, where each chunk is a list of Turn objects.
            Chunk size is determined by self.chunk_size.

        """
        return [
            turns[i : i + self.chunk_size]
            for i in range(0, len(turns), self.chunk_size)
        ]

    def _format_chunk(self, turns: List[Turn]) -> str:
        """
        Convert a chunk of turns into a deterministic text representation
        suitable for LLM consumption.

        Input:
            turns: List of Turn objects belonging to a single chunk.

        Output:
            A formatted string preserving:
                - turn numbers
                - speaker roles
                - original message content

        """
        lines = []
        for t in turns:
            lines.append(f"Turn {t.turn} ({t.role}): {t.content}")
        return "\n".join(lines)

    def _call_llm(self, chunk_text: str, chunk_id: int) -> Dict[str, Any]:
        """
        Invoke the language model to extract memory candidates from a chunk.

        Input:
            chunk_text: Formatted conversation chunk as a string.
            chunk_id: Integer identifier for the chunk.

        Output:
            A parsed JSON object containing:
                - memory_candidates: list of extracted memory items
        """

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": self._user_prompt(chunk_text, chunk_id),
                },
            ],
        )
        raw = completion.choices[0].message.content
        # print(raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(
                f"Invalid JSON returned for chunk {chunk_id}:\n{raw}"
            )

    def _system_prompt(self) -> str:
        return """You are a memory extraction system for a companion AI.

Your task is to identify information worth remembering long-term to personalize future conversations.

WHAT TO EXTRACT:
1. Preferences - stable likes/dislikes/tendencies that shape how the user wants to interact
2. Emotional Patterns - recurring emotional responses or causal relationships (X leads to Y)
3. Long-term Facts - biographical details, life circumstances, or stable truths about the user

WHAT NOT TO EXTRACT:
- Transient emotions ("I'm sad today") unless they're part of a pattern
- Temporary situations that will quickly become outdated
- Information already implicit in the conversation format itself
- Vague or ambiguous statements without clear meaning

REASONING PROCESS:
For each potential memory:
1. Identify the claim explicitly stated by the user
2. Determine if it's stable/reusable or temporary/contextual
3. Classify the type (preference/pattern/fact)
4. Assess confidence based on clarity and directness of the statement
5. Extract the exact supporting quote and turn numbers

CONFIDENCE SCORING:
- 0.9-1.0: Explicit, direct statement ("I prefer X", "I am Y")
- 0.7-0.8: Strongly implied, clear from context
- 0.5-0.6: Inferred from behavior or indirect statement
- Below 0.5: Uncertain, ambiguous, or speculative

Output ONLY valid JSON. No explanations or commentary."""

    def _user_prompt(self, chunk_text: str, chunk_id: int) -> str:
        return f"""Conversation chunk (chunk_id={chunk_id}):

{chunk_text}

TASK: Extract memory candidates using the following reasoning process.

STEP 1 - IDENTIFY CANDIDATE STATEMENTS
Read through the conversation and identify any user statements that reveal:
- Preferences (likes, dislikes, tendencies, communication style)
- Emotional patterns (recurring feelings, cause-effect relationships)
- Long-term facts (job, living situation, habits, biographical info)

STEP 2 - EVALUATE EACH CANDIDATE
For each candidate, ask:
- Is this stable information that will remain relevant in future conversations?
- Is this explicitly stated or clearly implied?
- Can I extract this without inventing details?
- What exact quote supports this, and from which turn(s)?

STEP 3 - CLASSIFY AND STRUCTURE
Determine:
- Type: preference | emotional_pattern | long_term_fact
- Key: A short descriptive label (e.g., "exercise_preference", "sleep_anxiety_pattern")
- Value: Normalized statement of the memory (e.g., "prefers running over gym")
- Confidence: 0.0-1.0 based on explicitness and clarity
- Evidence: Direct quote and turn number(s)

STEP 4 - OUTPUT FORMAT
Return a JSON object with this exact structure:

{{
  "memory_candidates": [
    {{
      "type": "preference|emotional_pattern|long_term_fact",
      "key": "descriptive_key_name",
      "value": "normalized statement of the memory",
      "confidence": 0.0,
      "evidence": {{
        "quote": "exact quote from conversation",
        "turns": [turn_number]
      }},
      "source_chunk": {chunk_id}
    }}
  ]
}}

EXAMPLES:

User says: "I prefer running over going to the gym"
→ Type: preference
→ Key: exercise_preference
→ Value: prefers running over gym workouts
→ Confidence: 0.95 (explicit statement)
→ Quote: "I prefer running over going to the gym"

User says: "my sleep gets messed up and I feel anxious the next day"
→ Type: emotional_pattern
→ Key: sleep_anxiety_pattern
→ Value: sleep disruption leads to next-day anxiety
→ Confidence: 0.85 (clear causal relationship)
→ Quote: "my sleep gets messed up and I feel anxious the next day"

User says: "I'm a software engineer"
→ Type: long_term_fact
→ Key: occupation
→ Value: software engineer
→ Confidence: 1.0 (direct factual statement)
→ Quote: "I'm a software engineer"

Now extract memory candidates from the conversation chunk above.
Return ONLY the JSON output, no other text."""
    
if __name__ == "__main__":
    sample_convo_path = "data/sample_conversation.json"
    output_path = "data/user_memory.json"

    with open(sample_convo_path, "r") as f:
        data = json.load(f)
    total_conversation = data["conversation"]

    extractor = MemoryExtractor(
        model="openai/gpt-oss-20b",
        chunk_size=10,
        temperature=0.0
    )

    print("Running memory extraction...\n")

    memories = extractor.extract(total_conversation, output_path=output_path)

    total_count = (
        len(memories["preferences"]) +
        len(memories["emotional_patterns"]) +
        len(memories["long_term_facts"])
    )
    print(f"\nExtracted {total_count} unique memories:")
    print(f"  - Preferences: {len(memories['preferences'])}")
    print(f"  - Emotional patterns: {len(memories['emotional_patterns'])}")
    print(f"  - Long-term facts: {len(memories['long_term_facts'])}")