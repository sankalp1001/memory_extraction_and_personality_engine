"""
Test All Personalities

Extracts memories for different user types and tests that each
triggers the expected personality classification.
"""

import json
from memory_extractor import MemoryExtractor
from personality_engine import PersonalityEngine, NEUTRAL_PROMPT
from base_agent import BaseAgent


def run_test():
    """Test all personality types with different user profiles."""
    
    # Load test users
    with open("data/test_users.json", "r") as f:
        data = json.load(f)
    
    extractor = MemoryExtractor(chunk_size=10, temperature=0.0)
    engine = PersonalityEngine()
    agent = BaseAgent(temperature=0.7)
    
    print("=" * 70)
    print("TESTING ALL PERSONALITY TYPES")
    print("=" * 70)
    
    results = []
    
    for user in data["users"]:
        user_id = user["id"]
        expected = user["expected_personality"]
        description = user["description"]
        conversation = user["conversation"]
        test_message = user["test_message"]
        
        print(f"\n{'=' * 70}")
        print(f"USER: {user_id}")
        print(f"Description: {description}")
        print(f"Expected Personality: {expected}")
        print("=" * 70)
        
        # Step 1: Extract memories
        print("\n[1] Extracting memories...")
        try:
            memories = extractor.extract(conversation)
            pref_count = len(memories.get("preferences", []))
            pattern_count = len(memories.get("emotional_patterns", []))
            fact_count = len(memories.get("long_term_facts", []))
            print(f"    Extracted: {pref_count} preferences, {pattern_count} patterns, {fact_count} facts")
        except Exception as e:
            print(f"    Error extracting memories: {e}")
            continue
        
        # Step 2: Classify personality
        print("\n[2] Classifying personality...")
        selected = engine.select_personality(memories)
        selected_config = engine.get_personality(selected)
        
        match = "✓ MATCH" if selected == expected else "✗ MISMATCH"
        print(f"    Selected: {selected_config['name']} {match}")
        
        results.append({
            "user": user_id,
            "expected": expected,
            "selected": selected,
            "match": selected == expected
        })
        
        # Step 3: Generate response
        print(f"\n[3] Test Message: \"{test_message}\"")
        
        # Neutral response
        print("\n    [NEUTRAL]")
        neutral = agent.respond(test_message, system_prompt=NEUTRAL_PROMPT)
        # Truncate if too long
        if len(neutral) > 300:
            neutral = neutral[:300] + "..."
        print(f"    {neutral}")
        
        # Personality response
        prompt = engine.build_prompt(personality=selected, memory=memories)
        print(f"\n    [{selected_config['name'].upper()}]")
        styled = agent.respond(test_message, system_prompt=prompt)
        print(f"    {styled}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    matches = sum(1 for r in results if r["match"])
    total = len(results)
    
    print(f"\nClassification Accuracy: {matches}/{total}")
    print()
    
    for r in results:
        status = "✓" if r["match"] else "✗"
        print(f"  {status} {r['user']}: expected {r['expected']}, got {r['selected']}")
    
    print("\n" + "=" * 70)
    print("Test complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_test()

