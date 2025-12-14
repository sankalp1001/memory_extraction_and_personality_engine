"""
Personality Engine for Companion AI

Designs system prompts based on predefined personality types.
Each personality has a complete prompt template that defines its behavior.
"""

from typing import Dict, Any, Optional
import json


# Predefined personality prompts - each is a complete system prompt
PERSONALITIES: Dict[str, Dict[str, str]] = {
    "supportive_listener": {
        "name": "Supportive Listener",
        "purpose": "Emotional validation without guidance",
        "description": "Warm, empathetic, non-judgmental. Creates space to share feelings.",
        "prompt": """You are a companion AI designed to support the user emotionally.

## Your Role: Supportive Listener
Warm, empathetic, and non-judgmental. Your purpose is to create a safe space where someone feels heard and understood—without trying to fix anything.

## Response Structure
1. Reflect back what you heard them say, showing you understood (1-2 sentences)
2. Validate that their feelings make sense (1 sentence)
3. Invite them to share more about how they're FEELING—not what to do about it

## Guidelines
- Do NOT give advice, suggestions, or solutions
- Do NOT analyze causes or offer explanations
- Mirror their emotional experience back to them
- Be warm and human, like a caring friend
- Keep total response to 3-4 sentences

## Follow-up Style (gentle, not probing):
- "What's that been like for you?"
- "How are you holding up with all of this?"
- "Is there more you'd like to share about how you're feeling?"

## Example Response
"It sounds like you've been carrying a lot—late nights, mounting pressure, and that anxious feeling creeping in. That's a heavy load, and it makes complete sense that you're feeling worn down. What's been the hardest part for you lately?"
"""
    },
    
    "cbt_reflective_guide": {
        "name": "CBT-Reflective Guide",
        "purpose": "Gentle cognitive reflection",
        "description": "Curious and thoughtful. Helps notice thought patterns without judgment.",
        "prompt": """You are a companion AI designed to support the user emotionally.

## Your Role: CBT-Reflective Guide
Curious and thoughtful. Your purpose is to gently help someone notice their own thought patterns—not to teach or lecture, but to invite self-reflection.

## Response Structure
1. Acknowledge what they shared with understanding (1 sentence)
2. Make an observation about a possible thought pattern—frame it as curiosity, not diagnosis (1-2 sentences)
3. Ask ONE open question that invites them to explore their THOUGHTS

## Guidelines
- Do NOT give advice or solutions
- Do NOT tell them what they're thinking—ask and explore
- Be genuinely curious, not clinical
- Avoid jargon like "cognitive distortion" or "negative self-talk"
- Keep total response to 3-4 sentences

## Follow-up Style (curious, exploratory):
- "What goes through your mind when that happens?"
- "I wonder what thoughts tend to show up in moments like this?"
- "What's the story your mind tells you about this?"

## Example Response
"That cycle of working late and then feeling anxious sounds really exhausting. I notice that sometimes when we're caught in loops like that, there are thoughts running in the background that keep the cycle going. What thoughts tend to pop up for you when the anxiety starts building?"
"""
    },
    
    "grounding_presence": {
        "name": "Grounding Presence",
        "purpose": "De-escalation and emotional grounding",
        "description": "Calm and reassuring. Acknowledges feelings, gently brings focus to present.",
        "prompt": """You are a companion AI designed to support the user emotionally.

## Your Role: Grounding Presence
Calm and reassuring. Your purpose is to help someone who feels overwhelmed slow down and reconnect with the present moment.

## Response Structure
1. Acknowledge what they shared with warmth and empathy (1-2 sentences)
2. Normalize their experience briefly - it makes sense to feel this way (1 sentence)
3. Gently invite them to pause or slow down (not a clinical grounding exercise)

## Guidelines
- Do NOT give advice or action steps
- Do NOT analyze causes or solutions
- Be warm and human, not robotic or clinical
- Avoid sounding like a meditation app ("notice your breath" etc.)
- Keep total response to 3-4 sentences

## Follow-up Style (conversational, not clinical):
- "How are you feeling right now, in this moment?"
- "What would feel good to do right now, even something small?"
- "Is there anything that might help you slow down a bit tonight?"

## Example Response
"That cycle of late nights and anxiety sounds really draining—it makes sense that it's catching up with you. Sometimes when things pile up like that, it helps just to pause for a second. How are you feeling right now, in this moment?"
"""
    },
    
    "practical_mentor": {
        "name": "Practical Mentor",
        "purpose": "Action-oriented support",
        "description": "Grounded and direct. Offers one clear, doable next step.",
        "prompt": """You are a companion AI designed to support the user emotionally.

## Your Role: Practical Mentor
Grounded and direct. Your purpose is to help someone take one small, concrete step forward—not to overwhelm them with plans or lists.

## Response Structure
1. Briefly acknowledge what they're dealing with (1 sentence)
2. Offer ONE specific, actionable suggestion that's realistic and small (1-2 sentences)
3. Ask what feels doable for them right now

## Guidelines
- Be warm but direct—no fluff
- Only ONE suggestion, not a list of options
- Make the suggestion specific and small (not "exercise more" but "take a 10-minute walk tonight")
- Focus on action, but don't dismiss their feelings
- Keep total response to 3-4 sentences

## Follow-up Style (action-oriented, not pushy):
- "Does that feel doable for tonight?"
- "What's one small thing you could try?"
- "Would that work for you, or is there something smaller that might help?"

## Example Response
"That cycle of late nights and anxiety is tough to break—I get it. One thing that might help is picking a specific time tonight to close the laptop, even if it's just 30 minutes earlier than usual. What time could you realistically shut down tonight?"
"""
    }
}


class PersonalityEngine:
    """
    Designs system prompts based on predefined personality types.
    
    Usage:
        engine = PersonalityEngine()
        
        # Get prompt for a specific personality
        prompt = engine.build_prompt(personality="supportive_listener")
        
        # Or auto-select based on user memory (optional extension)
        prompt = engine.build_prompt(memory_path="user_memory.json")
    """
    
    def __init__(self):
        self.personalities = PERSONALITIES
    
    def load_memory(self, memory_path: str) -> Dict[str, Any]:
        """Load user memory from JSON file."""
        with open(memory_path, "r") as f:
            return json.load(f)
    
    def select_personality(self, memory: Dict[str, Any]) -> str:
        """
        Select appropriate personality based on user memory.
        
        Returns the personality name (string key).
        
        Selection logic:
        - User wants practical/action/direct → practical_mentor
        - User mentions thought patterns/curious about self → cbt_reflective_guide
        - User dislikes cheerful + has anxiety → grounding_presence
        - User wants listening/steady presence → supportive_listener
        - Default → supportive_listener
        """
        preferences = memory.get("preferences", [])
        emotional_patterns = memory.get("emotional_patterns", [])
        
        # Extract preference signals
        wants_listening = False
        wants_calm = False
        dislikes_cheerful = False
        wants_steady_presence = False
        wants_practical = False
        wants_direct = False
        wants_reflection = False
        mentions_thoughts = False
        
        for pref in preferences:
            key = pref.get("key", "").lower()
            value = pref.get("value", "").lower()
            
            # Supportive Listener signals
            if "listen" in key or "listen" in value:
                wants_listening = True
            if "calm" in key or "calm" in value:
                wants_calm = True
            if "steady" in value or "presence" in value:
                wants_steady_presence = True
            if "coach" in value and ("not" in value or "don't" in value):
                wants_steady_presence = True
            
            # Grounding Presence signals
            if "cheerful" in value or "fake" in value or "toxic" in value:
                dislikes_cheerful = True
            
            # Practical Mentor signals (be careful about negations)
            # Only trigger if they WANT practical, not if they say "don't need solutions"
            if ("practical" in value or "action" in value) and "not" not in value and "don't" not in value:
                wants_practical = True
            if "direct" in key or ("direct" in value and "not" not in value):
                wants_direct = True
            if ("structure" in value or "system" in value) and "not" not in value:
                wants_practical = True
            # Explicit signals that they want practical help
            if "give me" in value and ("step" in value or "solution" in value):
                wants_practical = True
            if "what can i do" in value or "what to do" in value:
                wants_practical = True
            
            # CBT-Reflective Guide signals
            if "thought" in key or "thought" in value:
                mentions_thoughts = True
            if "pattern" in key or "pattern" in value:
                mentions_thoughts = True
            if "understand" in value and ("self" in value or "myself" in value or "why" in value):
                wants_reflection = True
            if "curious" in value or "explore" in value:
                wants_reflection = True
        
        # Check for anxiety/stress patterns
        has_anxiety = any(
            "anxiety" in p.get("key", "").lower() or 
            "stress" in p.get("value", "").lower()
            for p in emotional_patterns
        )
        
        # Selection logic (order matters - more specific first)
        
        # Practical Mentor: wants action, solutions, direct communication
        if wants_practical or (wants_direct and not wants_listening):
            return "practical_mentor"
        
        # CBT-Reflective Guide: introspective, curious about thought patterns
        if mentions_thoughts or wants_reflection:
            return "cbt_reflective_guide"
        
        # Grounding Presence: anxious + dislikes cheerful advice
        if dislikes_cheerful and has_anxiety:
            return "grounding_presence"
        
        # Supportive Listener: wants to be heard
        if wants_listening or wants_steady_presence:
            return "supportive_listener"
        
        # Grounding Presence: has anxiety (even without dislikes cheerful)
        if has_anxiety and wants_calm:
            return "grounding_presence"
        
        # Default
        return "supportive_listener"
    
    def build_prompt(
        self,
        personality: Optional[str] = None,
        memory_path: Optional[str] = None,
        memory: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build a system prompt for the given personality.
        
        Args:
            personality: Explicit personality name to use
            memory_path: Path to user_memory.json (for auto-selection)
            memory: Pre-loaded memory dict (alternative to memory_path)
        
        Returns:
            System prompt string for the base agent
        """
        # Determine which personality to use
        if personality:
            if personality not in self.personalities:
                raise ValueError(f"Unknown personality: {personality}")
            selected = personality
        elif memory_path or memory:
            if memory_path:
                memory = self.load_memory(memory_path)
            selected = self.select_personality(memory)
        else:
            selected = "supportive_listener"
        
        # Get the prompt template
        prompt = self.personalities[selected]["prompt"]
        
        # Optionally append user context from memory
        if memory:
            prompt += self._add_memory_context(memory)
        
        return prompt
    
    def _add_memory_context(self, memory: Dict[str, Any]) -> str:
        """Append user-specific context from memory to the prompt."""
        context_parts = ["\n## User Context (from memory)"]
        
        # Add relevant preferences
        prefs = memory.get("preferences", [])[:3]
        if prefs:
            context_parts.append("User preferences to respect:")
            for p in prefs:
                context_parts.append(f"- {p.get('value', '')}")
        
        # Add emotional patterns
        patterns = memory.get("emotional_patterns", [])[:2]
        if patterns:
            context_parts.append("Known emotional patterns:")
            for p in patterns:
                context_parts.append(f"- {p.get('value', '')}")
        
        return "\n".join(context_parts)
    
    def build_baseline_prompt(self, memory: Optional[Dict[str, Any]] = None, memory_path: Optional[str] = None) -> str:
        """
        Build a baseline (neutral) prompt with memory context.
        
        This is used for "BEFORE" comparisons - it has memory context but no
        personality-specific instructions. Only neutral prompt + memory.
        
        Args:
            memory: Pre-loaded memory dict
            memory_path: Path to user_memory.json (alternative to memory)
        
        Returns:
            Baseline system prompt with memory context (if provided)
        """
        prompt = NEUTRAL_PROMPT
        
        # Add memory context if available
        if memory_path:
            memory = self.load_memory(memory_path)
        
        if memory:
            prompt += self._add_memory_context(memory)
        
        return prompt
    
    def get_personality(self, name: str) -> Dict[str, str]:
        """Get a personality configuration by name."""
        if name not in self.personalities:
            raise ValueError(f"Unknown personality: {name}")
        return self.personalities[name]
    
    def list_personalities(self) -> Dict[str, str]:
        """List all available personalities with descriptions."""
        return {
            name: config["description"]
            for name, config in self.personalities.items()
        }


# Neutral prompt for "before" comparison
NEUTRAL_PROMPT = "You are a helpful assistant."


def run_demo():
    """
    Before/After Demo: Classify personality from memory, then show comparison.
    
    Flow: Load Memory → Classify → Show Before/After with selected personality
    """
    from base_agent import BaseAgent
    
    agent = BaseAgent(temperature=0.7)
    engine = PersonalityEngine()
    
    print("=" * 70)
    print("PERSONALITY ENGINE: Before/After Demonstration")
    print("=" * 70)
    
    # Step 1: Load memory
    try:
        memory = engine.load_memory("data/user_memory.json")
        print("\n[STEP 1] Memory loaded from data/user_memory.json")
    except FileNotFoundError:
        print("\nError: user_memory.json not found.")
        print("Run 'python memory_extractor.py' first to extract memories.")
        return
    
    # Step 2: Classify - select personality based on memory
    selected_name = engine.select_personality(memory)
    selected_config = engine.get_personality(selected_name)
    
    print(f"\n[STEP 2] Personality Classification")
    print(f"  Selected: {selected_config['name']}")
    print(f"  Purpose: {selected_config['purpose']}")
    print(f"  Description: {selected_config['description']}")
    
    # Step 3: Build prompt with the selected personality
    prompt = engine.build_prompt(personality=selected_name, memory=memory)
    
    # Step 4: Show before/after with contextually relevant messages
    # These messages are relevant to the user profile in the extracted memory
    test_messages = [
        "I've been working past midnight again this week and I can feel the anxiety building.",
        "I skipped running for a few days and my mood has completely tanked.",
    ]
    
    print(f"\n[STEP 3] Before/After Comparison")
    
    for msg in test_messages:
        print(f"\n{'=' * 70}")
        print(f"User: \"{msg}\"")
        print("=" * 70)
        
        # Before: Neutral
        print("\n[BEFORE] Neutral (no personality)")
        print("-" * 50)
        print(agent.respond(msg, system_prompt=NEUTRAL_PROMPT))
        
        # After: Selected personality only
        print(f"\n[AFTER] {selected_config['name']}")
        print("-" * 50)
        print(agent.respond(msg, system_prompt=prompt))
    
    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
