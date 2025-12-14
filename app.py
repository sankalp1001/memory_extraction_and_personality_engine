"""
Memory Extraction & Personality Engine - Demo App

A minimal Streamlit interface to demonstrate:
1. User memory (pre-extracted from conversation)
2. Personality classification based on memory
3. Before/After response comparison
"""

import streamlit as st
import json
from personality_engine import PersonalityEngine, NEUTRAL_PROMPT, PERSONALITIES
from base_agent import BaseAgent

# Page config
st.set_page_config(
    page_title="Personality Engine Demo",
    layout="centered"
)

# Initialize components
@st.cache_resource
def load_components():
    engine = PersonalityEngine()
    agent = BaseAgent(temperature=0.7)
    return engine, agent

@st.cache_data
def load_memory():
    with open("data/user_memory.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_sample_conversation():
    with open("data/sample_conversation.json", "r") as f:
        return json.load(f)

engine, agent = load_components()

# Title
st.title("Personality Engine")
st.markdown("*A companion AI that adapts its response style based on user memory*")

st.divider()

# --- MEMORY EXTRACTION SECTION ---
st.header("Extracted User Memory")

try:
    memory = load_memory()
    sample_convo = load_sample_conversation()
except FileNotFoundError:
    st.error("Memory file not found. Run `python memory_extractor.py` first.")
    st.stop()

# Show source
with st.expander("View Source: Sample Conversation (30 messages)"):
    st.markdown("*This memory was extracted from the following conversation:*")
    for turn in sample_convo.get("conversation", [])[:10]:
        role = "User" if turn["role"] == "user" else "Assistant"
        st.markdown(f"**{role}:** {turn['content']}")
    st.markdown("*... and 20 more messages*")
    st.markdown("[View full conversation on GitHub](https://github.com/sankalp1001/memory_extraction_and_personality_engine/blob/main/data/sample_conversation.json)")

# Show extracted memory
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Preferences")
    for pref in memory.get("preferences", [])[:4]:
        st.markdown(f"- {pref.get('value', '')}")

with col2:
    st.subheader("Emotional Patterns")
    for pattern in memory.get("emotional_patterns", []):
        st.markdown(f"- {pattern.get('value', '')}")

with col3:
    st.subheader("Facts")
    for fact in memory.get("long_term_facts", []):
        st.markdown(f"- {fact.get('value', '')}")

with st.expander("View Full Extracted Memory"):
    st.markdown("[View user_memory.json on GitHub](https://github.com/sankalp1001/memory_extraction_and_personality_engine/blob/main/data/user_memory.json)")
    st.markdown("*Preview (first few entries):*")
    # Show just a preview
    preview = {
        "preferences": memory.get("preferences", [])[:2],
        "emotional_patterns": memory.get("emotional_patterns", [])[:1],
        "long_term_facts": memory.get("long_term_facts", [])[:1]
    }
    st.json(preview)

st.divider()

# --- PERSONALITY CLASSIFICATION SECTION ---
st.header("Personality Classification")

# Classify
selected_name = engine.select_personality(memory)
selected_config = engine.get_personality(selected_name)

# Show result
st.success(f"**Selected Personality: {selected_config['name']}**")
st.markdown(f"_{selected_config['description']}_")

# Show reasoning
with st.expander("Why this personality?"):
    st.markdown("""
    **Classification logic based on extracted preferences:**
    
    The user's memory indicates:
    - Dislikes overly cheerful advice
    - Prefers someone to listen, not fix
    - Has anxiety patterns (sleep disruption leads to anxiety)
    - Wants a "steady presence, not a coach"
    
    **Result:** Grounding Presence is selected because it provides calm, 
    brief responses without advice—matching the user's stated preferences.
    """)

# Show all personalities
with st.expander("All Available Personalities"):
    for name, config in PERSONALITIES.items():
        selected_marker = " (selected)" if name == selected_name else ""
        st.markdown(f"**{config['name']}**{selected_marker}")
        st.markdown(f"_{config['description']}_")
        st.markdown("---")

st.divider()

# --- BEFORE/AFTER DEMO SECTION ---
st.header("Before/After Comparison")
st.markdown("*See how the personality transforms the response*")

# Personality selector
personality_options = list(PERSONALITIES.keys())
personality_labels = [f"{PERSONALITIES[p]['name']} {'(auto-selected)' if p == selected_name else ''}" for p in personality_options]

selected_personality = st.selectbox(
    "Select Personality to Test:",
    options=personality_options,
    format_func=lambda x: f"{PERSONALITIES[x]['name']} {'(auto-selected from memory)' if x == selected_name else ''}",
    index=personality_options.index(selected_name)
)

demo_config = engine.get_personality(selected_personality)
st.caption(f"*{demo_config['description']}*")

st.markdown("---")

# Example messages
example_messages = [
    "I've been working past midnight again and the anxiety is building up.",
    "I skipped running for a few days and my mood has tanked.",
    "I'm feeling overwhelmed and don't know what to do.",
]

# User input
user_message = st.text_area(
    "Enter a message:",
    value=example_messages[0],
    height=80
)

# Quick examples
st.markdown("**Try an example:**")
cols = st.columns(3)
for i, msg in enumerate(example_messages):
    if cols[i].button(f"Example {i+1}", key=f"ex_{i}"):
        st.session_state.user_message = msg
        st.rerun()

if st.button("Generate Responses", type="primary"):
    if not user_message.strip():
        st.warning("Please enter a message.")
    else:
        # Build prompt with selected personality
        prompt = engine.build_prompt(personality=selected_personality)
        
        # Generate responses
        with st.spinner("Generating responses..."):
            
            # BEFORE
            st.subheader("BEFORE (Neutral Agent)")
            st.caption("No personality applied - generic response")
            try:
                neutral_response = agent.respond(user_message, system_prompt=NEUTRAL_PROMPT)
                if len(neutral_response) > 600:
                    neutral_response = neutral_response[:600] + "\n\n*... [response truncated - original was very long]*"
                st.warning(neutral_response)
            except Exception as e:
                st.error(f"Error: {e}")
            
            st.markdown("")
            
            # AFTER
            st.subheader(f"AFTER ({demo_config['name']})")
            st.caption(demo_config['description'])
            try:
                styled_response = agent.respond(user_message, system_prompt=prompt)
                st.success(styled_response)
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()

# --- FOOTER ---
st.markdown("""
### How It Works

| Component | Purpose |
|-----------|---------|
| **Memory Extractor** | Analyzes conversation to extract preferences, patterns, facts |
| **Personality Engine** | Classifies user and selects appropriate personality |
| **Base Agent** | Generates response using the personality's system prompt |

[View source code on GitHub](https://github.com/sankalp1001/memory_extraction_and_personality_engine)
""")
