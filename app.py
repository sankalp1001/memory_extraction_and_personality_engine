"""
Memory Extraction & Personality Engine - Demo App

A minimal Streamlit interface to demonstrate:
1. User memory (pre-extracted from conversation)
2. Personality classification based on memory
3. Before/After response comparison
"""

import streamlit as st
import json
import os
from personality_engine import PersonalityEngine, NEUTRAL_PROMPT, PERSONALITIES
from base_agent import BaseAgent
from memory_extractor import MemoryExtractor

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

@st.cache_data
def load_test_users():
    with open("data/test_users.json", "r") as f:
        return json.load(f)["users"]

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
st.markdown("*Compare baseline (memory only) vs personality-adjusted responses*")

# Explanation of the process
with st.expander("How This Demo Works", expanded=True):
    # Visual flow
    st.markdown("""
    **Process Flow:**
    ```
    Test User Conversation
           ↓
    Memory Extraction (preferences, patterns, facts)
           ↓
    ┌──────────────────┬──────────────────┐
    │   BEFORE         │    AFTER         │
    │                  │                  │
    │ Neutral Prompt   │ Personality      │
    │ + Memory         │ Instructions     │
    │                  │ + Memory         │
    │                  │                  │
    │ Generic helpful  │ Tailored to      │
    │ response         │ personality      │
    └──────────────────┴──────────────────┘
    ```
    """)
    st.markdown("""
    This demo shows how **personality instructions** transform responses when combined with user memory.
    
    **The Process:**
    
    1. **Select a Test User** - Each test user has a conversation history that reveals their preferences and patterns
    
    2. **Memory Extraction** - The system extracts structured memory (preferences, emotional patterns, facts) from their conversation
    
    3. **Generate Two Responses** to the same message:
       - **BEFORE (Baseline)**: Uses a neutral prompt + user memory context only
         - The LLM knows about the user's preferences but has no special instructions on how to respond
       - **AFTER (Personality-Adjusted)**: Uses personality-specific instructions + user memory context
         - The LLM follows detailed personality guidelines (e.g., "don't give advice", "be brief", etc.) while respecting user memory
    
    4. **Compare** - See how personality instructions shape the response style while maintaining personalization from memory
    
    **Key Insight:** Both responses have access to the same user memory, but the personality instructions dramatically change how the AI responds to that information.
    """)

# Load test users
try:
    test_users = load_test_users()
except FileNotFoundError:
    st.error("Test users file not found. Please ensure data/test_users.json exists.")
    st.stop()

# Initialize session state for selected user
if "selected_user_id" not in st.session_state:
    st.session_state.selected_user_id = None

st.markdown("---")
st.markdown("### Step 1: Select a Test User")
st.markdown("**Choose a test user to see how their extracted memory affects responses:**")
st.caption("Each user has a conversation that reveals their preferences, communication style, and needs")

# Show buttons for each test user
cols = st.columns(2)
for i, user in enumerate(test_users):
    col = cols[i % 2]
    with col:
        personality_name = PERSONALITIES[user["expected_personality"]]["name"]
        button_label = f"{personality_name}\n_{user['description']}_"
        
        if st.button(button_label, key=f"user_{user['id']}", use_container_width=True):
            st.session_state.selected_user_id = user["id"]
            st.rerun()

# If a user is selected, show their demo
if st.session_state.selected_user_id:
    selected_user = next(u for u in test_users if u["id"] == st.session_state.selected_user_id)
    expected_personality = selected_user["expected_personality"]
    
    st.markdown("---")
    st.markdown("### Step 2: Memory Extraction")
    st.subheader(f"User: {PERSONALITIES[expected_personality]['name']}")
    st.markdown(f"*{selected_user['description']}*")
    
    st.info("""
    **Extracting structured memory from conversation...**
    
    The system analyzes the user's conversation history to identify:
    - **Preferences**: How they like to communicate, what they value
    - **Emotional Patterns**: Recurring patterns in their emotional experiences
    - **Facts**: Stable biographical or contextual information
    """)
    
    # Use cache for extraction (expensive operation)
    @st.cache_data
    def extract_user_memory(user_conversation):
        extractor = MemoryExtractor()
        # Conversation format already matches what extractor expects (turn, role, content)
        return extractor.extract(user_conversation)
    
    try:
        with st.spinner("Analyzing conversation and extracting memory..."):
            user_memory = extract_user_memory(selected_user["conversation"])
        
        st.success("Memory extracted successfully!")
        
        # Show extracted memory summary
        st.markdown("**Extracted Memory Summary:**")
        with st.expander("View Extracted Memory Details", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Preferences**")
                prefs = user_memory.get("preferences", [])
                if prefs:
                    for pref in prefs[:3]:
                        st.write(f"- {pref.get('value', '')}")
                    if len(prefs) > 3:
                        st.caption(f"*... and {len(prefs) - 3} more*")
                else:
                    st.caption("*No preferences extracted*")
            with col2:
                st.write("**Emotional Patterns**")
                patterns = user_memory.get("emotional_patterns", [])
                if patterns:
                    for pattern in patterns[:2]:
                        st.write(f"- {pattern.get('value', '')}")
                    if len(patterns) > 2:
                        st.caption(f"*... and {len(patterns) - 2} more*")
                else:
                    st.caption("*No patterns extracted*")
            with col3:
                st.write("**Facts**")
                facts = user_memory.get("long_term_facts", [])
                if facts:
                    for fact in facts[:2]:
                        st.write(f"- {fact.get('value', '')}")
                    if len(facts) > 2:
                        st.caption(f"*... and {len(facts) - 2} more*")
                else:
                    st.caption("*No facts extracted*")
        
        # Show test message
        test_message = selected_user["test_message"]
        st.markdown("---")
        st.markdown("### Step 3: Test Message")
        st.markdown("**This is the message we'll respond to:**")
        st.info(f'"{test_message}"')
        
        st.markdown("---")
        st.markdown("### Step 4: Generate Comparison")
        
        # Generate responses
        col1, col2 = st.columns([1, 4])
        with col1:
            generate_btn = st.button("Generate Comparison", type="primary", use_container_width=True)
        
        with col2:
            st.caption("Click to generate both responses using the same memory but different prompts")
        
        if generate_btn:
            with st.spinner("Generating responses (this may take 10-20 seconds)..."):
                # Build prompts
                baseline_prompt = engine.build_baseline_prompt(memory=user_memory)
                personality_prompt = engine.build_prompt(personality=expected_personality, memory=user_memory)
                
                # Show what's happening
                st.markdown("---")
                st.markdown("### Comparison Results")
                
                tab1, tab2 = st.tabs(["BEFORE (Baseline)", "AFTER (Personality-Adjusted)"])
                
                # BEFORE: Baseline with memory
                with tab1:
                    st.markdown("#### Baseline Response: Memory Context Only")
                    st.caption("""
                    **Prompt Used:** Neutral prompt ("You are a helpful assistant") + user memory context
                    
                    This response has access to the user's extracted memory (preferences, patterns, facts) 
                    but no specific instructions on how to communicate. It's a generic assistant that knows 
                    about the user but responds in a standard helpful way.
                    """)
                    
                    try:
                        baseline_response = agent.respond(test_message, system_prompt=baseline_prompt)
                        if len(baseline_response) > 600:
                            baseline_response = baseline_response[:600] + "\n\n*... [response truncated]*"
                        st.warning(baseline_response)
                        
                        # Show prompt preview
                        with st.expander("View Baseline Prompt (for debugging)"):
                            st.code(baseline_prompt[:500] + "..." if len(baseline_prompt) > 500 else baseline_prompt)
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                # AFTER: Personality with memory
                with tab2:
                    personality_config = engine.get_personality(expected_personality)
                    st.markdown(f"#### Personality Response: {personality_config['name']}")
                    st.caption(f"""
                    **Prompt Used:** {personality_config['name']} personality instructions + user memory context
                    
                    This response has the same memory context but also follows detailed personality guidelines:
                    - Communication style and tone
                    - What to do and what to avoid
                    - Response structure and length
                    
                    Notice how the personality instructions transform the response while still respecting the user's memory.
                    """)
                    
                    try:
                        personality_response = agent.respond(test_message, system_prompt=personality_prompt)
                        st.success(personality_response)
                        
                        # Show prompt preview
                        with st.expander("View Personality Prompt (for debugging)"):
                            st.code(personality_prompt[:500] + "..." if len(personality_prompt) > 500 else personality_prompt)
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                # Key differences
                st.markdown("---")
                st.markdown("#### Key Differences")
                st.info("""
                **Both responses have access to the same user memory**, but:
                - **BEFORE**: Generic helpful style, may give advice, standard length
                - **AFTER**: Follows specific personality rules (e.g., no advice, brief responses, certain tone)
                
                The personality instructions act as a "filter" that shapes how the AI responds to the memory context.
                """)
    
    except Exception as e:
        st.error(f"Error extracting memory: {e}")
        st.exception(e)

st.divider()

