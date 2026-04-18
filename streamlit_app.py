import streamlit as st
from ai_assistant import ask_recipe, RecipeOutput, RAGResponse, get_vectorstore
import pandas as pd
import re
import json
import ast

# -- Utility Functions for Rendering RAGResponse Content --
def display_ingredients_table(ingredients):
    if not ingredients:
        st.write("No ingredients found.")
        return
    data = [{"Ingredient": ing.name, "Amount": ing.amount if ing.amount else "N/A"} for ing in ingredients]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

def format_directions(directions_text):
    if not directions_text:
        return ""
    # Use a specific regex to identify step starts (e.g., "1. ") while avoiding 
    # false positives from temperatures (e.g., "350."). 
    # We look for 1-2 digits followed by a period and space, not preceded by another digit.
    # \s* replaces existing whitespace/newlines to ensure clean double-spacing for Streamlit.
    return re.sub(r'\s*(?<![0-9])(?=[0-9]{1,2}\.\s)', '\n\n', directions_text).strip()

def try_parse_json_or_dict(text: str):
    """Try to parse text as JSON or Python dict and return parsed object, or None if fails."""
    try:
        # Try JSON parse
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    
    try:
        # Try literal_eval as a safer alternative to eval
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except (SyntaxError, ValueError, ValueError):
        pass
    
    return None

def render_dict_as_table(data):
    """Render a dict or list of dicts as a formatted table."""
    if isinstance(data, dict):
        # Single dict - convert to DataFrame
        df = pd.DataFrame([data])
        st.dataframe(df, use_container_width=True, hide_index=True)
    elif isinstance(data, list):
        if data and isinstance(data[0], dict):
            # List of dicts
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            # List of other types - display as is
            st.write(data)
    else:
        st.write(data)

def render_response(content):
    """Centralized rendering for RAGResponse objects."""
    if isinstance(content, RAGResponse):
        if content.routing_decision:
            st.caption(f"🧭 Route: **{content.routing_decision.upper()}**")

        if content.answer_type == "recipe" and content.recipes:
            for recipe in content.recipes:
                with st.expander(f"Recipe: {recipe.recipe_title}", expanded=True):
                    if recipe.recipe_link:
                        st.caption(f"🔗 [View Original Recipe]({recipe.recipe_link})")
                    
                    tab1, tab2 = st.tabs(["🛒 Ingredients", "👨‍🍳 Directions"])
                    with tab1:
                        display_ingredients_table(recipe.ingredients)
                    with tab2:
                        st.write(format_directions(recipe.directions))
        
        elif content.answer_type in ["general", "not_found"] and content.general_answer:
            # Try to parse general_answer as JSON/dict and display as table
            parsed_data = try_parse_json_or_dict(content.general_answer)
            if parsed_data is not None:
                # st.info("📊 Response:")
                render_dict_as_table(parsed_data)
            else:
                st.info(content.general_answer)
    else:
        # Try to render plain text response as dict/table if possible
        parsed_data = try_parse_json_or_dict(str(content))
        if parsed_data is not None:
            # st.info("📊 Response:")
            render_dict_as_table(parsed_data)
        else:
            st.write(str(content))

## -- Streamlit App Setup --
st.set_page_config(page_title="Recipe Assistant", page_icon="🍳", layout="wide")

st.title("🍳 Culinary AI Assistant")
st.caption("Powered by RAG and OpenAI to find the perfect dish.")

# -- Initialize session state for chat history --
if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    get_vectorstore()
except Exception as e:
    st.error(f"Database Error: {e}")

## -- Chat Interface --
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_response(message["content"])

## -- User Input Handling --
if user_question := st.chat_input("Ask me about a recipe..."):

    ## -- Chat history formatting for RAG input --
    formatted_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            formatted_history.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            content = msg["content"]
            if isinstance(content, RAGResponse):
                if content.answer_type == "recipe" and content.recipes:
                    titles = ", ".join([r.recipe_title for r in content.recipes])
                    formatted_history.append({"role": "assistant", "content": f"I provided recipes for: {titles}"})
                elif content.general_answer:
                    formatted_history.append({"role": "assistant", "content": content.general_answer})
            else:
                formatted_history.append({"role": "assistant", "content": str(content)})

    # Add new user message to visual UI
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)
    
    # Get response from RAG
    with st.chat_message("assistant"):
        with st.spinner("Searching recipes..."):
            result = ask_recipe(user_question, chat_history=formatted_history)
        
        render_response(result)
        st.session_state.messages.append({"role": "assistant", "content": result})