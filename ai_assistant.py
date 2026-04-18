import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, Literal
import re
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

## -- Negation Schema for extracting excluded ingredients and clean query from user input --
class NegationConstraints(BaseModel):
    excluded_ingredients: list[str] = Field(
        default_factory=list,
        description="Items the user wants to EXCLUDE (ingredients, techniques, or tools like 'nuts', 'frying', 'oven')"
    )
    clean_query: str = Field(
        description="The original query stripped of negation phrases, for embedding search"
    )

class ClassificationResult(BaseModel):
    intent: Literal["recipe", "general"] = Field(description="The category of the request.")
    n_results: int = Field(default=5, description="Number of results requested. 'a'/'one'=1, 'all'=10, default=5, max=20.")

## -- Pydantic models for structured recipe output --
class Ingredient(BaseModel):
    name: str = Field(description="Ingredient name")
    amount: Optional[str] = Field(default=None, description="Amount of ingredient if found")

class RecipeOutput(BaseModel):
    recipe_title: str = Field(description="Recipe title")
    recipe_link: Optional[str] = Field(default=None, description="Link to recipe")
    ingredients: list[Ingredient] = Field(description="List of ingredients with name and amount")
    directions: str = Field(description="Directions numbered")

class RAGResponse(BaseModel):
    answer_type: Literal["recipe", "general", "not_found"] = Field(
        description="Choose 'recipe' if extracting a specific recipe. Choose 'general' if answering a broad question. Choose 'not_found' if context is missing."
    )
    general_answer: Optional[str] = Field(
        default=None,
        description="Text answer for general questions or not_found messages."
    )
    recipes: Optional[list[RecipeOutput]] = Field(
        default=None,
        description="A list of structured recipe data if answer_type is 'recipe'."
    )
    routing_decision: Optional[str] = Field(default=None, description="The classification route used (recipe or general)")


## --Global state --
_vectorstore  = None
_llm          = None
_df           = None


## -- Initialization -- 
def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    return _llm

## -- Get Chroma DB vector Store --
def get_vectorstore(collection_name: str = "recipes_df", db_path: str = "./chroma_db"):
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        )
    return _vectorstore

## -- Get DataFrame for general queries --
def get_df(csv_path: str = "recipes_sample.csv"):
    global _df
    if _df is None:
        _df = pd.read_csv(csv_path)
        # Fill NaN values in ingredients_clean with empty strings to avoid NaN errors
        if 'ingredients_clean' in _df.columns:
            _df['ingredients_clean'] = _df['ingredients_clean'].fillna('')
    return _df




## -- Classification  to route between recipe retrieval and general questions --
def classify_input():
    prompt = ChatPromptTemplate.from_template("""Given the chat history and the current question, classify the request and identify the number of results requested.

Categories:
- "recipe": Finding/retrieving recipes, cooking instructions, or asking about specific details/follow-ups (e.g., "Does it have nuts?", "Show me pasta recipes").
- "general": Dataset-wide statistics, counts, or metadata (e.g., "How many recipes?", "What sources?").

Extraction Rules for n_results which is the number of recipes or dishes ONLY:
1. Look for numbers (e.g., "3 recipes", "10 dishes") → use that number directly.
2. Look for the word "all" → set n_results to 10.
3. Look for "a" or "an" used as an indefinite article (e.g., "give me a recipe", "find an option"),
   or keywords "one" / "single" → set n_results to 1.
4. If no quantity is specified, default to 5.
5. Max value allowed is 20. If the user asks for more than 20, set n_results to 20.
History:
{history}

Question: {question}""")
    return prompt | get_llm().with_structured_output(ClassificationResult)

## -- Helper function to rewrite follow-up questions into standalone queries for better retrieval --
def rewrite_prompt():
    prompt = ChatPromptTemplate.from_template("""Given the following conversation history, rewrite the user's latest follow-up question into a standalone search query.

Rules:
1. If the question contains pronouns (it, they, this, the recipe), replace them with the actual subject from the history.
2. If the user asks for a generic recipe (e.g., "a recipe", "something else", "show me more"), treat it as a fresh search. Do NOT carry over specific dish names (like "shakshuka") from history unless the user implies a connection (e.g., "another one like that").
3. If the question is REFINING the current topic (e.g., "now show me ones with olive oil"), combine them.
4. If a previous search FAILED or returned no results, treat the next user input as a fresh attempt/new topic.
5. Make the query specific and searchable.

Example:
 - History: "show me cake recipes without eggs" -> Results: [Vegan Cake, Apple Cake]
 - Question: "a recipe without oven"
 - Rewrite: "recipe without oven"

History:
{history}

Latest Question: {question}
Return ONLY the rewritten search query, nothing else.""")
    return prompt | get_llm()


## -- General function to handle non-recipe questions using text-to-pandas --
def get_pandas_query():
    prompt = ChatPromptTemplate.from_template("""You are a data analyst. Given this pandas dataframe schema:
{schema}

Sample row:
{sample}

Write a single Python pandas expression to answer this question: {question}

IMPORTANT INSTRUCTIONS FOR INGREDIENT QUESTIONS:
- If the question asks about ingredients, their frequency, or ingredient statistics (e.g., "most common ingredient", "top ingredients", "ingredient distribution"):
  ALWAYS use the 'ingredients_clean' column instead of 'ingredients'.
- The 'ingredients_clean' column contains clean ingredient names WITHOUT:
  - Cooking adjectives (chopped, minced, diced, sliced, etc.)
  - Measurement words (cup, tbsp, tsp, oz, pinch, etc.)
  - Container types (can, jar, package, etc.)
- If searching within ingredients_clean using .str.contains(): ALWAYS use na=False to handle empty strings
- For non-ingredient questions, use the appropriate column normally.

Examples:
CORRECT for ingredient search: df[df['ingredients_clean'].str.contains('garlic', na=False)].shape[0]
CORRECT for frequency: df['ingredients_clean'].value_counts().head(10)
WRONG: df['ingredients'].value_counts()
WRONG: df[df['ingredients_clean'].str.contains('garlic')] (missing na=False)

Respond ONLY with the pandas code, no explanation, no markdown.""")
    return prompt | get_llm()

## -- Main recipe extraction that uses retrieved context to answer recipe questions --
def get_recipe():
    prompt = ChatPromptTemplate.from_template("""You are an expert culinary assistant. Use only the provided context to answer the user's request.

Context (Recipes):
{context}

User Question/Intent: {question}

Instructions:
1. If the user wants to FIND or LIST recipes: Set `answer_type` to "recipe" and include up to {n_results} matching recipes from the provided context.
2. If the user is asking a SPECIFIC QUESTION about a recipe (e.g., "Does it have nuts?"): Set `answer_type` to "general" and answer directly based on context.
3. Formatting Directions: In the `directions` field, ensure every step starts with a number (e.g., "1. ") and is followed by exactly one newline.
4. Ingredients: Carefully extract the exact quantity/amount for every ingredient. If a quantity is mentioned (e.g., "2 cups", "pinch"), include it in the `amount` field.
5. EXCLUSIONS: Only enforce the specific items (ingredients/techniques/tools) mentioned in the exclusion note. If a recipe contains them, omit it from the results.
6. If NO recipes meet the criteria: Set `answer_type` to "not_found" and explain which exclusions caused the conflict.

Response Schema:
{schema}""")
    return prompt | get_llm().with_structured_output(RAGResponse)

## -- Extract negation/exclusion constraints from user queries --
def get_negation_prompt():
    prompt = ChatPromptTemplate.from_template("""Analyze this recipe query for negation/exclusion constraints.

Query: {query}

Extract:
1. `excluded_ingredients`: Any ingredients the user does NOT want (look for: "without", "no", "not", "free", "avoid", "exclude", "allergic to", "dairy-free", "nut-free", etc.)
   - Include ingredients (e.g., "no eggs"), techniques (e.g., "without fry"), or tools (e.g., "no oven").
   - Normalize to simple singular names: "egg" not "eggs", "fry" not "frying".
   - Include variants: "dairy" covers milk, butter, cream, cheese
   - Ignore nonsensical non-culinary items (e.g., "without basketball").
2. `clean_query`: Rewrite the query for the search engine. If the user excludes a core ingredient of a dish (e.g., "shakshuka without eggs"), the clean_query should be the general dish type or its base ingredients (e.g., "tomato and pepper stew") to find successful alternatives.

Examples:
- "chocolate cake without nuts" → excluded: ["nut"], clean: "chocolate cake"
- "shakshuka without eggs" → excluded: ["egg"], clean: "spicy tomato pepper vegetable stew"
- "pasta recipe, no boiling" → excluded: ["boil"], clean: "pasta recipe"
- "vegan cookies" → excluded: [], clean: "vegan cookies"  ← no explicit negation
- "chicken recipes without oven" → excluded: ["oven"], clean: "chicken recipes"

Response schema: {schema}""")
    return prompt | get_llm().with_structured_output(NegationConstraints)

## -- Main recipe question-answering that incorporates negation constraints and history rewriting --
def filter_by_exclusions(docs: list, excluded: list[str], df: pd.DataFrame) -> list:
    """
    Cross-reference retrieved docs against ingredients and directions to drop 
    any recipe that contains a forbidden term.
    """
    if not excluded:
        return docs

    pattern = "|".join([rf"\b{re.escape(e.lower())}\b" for e in excluded])

    filtered = []
    for doc in docs:
        title = doc.metadata.get("title", "")
        
        # Check doc.page_content (contains title, raw ingredients, and directions)
        text_to_check = doc.page_content.lower()
        
        # Also check clean ingredients from DF for precision on food items
        row = df[df["title"].str.lower() == title.lower()]
        if not row.empty:
            ing_clean = row.iloc[0].get("ingredients_clean", "").lower()
            text_to_check += f" {ing_clean}"

        if not re.search(pattern, text_to_check):
            filtered.append(doc)
        else:
            found = [e for e in excluded if re.search(rf"\b{re.escape(e.lower())}\b", text_to_check)]
            print(f"🚫 Excluded '{title}' — contains: {found}")

    return filtered


##  -- Format history --
def format_history(chat_history: list) -> str:
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])


def to_langchain_messages(chat_history: list) -> list:
    mapping = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}
    return [mapping[m["role"]](content=m["content"]) for m in chat_history[-6:]]


# ── Main Function ──────────────────────────────────────────────────────────────
def ask_recipe(question: str, chat_history: list = None, n_results: int = 5) -> RAGResponse:
    if chat_history is None:
        chat_history = []

    # 1. Classify
    classification = classify_input().invoke({
        "question": question,
        "history": format_history(chat_history)
    })
    question_type = classification.intent
    n_results = classification.n_results 
    print("number of results requested:", n_results)

    # 2. General → Text-to-Pandas (unchanged)
    if question_type == "general":
        df = get_df()
        code = get_pandas_query().invoke({
            "schema":   df.dtypes.to_string(),
            "sample":   df.iloc[0].to_dict(),
            "question": question,
        }).content.strip()
        print(f"🐼 Running: {code}")
        try:
            # Pass a restricted namespace: only 'df' is accessible
            result = eval(code, {"df": df, "pd": pd}, {})

            # Handle Pandas Series/DataFrame results for better UI rendering
            if isinstance(result, (pd.Series, pd.DataFrame)):
                if isinstance(result, pd.Series):
                    # Convert Series (like value_counts) to list of dicts via reset_index
                    output_data = result.reset_index().to_dict(orient="records")
                else:
                    output_data = result.to_dict(orient="records")
                general_answer = json.dumps(output_data)
            else:
                general_answer = str(result)

            return RAGResponse(answer_type="general", general_answer=general_answer, routing_decision="general")
        except Exception as e:
            return RAGResponse(answer_type="not_found", general_answer=f"Could not compute answer: {e}", routing_decision="general")

    # 3.a Rewrite for history context first
    search_query = question
    if chat_history:
        search_query = rewrite_prompt().invoke({
            "history":  format_history(chat_history),
            "question": question,
        }).content.strip()
        print(f"🧠 Rewrote: '{question}' ➔ '{search_query}'")

    # 3.b Check for negation keywords in the rewritten query
    negation_keywords = r"\b(no|without|don't|dont|doesn't|doesnt|free|avoid|allergic|none|minus|excluding|except)\b"
    
    if re.search(negation_keywords, search_query.lower()):
        negation: NegationConstraints = get_negation_prompt().invoke({
            "query":  search_query,
            "schema": NegationConstraints.model_json_schema(),
        })
    else:
        negation = NegationConstraints(excluded_ingredients=[], clean_query=search_query)

    print(f"🚫 Excluded: {negation.excluded_ingredients}")

    # 5. Retrieve with High Recall if negations exist
    actual_search = negation.clean_query
    # Increase recall significantly to find non-oven/non-egg recipes
    n_candidates = max(n_results * 6, 30) if negation.excluded_ingredients else n_results
    
    docs = get_vectorstore().as_retriever(
        search_kwargs={"k": n_candidates}
    ).invoke(actual_search)

    # 6. POST-FILTER: drop docs containing excluded ingredients
    docs = filter_by_exclusions(docs, negation.excluded_ingredients, get_df())

    if not docs:
        excluded_str = ", ".join(negation.excluded_ingredients)
        return RAGResponse(
            answer_type="not_found",
            general_answer=f"No recipes found that exclude: {excluded_str}.",
            routing_decision="recipe"
        )

    context = "\n\n---\n\n".join([doc.page_content for doc in docs[:n_results]])

    # 7. Also reinforce exclusions in the LLM prompt
    exclusion_note = f"\nUser Intent: {search_query}"
    if negation.excluded_ingredients:
        exclusion_note += f"\nIMPORTANT: The user wants recipes WITHOUT: {negation.excluded_ingredients}. If a recipe in context contains these, OMIT it. If no recipes remain, follow Instruction 6."

    response = get_recipe().invoke({
        "context":  context,
        "question": exclusion_note,
        "n_results": n_results,
        "schema":   RAGResponse.model_json_schema(),
    })
    response.routing_decision = "recipe"
    return response


## -- Example usage and testing --
if __name__ == "__main__":
    chat_history = []

    questions = [
        "How do I make chocolate chip cookies?",
        "Does it have nuts?",                       # follow-up
        "What is the most common ingredient?",      # text-to-pandas
        "How many recipes per site?",               # text-to-pandas
        "Give me a recipe from mars",               # not found
    ]

    for question in questions:
        print(f"\n🔍 {question}")
        result = ask_recipe(question, chat_history)

        if result.answer_type == "recipe" and result.recipes:
            for r in result.recipes:
                print(f"🍽  {r.recipe_title} — {r.recipe_link}")
                for ing in r.ingredients:
                    print(f"   - {ing.name}: {ing.amount}")
                print(f"📋  {r.directions}")
        else:
            print(f"💬  {result.general_answer}")

        # Update chat history
        chat_history.append({"role": "user",      "content": question})
        chat_history.append({"role": "assistant", "content": result.general_answer or str(result.recipes)})