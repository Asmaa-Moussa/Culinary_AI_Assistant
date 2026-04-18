import os
import json
from dotenv import load_dotenv
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

## -- initialize recipe chunks --
def get_chunks(df: pd.DataFrame) -> list:
    chunks = (
        "Recipe Title: " + df["title"] + "\n" +
        "link: " + df["link"] + "\n" +
        "Ingredients: " + df["ingredients"] + "\n" +
        "Other Ingredients: " + df["NER"] + "\n" +
        "Directions: " + df["directions"] + "\n" + 
        "Source: " + df["source"] + "\n" +
        "Site: " + df["site"] + "\n"
    ).tolist()
    return chunks

## -- create chromadb collection and upsert chunks --
def create_chromadb(df: pd.DataFrame, chroma_path: str = "./chroma_db"):

    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )

    client = chromadb.PersistentClient(path=chroma_path)
    
    # Delete existing collection if it exists to avoid hnsw:space modification error
    try:
        client.delete_collection("recipes_df")
        print("🗑️  Deleted existing collection")
    except:
        pass
    
    collection = client.create_collection(
        name="recipes_df",
        embedding_function=embedding_fn,
    )

    chunks = get_chunks(df)

    # Upsert in batches
    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        collection.upsert(
            ids=[f"chunk_{j}" for j in range(i, i + len(batch))],
            documents=batch,
            metadatas=[{"index": j} for j in range(i, i + len(batch))],
        )
    
    ## -- Compute and store metadata about the collection --
    total_recipes = len(df)
    
    # Get source counts
    source_counts = df["source"].value_counts().to_dict()
    sources_summary = {str(k): int(v) for k, v in source_counts.items()}
    
    # Get site counts
    site_counts = df["site"].value_counts().to_dict()
    sites_summary = {str(k): int(v) for k, v in site_counts.items()}
    
    # Store metadata in the collection
    collection.metadata = {
        "total_recipes": total_recipes,
        "sources_summary": json.dumps(sources_summary),
        "sites_summary": json.dumps(sites_summary),
    }
    return collection

create_chromadb(pd.read_csv("recipes_sample.csv"))

