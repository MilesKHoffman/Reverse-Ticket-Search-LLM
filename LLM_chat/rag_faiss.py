import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama_chat
import pickle

def load_data(file_path):
    """Load CSV data into a DataFrame."""
    return pd.read_csv(file_path)

def create_embeddings(problems, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings for the problems using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(problems)
    return model, embeddings

def load_model(model_name="all-MiniLM-L6-v2"):
    """Loading the tokenizer model"""
    model = SentenceTransformer(model_name)
    return model

def initialize_faiss_index(embeddings):
    """Initialize and populate a FAISS index with embeddings."""
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))
    return index

def semantic_search(query_text, model, index, top_k=2):
    """Perform semantic search to find the most relevant IDs."""
    query_embedding = model.encode([query_text])
    distances, indices = index.search(np.array(query_embedding), k=top_k)
    matched_ids = []
    matched_ids.extend(idx + 1 for idx in indices[0].tolist())
    return matched_ids, distances[0].tolist()

def retrieve_solutions(df, matched_ids, distances):
    """Retrieve solutions and associated problems for the matched IDs."""
    solutions = []
    problems = []
    for matched_id in matched_ids:
        matched_solutions = df.loc[df['ID'] == matched_id, 'solution']
        solutions.extend(matched_solutions.tolist())
        matched_problems = df.loc[df["ID"] == matched_id, "problem"]
        problems.extend(matched_problems.tolist())
    return solutions, problems

def generate_response(query_text, solutions, problems, distances, temp=.5, model_name="phi3:3.8b"):
    """Generate and print the final response using Ollama and retrieved solutions."""
    solutions_text = "SOLUTION: " + '\nSOLUTION: '.join(solutions)
    prob_text = "PROBLEM: " + '\nPROBLEM: '.join(problems)
    distances_text = [f"DISTANCE: {distance}\n" for distance in distances]
    formatted_query = (
        "Your task is to assist an IT technician by providing concise, direct, and clear instructions. "
        "Below is the technician's reported issue and a list of solutions from previous tickets related"
        " to this problem. Use each of the solutions (separately) to guide the technician in resolving the issue"
        ", providing detailed steps where necessary. Present each solution separately and use formatting techniques"
        "    such as bullet points or numbered lists. No markdown. Do not include links, introduce new information not"
        " present in the data, or add content you are not fully confident about. Maintain professionalism,"
        " accuracy, and brevity in your responses. Keep your responses straight to the point.\n\n"
        f"Problem: {query_text}\n\nSolutions: {solutions}"
    )

    print("\n" + "#" * 80)
    ollama_chat.message_ollama(query=formatted_query, temp=temp, default_sys=False, model_name=model_name)
    print("\n" + "#" * 80 + "\n")
    print("\n" + prob_text)
    print("\n" + solutions_text)
    print("\n" + "".join(distances_text))

def save_index_and_dataframe(index, df, index_file_path, df_file_path):
    """Save FAISS index and dataframe locally."""
    faiss.write_index(index, index_file_path)
    df.to_pickle(df_file_path)

def load_index_and_dataframe(index_file_path, df_file_path):
    """Load FAISS index and dataframe from local storage."""
    if os.path.exists(index_file_path) and os.path.exists(df_file_path):
        index = faiss.read_index(index_file_path)
        df = pd.read_pickle(df_file_path)
        return index, df
    return None, None

def main():
    # Define file paths
    index_file_path = 'faiss_index.bin'
    df_file_path = 'ticket_data.pkl'
    data_file_path = r'C:\Users\Miles\LLMs\Data\TicketTests.csv'
    model = None

    # This will reload the index and dataframe.
    reload = input('Reload data? (y/n) >> ')

    # Attempt to load existing index and dataframe
    print("\n...Checking for existing index and data...")
    index, df = load_index_and_dataframe(index_file_path, df_file_path)

    if index is None or df is None or reload == "y":
        # Load and preprocess data
        print("\n...Loading and preparing data...")
        df = load_data(data_file_path)

        # Create embeddings and initialize FAISS index
        print("\n...Creating faiss index and loading model...")
        model, embeddings = create_embeddings(df["problem"])
        index = initialize_faiss_index(embeddings)

        # Save the index and dataframe
        print("\n...Saving index and dataframe...")
        save_index_and_dataframe(index, df, index_file_path, df_file_path)

    else:
        print("\n...Loading transformer model...")
        model = load_model()

    # Perform semantic search and retrieve solutions
    print("\n...Performing similarity search...")
    query_text = input("Query >> ")
    matched_ids, distances = semantic_search(query_text=query_text, model=model, index=index, top_k=3)
    solutions, problems = retrieve_solutions(df, matched_ids, distances)

    # Generate and print the response
    print("\n...Generating response...")
    generate_response(query_text=query_text, solutions=solutions, problems=problems, temp=.1,
                      model_name="llama3.1:8b-q5_k_m", distances=distances)
    """
    Viable models:
    - llama3.1:8b >> Great formatting, good speed, avg intelligence
    - phi3:3.8b >> Best speed, good intelligence, but formatting isn't great. (Prompt engineering could improve it)
    - phi3:14b >> Best at everything but speed. 
    - phi3:3.8b-q8_0 >> higher quantization than 3.8b
    - llama3.1:8b-q8_0 >> higher quantization than 8b
    - llama3.1:8b-q5_k_m >> lower quant than q8
    
    """

if __name__ == "__main__":
    main()
