import contextlib
import os
import subprocess
import time

import pandas as pd
from typing import Sequence
from llama_index.core.indices.base import BaseIndex
from llama_index.core import (
    get_response_synthesizer, Settings, StorageContext,
    load_index_from_storage, Document, VectorStoreIndex
)
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import ollama_chat
import logging
from ClassArg import Args


def load_data(args: Args) -> pd.DataFrame:
    """
    Load CSV data from the specified directory into a DataFrame.

    Args:
        args (Args): The arguments containing directory path.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    if args.verbose:
        print("...loading data...")
    return pd.read_csv(args.directory)


def create_documents(df: pd.DataFrame, args: Args, dev_verbose: bool = False) -> Sequence[Document]:
    """
    Convert DataFrame rows into a list of Document objects.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        args (Args): The arguments for configuration.
        dev_verbose (bool): Flag for development verbose output.

    Returns:
        Sequence[Document]: List of Document objects.
    """
    if args.verbose:
        print("...creating documents...")

    documents = []
    for index, row in df.iterrows():
        doc = Document(
            id_=row['ID'],
            text=f"Ticket Number: {row['ID']}",
            metadata={
                "previously solved problem": f"Ticket {row['ID']}'s problem: {row['problem']}",
                "the solution": f"Ticket {row['ID']}'s solution: {row['solution']}",
            }
        )

        # Prevent LLM or embed model from seeing specific metadata
        doc.excluded_llm_metadata_keys = []  # LLM currently sees all
        doc.excluded_embed_metadata_keys = ["the solution"]

        # Prints what the LLM and embedded model can see, and the entire document's content.
        if args.dev_verbose and dev_verbose:
            print(f"\n\n{'##' * 20}\n\n>>> Document content from LLM perspective:\n~~~~~~~~~~")
            print(doc.get_content(metadata_mode=MetadataMode.LLM))
            print(f"\n{'==' * 10}\n\n>>> Document content from EMBED perspective:\n~~~~~~~~~~")
            print(doc.get_content(metadata_mode=MetadataMode.EMBED))
            print(f"\n{'==' * 10}\n\n>>> Document content:\n~~~~~~~~~~")
            print(f"Document ID: {doc.get_doc_id()}")
            print(f"Document text: {doc.get_text()}")
            print(f"Document metadata: {doc.get_metadata_str()}")
            print(f"\n{'##' * 20}")

        documents.append(doc)
    return documents


def create_document(ticket_num: str, problem: str, solution: str, args: Args) -> Document:
    """
    Makes a document and returns it.

    Args:
        ticket_num (str): Ticket's number
        problem (str): Ticket's problem
        solution (str): The solution

    Returns:
        Document: The ticket's document
    """
    if args.verbose:
        print("...creating document...")

    doc = Document(
        id_=str(ticket_num),
        text=f"Ticket Number: {ticket_num}",
        metadata={
            "previously solved problem": f"Ticket {ticket_num}'s problem: {problem}",
            "the solution": f"Ticket {ticket_num}'s solution: {solution}",
        }
    )
    return doc


def add_doc_to_index(doc: Document, index: BaseIndex, args: Args):
    """
    Updates the index by adding a document to it.

    Args:
        doc (Document): A document
        index (BaseIndex): The index to update

    Returns:
        BaseIndex: The newly updated index.
    """
    if args.verbose:
        print("...adding document to index...")

    index.insert(doc)
    return index


def delete_doc_from_index(doc_id: str, index: BaseIndex, args: Args):
    """
    Deletes the document from the index.

    Args:
        doc_id (str): The document's ID is the ticket number.
        index (BaseIndex): The index to delete from.

    Returns:
        BaseIndex: Updated index
    """
    if args.verbose:
        print("...deleting document from index...")

    index.delete_ref_doc(doc_id, delete_from_docstore=True)
    return index


def index_documents(documents: Sequence[Document], args: Args) -> VectorStoreIndex:
    """
    Index the documents using a vector store index.

    Args:
        documents (Sequence[Document]): List of Document objects.
        args (Args): The arguments for configuration.

    Returns:
        VectorStoreIndex: The created vector store index.
    """
    if args.verbose:
        print("...indexing documents...")

    index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)
    return index


def save_index(index: BaseIndex, args: Args):
    """
    Save the index to local storage.

    Args:
        index (VectorStoreIndex): The index to save.
        args (Args): The arguments for configuration.
    """
    if args.verbose:
        print("...saving index...")

    index.storage_context.persist(persist_dir=args.save_dir)


def load_index(args: Args) -> StorageContext:
    """
    Load the index from local storage.

    Args:
        args (Args): The arguments for configuration.

    Returns:
        StorageContext: The loaded storage context.
    """
    if args.verbose:
        print("...loading index...")

    return StorageContext.from_defaults(persist_dir=args.save_dir)


def set_settings(args: Args):
    """
    Set the necessary settings for the application.

    Args:
        args (Args): The arguments for configuration.
    """
    if args.verbose:
        print("...configuring settings and embedding model...")

    Settings.enabled = False
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        # Suppress console output during this block
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)  # Suppress warnings
        Settings.llm = None
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)  # Restore warnings


def create_query_engine(index: BaseIndex, args: Args) -> RetrieverQueryEngine:
    """
    Create and return a query engine for the provided index.

    Args:
        index (VectorStoreIndex): The vector store index to use.
        args (Args): The arguments for configuration.

    Returns:
        RetrieverQueryEngine: The created query engine.
    """
    if args.verbose:
        print("...Creating query engine...")

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=args.top_docs,
        embed_model=Settings.embed_model
    )
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=get_response_synthesizer(),
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=args.similarity)],
    )
    return query_engine


def query_index(query: str, query_engine: RetrieverQueryEngine, args: Args):
    """
    Query the index and return matching documents.

    Args:
        query (str): The query string.
        query_engine (RetrieverQueryEngine): The query engine to use.
        args (Args): Arguments for configuration

    Returns:
        Response: The response from the query engine.
    """
    if args.verbose:
        print("...querying index...")

    response = query_engine.query(query)
    return response


def reload_from_db(args: Args):
    """
    Reload data from the database, create documents, index them, and save the index.

    Args:
        args (Args): The arguments for configuration.

    Returns:
        VectorStoreIndex: The newly created index.
    """
    if args.verbose:
        print("...reloading index from db...")

    df = load_data(args=args)
    documents = create_documents(df=df, args=args, dev_verbose=False)
    index = index_documents(documents=documents, args=args)
    save_index(index=index, args=args)
    return index


def reload_from_local(args: Args):
    """
    Load the index from local storage, save it, then return it.

    Args:
        args (Args): The arguments for configuration.

    Returns:
        VectorStoreIndex: The loaded index.
    """
    if args.verbose:
        print("...reloading index from storage...")

    storage_context = load_index(args=args)
    index = load_index_from_storage(storage_context)
    save_index(index=index, args=args)
    return index


def get_llm_response(args: Args, response, temp: float = 0.5, model_name: str = "phi3:3.8b", stop="/*"):
    """
    Get a response from the LLM based on the query and response.

    Args:
        args (Args): The arguments for configuration.
        response: The response from the query engine.
        temp (float): Temperature setting for the LLM.
        model_name (str): The model name to use for LLM.
        stop (str): Token indicating the end of the response.
    """

    if args.verbose:
        print("...getting LLM response...")

    # Rules that the LLM needs to follow
    instruct_query = (
        f"(System Instructions: 1. You are a helpful assistant that always follows all of your instructions.\n"
        f"2. Important: Your response must contain all of the tickets.\n"
        f"3. Important: Only use information from the most recent <context></context> tag. Do not include steps or info"
        f" that were not part of the context.\n"
        f"4. Important: If the newest context has no tickets, or is empty, then say so and end your response.\n"
        f"5. Keep in mind the user is tech adept.\n"
        f"6. Guide the technician in resolving the issue by presenting the solution in steps.\n"
        f"7. Use WYSIWYG formatting techniques, lists, outlines, circular bullet points, and"
        f" separators between the different tickets.\n"
        f"8. Maintain accuracy and professionalism in your short responses.\n"
        f"9. Carefully talk about every ticket by displaying the ticket's problem and number only, "
        f"then go into the steps for the solution.\n"
        f"10. Important: Your response must contain all of the tickets\n"
        f"11. Do not forget to include a ticket.\n"
        f"12. Make sure to only include information from the ticket's solution. Don't add in your own "
        f"solutions or explanations.\n"
        f"13. Don't include tags in your response.\n"
        f"14. Make sure you relay all of the ticket's solution info and convey the meaning.)\n"
    )

    # The following is a three shot prompt to prompt engineer the responses.
    three_ticket_ans = (
        """
**Wifi Problem Resolution**

To resolve WiFi problems, follow these steps:

### Ticket Number 16: 
### Ticket's problem:  Wifi is not working, no networks are being shown.
### Solution Steps:

1. Uninstall WiFi Driver
    • Go to Device Manager in your computer settings.
    • Locate and expand 'Network adapters'.
    • Find your WiFi adapter, right-click on it, then select 'Uninstall device'.
    • Confirm any prompts that appear.
2. Reinstall WiFi Driver
    • After uninstallation is complete, click the 'Scan for hardware changes' button to reinstall the driver automatically.
    • Alternatively, manually install a new driver if necessary.

<====================>

### Ticket Number 12: 
### Ticket's problem: Laptop is frozen and we can't clock in.
### Solution Steps:

1. Check Airplane Mode
    • Ensure your laptop isn’t in airplane mode; this can prevent it from connecting to networks and other devices, causing freezes during tasks like clocking in.
2. Turn Off Airplane Mode
    • Turn off airplane mode by going to 'Settings' > 'Network & Internet' on Windows, or follow a similar pathway depending on the OS version.

<====================>

### Ticket Number 5: 
### Ticket's problem: Windows screen is bugged out and task bar is messed up.
### Solution Steps:

1. Restart File Explorer
    • Open Task Manager.
    • Locate ‘File Explorer’ in the list of processes, click on it to highlight all instances if there are multiple.
    • Click 'Restart' for each instance; this will restart File Explorer without needing a full system reboot.

***/ Please remember, this content may not be an exact solution. It was generated by using similar tickets from the past.
You can use the ticket number's to find the original solutions. /***
        """
    )

    three_ticket_context = (
        f"""
<context>
Context information is below.
---------------------
previously solved problem: Ticket 16's problem: Wifi is not working, no networks are being shown.
the solution: Ticket 16's solution: Go to device manager and uninstall the wifi driver. 
Then click the scan for hardware changes button.

Ticket Number: 16

previously solved problem: Ticket 12's problem: Laptop is frozen and we can't clock in
the solution: Ticket 12's solution: The laptop was in airplane mode, simply turn airplane mode off.

Ticket Number: 12

previously solved problem: Ticket 5's problem: Windows screen is bugged out and task bar is messed up.
the solution: Ticket 5's solution: Restart windows explorer or file explorer from task manager 
or equivalent.

Ticket Number: 5
---------------------
Given the context information and not prior knowledge, answer the query.
Query: wifi problems
Answer: <instruct>{instruct_query}</instruct></context>

<llm>
{three_ticket_ans}
</llm>
        """
    )

    # Testing one shot prompt to reduce amount of prompt tokens
    one_shot_prompt = (
        f"""
<instruct>Here is a two shot prompt where you will learn how to respond. The following contexts are purely for learning and
are not the actual context.</instruct>

{three_ticket_context}

<instruct>This marks the end learning how to respond. Do not use the previous contexts for information in your response.
The next context that you see is where you will get your information from.</instruct>

<context>
{response}
<instruct>
{instruct_query}
</instruct>
</context>
        """
    )

    llm_response = None

    if args.verbose and args.llm_verbose:
        print("\n\n\n" + "#" * 80)

    # Send the prompt to the LLM for processing, unless the embedding model found no matches.
    if len(response.response) < 20:
        print(
            "Unfortunately, this query did not return any matches within the system. This could be because it was"
            " too specific, too general, or the problem just isn't recorded in the system yet."
        )
    else:
        llm_response = ollama_chat.message_ollama(query=one_shot_prompt, temp=temp, default_sys=False,
                                   model_name=model_name, args=args, end_token=stop)

    if args.verbose and args.llm_verbose:
        print("\n\n" + "#" * 80 + "\n")

    return llm_response


def start_ollama_server(args: Args):
    """
    Start the Ollama server process.

    Args:
        args (Args): Arguments for configuration
    """
    if args.verbose:
        print("...starting ollama...")

    # Start the server process
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,  # Suppress standard output
        stderr=subprocess.DEVNULL  # Suppress standard error
    )
    time.sleep(.5)


def kill_ollama(args: Args):
    """
    Stop the Ollama server process.

    Args:
        args (Args): Arguments for configuation.
    """
    if args.verbose:
        print("...stopping ollama process...")

    # Command to stop the Ollama process using PowerShell
    command = "Get-Process | Where-Object {$_.ProcessName -like '*ollama*'} | Stop-Process"
    subprocess.run(["powershell", "-command", command])


def main(args: Args):
    """
    Main function to run the application. Showcases the functionality of this script.
    """

    # Settings for llamaIndex
    set_settings(args=args)

    # Initializing vars
    index = None
    query_engine = None

    # Reloads data from database or local storage
    if input("\nDo you want to reload data from db? (y/n): ").lower() == "y":
        index = reload_from_db(args=args)
    else:
        index = reload_from_local(args=args)

    # Deletes a ticket/document from index
    delete_input = input("\nDo you want to delete a document? If yes, enter ticket number (if no, hit enter): ").lower()
    if delete_input != "":
        index = delete_doc_from_index(doc_id=delete_input, index=index, args=args)
        save_index(args=args, index=index)

    # Inserts ticket/document to index
    ticket_num = input("\nDo you want to insert a document? If yes, enter ticket number (if no, hit enter): ").lower()
    if ticket_num != "":
        problem = input("Enter the problem/symptoms: ")
        solution = input("Enter the solution/steps: ")
        doc = create_document(ticket_num=str(ticket_num), problem=problem, solution=solution, args=args)
        index = add_doc_to_index(doc=doc, index=index, args=args)
        save_index(args=args, index=index)

    # Start the Ollama server (hard sleeps for .5 secs currently)
    start_ollama_server(args=args)

    # Perform a query by inputting your problem
    query = input("\n\nQuery >> ")
    query_engine = create_query_engine(index=index, args=args)
    response = query_index(query=query, query_engine=query_engine, args=args)

    # Response generated by query engine using matching documents
    if args.dev_verbose:
        print("\n\nQUERY ENGINE RESPONSE:", response)


    # Get response from the LLM
    get_llm_response(response=response, temp=args.temp, model_name=args.model_name, args=args, stop="/*")

    # Stop the Ollama server
    kill_ollama(args=args)


if __name__ == "__main__":


    args = Args(
        verbose=True,  # General verbose output
        timings=True,  # Displays token times
        dev_verbose=True,  # Development verbose output
        directory=r".\TicketTests.csv",  # Path to the CSV
        save_dir=r".\llama_storage",  # Directory to save the index
        top_docs=3,  # Number of top docs considered during index query
        top_k=3,  # Number of top tokens that are considered
        top_p=.5,  # Top tokens with a cumulative % less than top_p are considered
        temp=0,  # Predictability
        similarity=.5,  # Low cut-off for similarity to query
        llm_verbose=True,  # Turns on formatting for LLM response
        model_name="llama3.1:8b-q5_k_m" # Different LLMs have varying formats and capabilities
    )

    main(args=args)

"""
    Viable models:
    - llama3.1:8b >> Great formatting, good speed, good intelligence (good for quick testing)
    - phi3:14b >> Best at everything but speed. (Still acceptable tokens/sec)
        * temp .05
    - llama3.1:8b-q8_0 >> Higher quantization than q5. Best version of llama3.1:8b.
        * temp .05
    - llama3.1:8b-q5_k_m >> Lower quantization than q8
"""
