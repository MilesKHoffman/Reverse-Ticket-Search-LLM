import ollama
import time

# Initialize an empty list to store conversation messages
messages = []

# Start an infinite loop for continuous interaction
while True:
    # Get user input
    prompt = input("\n\nYou >> ")

    # Append user message to messages list with predefined role and content
    messages.append({
        'role': 'user',
        'content': "(You are a helpful assistant that is direct and concise, and answers recent prompts) " + prompt
    })

    # Measure the time taken for generating the prompt
    prompt_timer = time.time()

    # Calculate total number of characters in all messages
    prompt_total = sum(len(message['content']) for message in messages)

    # Send the prompt to the model with specified options
    stream = ollama.chat(
        model='llama3.1:8b',
        messages=messages,
        stream=True,
        options={
            "num_gpu": -1  # Use the CPU
        }
    )

    # Calculate the prompt time and prevent division by zero
    prompt_end_time = time.time() - prompt_timer
    if prompt_end_time == 0:
        prompt_end_time = 0.001

    # Print prompt statistics
    print(f"\n\nPROMPT TOKENS: {prompt_total}")
    print(f"PROMPT TIME: {prompt_end_time:.4f} seconds")
    print(f"PROMPT TOKEN/SEC: {prompt_total / prompt_end_time:.4f}")

    # Initialize response variables
    total_tokens = 0
    response_content = ''
    response_timer = time.time()

    # Print assistant's response
    print("\n\nAssistant >>\n")
    for chunk in stream:
        # Print each chunk of the response as it is received
        print(chunk['message']['content'], end='', flush=True)
        total_tokens += 1
        response_content += chunk['message']['content']
        if chunk['done']:
            break

    # Append assistant response to messages list
    messages.append({'role': 'assistant', 'content': response_content})

    # Calculate response time and print evaluation statistics
    response_end_time = time.time() - response_timer
    print(f"\n\nEVAL TOKENS: {total_tokens}")
    print(f"EVAL TIME: {response_end_time:.4f} seconds")
    print(f"EVAL TOKEN/SEC: {total_tokens / response_end_time:.4f}")
