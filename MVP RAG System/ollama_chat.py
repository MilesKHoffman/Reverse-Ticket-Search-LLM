from typing import Sequence

import ollama
import time
from ClassArg import Args


def message_ollama(args: Args, query, default_sys=False, messages=None, temp=.5, model_name='phi3:3.8b', end_token = None):
    if messages is None:
        messages = []

    system = ""
    if default_sys:
        system = ("You are a helpful assistant that is direct"
                  ", doesn't give lengthy responses, and answers the user without "
                  "lengthy excuses. Here is the prompt: ")

    messages.append({'role': 'user', 'content': system + query})

    prompt_timer = time.time()
    prompt_total = 0
    for message in messages: prompt_total += len(message['content'])

    """
    class Options(TypedDict, total=False):
      # load time options
      numa: bool
      num_ctx: int
      num_batch: int
      num_gpu: int
      main_gpu: int
      low_vram: bool
      f16_kv: bool
      logits_all: bool
      vocab_only: bool
      use_mmap: bool
      use_mlock: bool
      embedding_only: bool
      num_thread: int

      # runtime options
      num_keep: int
      seed: int
      num_predict: int
      top_k: int
      top_p: float
      tfs_z: float
      typical_p: float
      repeat_last_n: int
      temperature: float
      repeat_penalty: float
      presence_penalty: float
      frequency_penalty: float
      mirostat: int
      mirostat_tau: float
      mirostat_eta: float
      penalize_newline: bool
      stop: Sequence[str]
    """
    stream = ollama.chat(
        model=model_name,
        messages=messages,
        stream=True,
        options={
            "num_gpu": -1,
            "temperature": temp,
            "stop": ["/*"],
            "top_p": args.top_p,
            "top_k": args.top_k
        }
    )

    total = 0
    chonk = ""
    timer = time.time()
    prompt_end_time = None

    for chunk in stream:
        if total == 0:
            prompt_end_time = time.time() - prompt_timer
            if prompt_end_time == 0: prompt_end_time = .001
            if args.verbose and args.llm_verbose: print(f"\nASSISTANT ({model_name}) RESPONSE:\n\n")
            print(chunk['message']['content'].lstrip(), end='', flush=True)

        elif not chunk["done"]:
            print(chunk['message']['content'], end='', flush=True)

        else:
            break

        total += 1
        chonk += chunk['message']['content']

    messages.append({'role': 'assistant', 'content': chonk})

    end_time = time.time() - timer
    if args.verbose and args.timings:
        print("\n\n" + "-_-" * 10 + f"\nPROMPT TOKENS: {prompt_total}")
        print(f"PROMPT TIME: {prompt_end_time.__round__(2)} secs")
        print(f"PROMPT TOKEN/SEC: {(prompt_total / prompt_end_time).__round__()}")
        print(f"EVAL TOKENS: {total}")
        print("EVAL TIME: ", end_time.__round__(2), " secs")
        print("EVAL TOKEN/SEC: ", (total / end_time).__round__())
        print("-_-" * 10)

    return chonk


def conversation_ollama():
    args = Args(
        verbose=True,  # General verbose like loading, finished, etc.
        timings=True,
        dev_verbose=False,  # Activates all dev verbose for variable prints (function specific ones exist too)
        directory=r"C:\Users\Miles\LLMs\Data\Usable\TicketTests.csv",  # Directory to the csv,
        save_dir=r"./llama_storage/",
        top_k=3,
        similarity=0,
        llm_verbose=True
    )
    messages = []
    while True:
        query = input("\n\nQuery >> ")
        if query == 'quit':
            break

        message_ollama(query=query, messages=messages, args=args)

    pass


if __name__ == '__main__':
    conversation_ollama()
