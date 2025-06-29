import subprocess
import time
from colors import *
from config import OPENAI_API_KEY
import openai
import math

# openai.api_key = OPENAI_API_KEY


def run_ollama_model(message, model_name="llama2"):
    print(f"{GREEN}Message passed: {message}")

    try:
        run_start_time = time.time()
        result = subprocess.run(['ollama', 'run', model_name, message], capture_output=True, text=True)
        run_end_time = time.time()
        run_time = run_end_time - run_start_time
        result_message = ""

        if result.returncode == 0:
            print(f"{PURPLE}Response from Ollama ({model_name}):\n{result.stdout}{RESET}")
            result_message = result.stdout
        else:
            print(f"{RED}Error occurred:\n {result.stderr}{RESET}")
            result_message = result.stderr

        return result_message, run_time

    except Exception as e:
        print(f"{RED}An error occurred: {e}{RESET}")
        return None, None

def run_gpt_model(message, model_name="gpt-3.5-turbo"):
    """Run the GPT model (default is gpt-3.5-turbo if model_name is not passed)."""
    print(f"{BG_GREEN}Message passed: {message}{RESET}")

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": message}],
            temperature=0,
            logprobs=True
        )

        result_message = response["choices"][0]["message"]["content"]
        log_probs = response['choices'][0].get('logprobs')

        confidence = calculate_confidence_level(log_probs)

        print(f"{BG_YELLOW}{result_message}{RESET}")
        return result_message, confidence

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def calculate_confidence_level(log_probs):
    confidence = 0.5
    if log_probs:
        logprobs = [entry["logprob"] for entry in log_probs["content"][:5]]
        probabilities = [math.exp(logprob) for logprob in logprobs]
        total_probability = math.prod(probabilities)
        
        confidence = total_probability 
        
        print(f"Total Probability (used as confidence): {total_probability:.6f}")
        print(f"Confidence Level (0 to 1): {confidence:.6f}")
    else:
        print("No log probabilities returned in the response.")
    
    return confidence



def run_model(model_name, message):
    if model_name == "gpt":
        return run_gpt_model(message)
    elif model_name == "gpt4":
        return run_gpt_model(message, "gpt-4")
    elif model_name == "ollama":
        return run_ollama_model(message, "llama2")
    elif model_name == "llama3":
        return run_ollama_model(message, "llama3")
    else:
        print("Model is not supported. Try 'gpt', 'gpt4', 'ollama', or 'llama3'")
        return None, None

