from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from config import *
from nltk.tokenize import sent_tokenize

import json
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def setup_simple_rag(embedding_model="all-minilm:33m", index_path=r"../data/med_index", model_name="llama3.2:3b-instruct-fp16", temperature=0):
    embeddings = OllamaEmbeddings(model=embedding_model)
    model = OllamaLLM(model=model_name, temperature=temperature)
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    return embeddings, model, vector_store

def prepare_benchmark_data(benchmark_path=r"../data/benchmark.json", dataset_name="mmlu"):

    with open(benchmark_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    mmlu_benchmark = data[dataset_name]
    ground_truth_answers = []

    for row in mmlu_benchmark:
        ground_truth_answers.append(mmlu_benchmark[row]["answer"])

    test_examples = []

    for q in mmlu_benchmark:
        question = mmlu_benchmark[q]["question"]
        options = mmlu_benchmark[q]["options"]

        test_examples.append((question, options))

    return test_examples, ground_truth_answers

def retrieve_relevant_document(question, vector_store, k=1):

    retrieved_context = vector_store.similarity_search(question, k=k)
    return retrieved_context

def generate_response(chain, data_dict):

    response = chain.invoke(data_dict)[0]
    return response

def run_simple_rag(chain, query, options, vector_store):

    context = retrieve_relevant_document(query, vector_store)
    response = generate_response(chain, {"paragraph": context, "question": query, "options": options})

    return response

###################################################
############# Explainability ######################


def remove_one_sentence_perturbations(context: str) -> list[tuple[str, str, int]]:
    """
    Generate perturbations of the context by removing one sentence at a time.

    For each sentence in the input context, this function removes that sentence and returns a new perturbed version of the context. Each perturbation includes the modified context, the removed sentence, and the index of that sentence.

    Returns all the generated perturbations:
        list[tuple[str, str, int]]: A list of tuples, each containing:
            - str: Perturbed context with one word removed
            - str: The removed sentence.
            - int: The index of the removed sentence in the original context.

    """

    sentences = extract_sentences(context)
    perturbations = []

    for index, sentence in enumerate(sentences):
        temp_sentences = sentences[:]

        removed_sentence = temp_sentences.pop(index)

        perturbed_context = " ".join(temp_sentences)
        perturbed_context = add_newlines_before_documents(perturbed_context)
        perturbed_context = add_newlines_before_documents(perturbed_context)

        temp_perturbation = (perturbed_context, removed_sentence, index)
        perturbations.append(temp_perturbation)

    return perturbations

def remove_word_span(context, span_size):
    words = extract_tokens(context)
    perturbations = []

    for i in range(len(words) - span_size + 1):
        temp_words = words[:i] + words[i + span_size:]
        removed_words = words[i:i + span_size]

        perturbed_context = " ".join(temp_words)
        perturbed_context = add_newlines_before_documents(perturbed_context)

        temp_perturbation = (perturbed_context, " ".join(removed_words), i)
        perturbations.append(temp_perturbation)

    return perturbations

def extract_tokens(text):
    doc = nlp(text)
    words = [token.text for token in doc if not token.is_punct]

    print(f"Total words: {len(words)}")
    return words

def extract_sentences(document):
    sentences = sent_tokenize(document)
    return sentences

def add_newlines_before_documents(text):
    updated_text = re.sub(r'(?<!^) (Chunk \d+)', r'\n\n\1', text)
    return updated_text