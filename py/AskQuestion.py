import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, pipeline
from torch.utils.data import Dataset
import nltk
from sentence_transformers import SentenceTransformer, util

# Ensure you have nltk resources for synonyms
nltk.download('wordnet')
nltk.download('omw-1.4')

# ======= Step 4: Query Using Semantic Similarity =======
# Load semantic search model
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings for all Q&A pairs
qa_embeddings = []
qa_prompts = []
qa_data = [
    {
        "question": "What are my key programming languages?",
        "answer": "C, C++, Python, Embedded Systems Development."
    },
    {
        "question": "What experience do I have with machine learning?",
        "answer": "I have 1-2 years of experience in deep learning and data science, including GNN-based path planning for AGVs."
    }
]
def generate_synonym_variants(question):
    """Generate variations of a question by replacing words with synonyms."""
    from nltk.corpus import wordnet
    words = question.split()
    variants = {question}  # Start with the original question

    for i, word in enumerate(words):
        synonyms = wordnet.synsets(word)
        for synonym in synonyms[:2]:  # Limit to 2 synonyms per word to avoid too many variants
            synonym_word = synonym.lemmas()[0].name().replace("_", " ")
            variant = words.copy()
            variant[i] = synonym_word
            variants.add(" ".join(variant))

    return list(variants)

# Augment questions using synonym variants
augmented_data = []
for item in qa_data:
    variants = generate_synonym_variants(item["question"])
    for variant in variants:
        augmented_data.append({"question": variant, "answer": item["answer"]})


for item in augmented_data:
    prompt = f"Q: {item['question']} A:"
    qa_prompts.append(prompt)
    qa_embeddings.append(semantic_model.encode(prompt, convert_to_tensor=True))

print("QA embeddings computed!")

# ======= Step 5: Answer User Questions =======
def find_best_answer(user_question):
    """Find the best answer using semantic similarity."""
    user_embedding = semantic_model.encode(f"Q: {user_question}", convert_to_tensor=True)

    # Ensure that qa_embeddings is a tensor
    qa_embedding_tensor = torch.stack(qa_embeddings)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(user_embedding, qa_embedding_tensor)

    # Get the index of the most similar prompt
    best_index = torch.argmax(similarities).item()

    # Generate text from the fine-tuned model using the closest matching prompt
    prompt = qa_prompts[best_index]
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return result[0]["generated_text"]



# Load fine-tuned pipeline for text generation
generator = pipeline("text-generation", model="./gpt2-finetuned-resume-qa", tokenizer="./gpt2-finetuned-resume-qa")

# Ask user questions and get robust answers
user_question = input("Ask a question about your resume: ")
response = find_best_answer(user_question)
print("Generated response:", response)
