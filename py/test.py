from transformers import pipeline

# Load a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Generate text
result = generator("How can I create a c++ code in c++ 20 version ", max_length=500, num_return_sequences=1)
print(result)
