from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from flask_cors import CORS
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained('data_output/fine_tuned_model')
tokenizer = AutoTokenizer.from_pretrained('data_output/fine_tuned_model')

# Ensure the pad_token is set to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Load product data from the 'data/products.json' file
with open('data_input/products.json', 'r') as f:
    products = json.load(f)

# Prepare product descriptions and names
product_descriptions = [product['description'] for product in products]
product_names = [product['name'] for product in products]
# Combine product names and descriptions for text processing
product_texts = [name + " " + desc for name, desc in zip(product_names, product_descriptions)]

# Initialize the vectorizer and fit it to the product texts
vectorizer = TfidfVectorizer().fit(product_texts)

# List of product names for reference
product_names_list = [product['name'] for product in products]

# Load comparison data from the 'data/comparisons.json' file
with open('data_input/comparisons.json', 'r') as f:
    comparisons = json.load(f)


def search_products(user_input):
    """
    Search for the most relevant product based on the user's input.

    Args:
        user_input (str): The user's query.

    Returns:
        dict: The most relevant product information.
    """
    # Vectorize user input and product texts
    user_vec = vectorizer.transform([user_input])
    product_vecs = vectorizer.transform(product_texts)

    # Compute cosine similarity
    similarities = cosine_similarity(user_vec, product_vecs).flatten()

    # Find the index of the most similar product
    most_similar_index = similarities.argmax()

    # Get the most relevant product
    relevant_product = products[most_similar_index]
    return relevant_product


def compare_products(product_name1, product_name2):
    """
    Retrieve comparison information between two products.

    Args:
        product_name1 (str): The name of the first product.
        product_name2 (str): The name of the second product.

    Returns:
        str: The comparison information or an error message.
    """
    # Find comparisons that match the product names
    for comparison in comparisons:
        products_involved = comparison['products']
        if product_name1 in products_involved and product_name2 in products_involved:
            return comparison['comparison']
    return "I'm sorry, I don't have enough information to compare those products."


def estimate_accuracy(full_ids, input_length):
    """
    Estimate the accuracy (confidence) of the model's response.

    Args:
        full_ids (torch.Tensor): The combined input and output token IDs.
        input_length (int): The length of the input IDs before generation.

    Returns:
        float: The estimated accuracy as a percentage.
    """
    # Disable gradient calculations for efficiency
    with torch.no_grad():
        outputs = model(full_ids)

    logits = outputs.logits  # Shape: [1, seq_length, vocab_size]

    # Calculate the start position of generated tokens
    gen_start_pos = input_length - 1  # Adjust for model's prediction alignment

    # Extract logits corresponding to generated tokens (excluding the last logit)
    gen_logits = logits[:, gen_start_pos:-1, :]  # Exclude the last logit

    # Extract generated tokens
    generated_tokens = full_ids[:, input_length:]  # Tokens generated by the model

    # Compute probabilities
    probabilities = F.softmax(gen_logits, dim=-1)

    # Get probabilities of the generated tokens
    generated_token_probs = probabilities.gather(2, generated_tokens.unsqueeze(-1)).squeeze(-1)

    # Calculate the average probability
    avg_prob = torch.mean(generated_token_probs).item()

    # Convert to percentage
    accuracy = avg_prob * 100
    return accuracy


def generate_response(user_input):
    """
    Generate a response to the user's input, including product information or comparisons.

    Args:
        user_input (str): The user's message.

    Returns:
        tuple: A tuple containing the response text and the estimated accuracy.
    """
    # Check if the user is asking for a comparison
    if "compare" in user_input.lower():
        # Extract product names from the user input
        # This is a simplified example; consider using NLP techniques to extract entities
        words = user_input.split()
        product_names_in_input = [word for word in words if word in product_names_list]

        if len(product_names_in_input) >= 2:
            product_name1 = product_names_in_input[0]
            product_name2 = product_names_in_input[1]
            # Retrieve comparison information
            comparison_info = compare_products(product_name1, product_name2)
            response = comparison_info
            accuracy = 100.0  # Since the comparison is directly from data
            return response, accuracy
        else:
            response = "Please specify two products you'd like to compare."
            accuracy = 100.0
            return response, accuracy
    else:
        # Search for relevant product
        relevant_product = search_products(user_input)
        # Prepare product information to include in the model input
        product_info = f"Product Name: {relevant_product['name']}\nDescription: {relevant_product['description']}\n"

        # Combine user input with product information
        model_input = user_input + "\n" + product_info + tokenizer.eos_token
        input_ids = tokenizer.encode(model_input, return_tensors='pt')

        # Generate a response using the language model
        output = model.generate(
            input_ids,
            max_length=input_ids.size(1) + 50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        output_ids = output.sequences  # Shape: [1, total_length]

        # Decode the response and remove the prompt and product info
        response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = response_text[len(user_input):].strip()

        # Estimate the accuracy of the response
        accuracy = estimate_accuracy(output_ids, input_ids.size(1))

        return response, accuracy


@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle incoming chat messages from the frontend.

    Returns:
        json: A JSON response containing the agent's reply and estimated accuracy.
    """
    user_input = request.json.get('message')
    response, accuracy = generate_response(user_input)
    return jsonify({'response': response, 'accuracy': round(accuracy, 2)})


if __name__ == '__main__':
    app.run(debug=True)