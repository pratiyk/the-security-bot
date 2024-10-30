from flask import Flask, request, jsonify, render_template
import openai
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = ""  # Replace with your OpenAI key

# Load sentence transformer model for similarity checking
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, efficient model for sentence similarity

# Define a base knowledge base with common cybersecurity questions and answers
knowledge_base = {
    "What is phishing?": "Phishing is a type of cyber attack in which attackers pretend to be a trusted entity to trick individuals into revealing sensitive information, such as usernames, passwords, and credit card numbers.",
    "What is malware?": "Malware is software designed to harm, exploit, or otherwise compromise a system. Types include viruses, trojans, ransomware, and spyware.",
    "How can I protect my online accounts?": "Use strong, unique passwords for each account, enable two-factor authentication, and be cautious of suspicious emails or links.",
    "What is a firewall?": "A firewall is a network security device that monitors and filters incoming and outgoing network traffic based on security rules. It acts as a barrier between a trusted network and an untrusted network.",
    "What is ransomware?": "Ransomware is a type of malware that encrypts a victim's files. The attacker then demands a ransom from the victim to restore access to the data.",
}

# Encode knowledge base questions for similarity search
knowledge_base_questions = list(knowledge_base.keys())
knowledge_base_embeddings = model.encode(knowledge_base_questions, convert_to_tensor=True)

# Function to fetch the latest vulnerabilities from the CVE API
def fetch_latest_vulnerabilities():
    url = "https://cve.circl.lu/api/last"  # Using CIRCL CVE API
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            vulnerabilities = []
            for item in data[:5]:  # Fetch top 5 recent vulnerabilities
                vuln_info = f"{item['id']}: {item['summary']}"
                vulnerabilities.append(vuln_info)
            print("Vulnerabilities fetched successfully.")
            return vulnerabilities
        else:
            print(f"Error fetching vulnerabilities: Status {response.status_code}")
            return ["Unable to fetch the latest vulnerabilities."]
    except Exception as e:
        print(f"Exception while fetching vulnerabilities: {e}")
        return [f"An error occurred while fetching vulnerabilities: {e}"]

# Function to scrape cybersecurity news articles
def scrape_cybersecurity_articles():
    url = "https://thehackernews.com/"
    articles = []
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        article_titles = soup.find_all('h2', class_='home-title')
        for title in article_titles[:5]:  # Get top 5 articles
            articles.append(title.get_text())
        print("Articles fetched successfully.")
        return articles
    except Exception as e:
        print(f"Exception while scraping articles: {e}")
        return [f"An error occurred while scraping articles: {e}"]

# Function to handle user queries with similarity matching and OpenAI fallback
def handle_query(user_query):
    # Encode user query
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    
    # Compute cosine similarity with knowledge base
    cos_similarities = util.pytorch_cos_sim(query_embedding, knowledge_base_embeddings)[0]
    best_match_idx = torch.argmax(cos_similarities).item()
    best_match_score = cos_similarities[best_match_idx].item()

    # Define a similarity threshold
    similarity_threshold = 0.7
    if best_match_score >= similarity_threshold:
        best_match_question = knowledge_base_questions[best_match_idx]
        print(f"Best match found in knowledge base with score: {best_match_score}")
        return knowledge_base[best_match_question]
    
    # If no good match in knowledge base, fallback to OpenAI API
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=user_query,
            temperature=0.5,
            max_tokens=150
        )
        print("Response from OpenAI API received.")
        return response.choices[0].text.strip()
    except openai.error.RateLimitError:
        print("Rate limit error from OpenAI API.")
        return "I'm currently experiencing a high volume of requests. Please try again later."
    except openai.error.OpenAIError as e:
        print(f"OpenAI error: {str(e)}")
        return f"An error occurred: {str(e)}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    print("Received user input:", user_input)  # Debugging line
    
    if "latest vulnerabilities" in user_input.lower():
        vulnerabilities = fetch_latest_vulnerabilities()
        print("Vulnerabilities fetched:", vulnerabilities)  # Debugging line
        return jsonify({"response": "Here are some of the latest vulnerabilities:\n" + "\n".join(vulnerabilities)})
    
    elif "latest cybersecurity news" in user_input.lower():
        articles = scrape_cybersecurity_articles()
        print("Articles fetched:", articles)  # Debugging line
        return jsonify({"response": "Here are some of the latest cybersecurity articles:\n" + "\n".join(articles)})

    response = handle_query(user_input)
    print("Response generated:", response)  # Debugging line
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
