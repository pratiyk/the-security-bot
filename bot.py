from flask import Flask, request, jsonify, render_template
import anthropic
from sentence_transformers import SentenceTransformer, util
import torch
import requests
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
NVD_API_KEY = os.getenv('NVD_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Load sentence transformer model for similarity checking
model = SentenceTransformer('all-MiniLM-L6-v2')

# Expanded knowledge base with more cybersecurity topics
knowledge_base = {
    "What is phishing?": "Phishing is a type of cyber attack in which attackers pretend to be a trusted entity to trick individuals into revealing sensitive information, such as usernames, passwords, and credit card numbers.",
    "What is malware?": "Malware is software designed to harm, exploit, or otherwise compromise a system. Types include viruses, trojans, ransomware, and spyware.",
    "How can I protect my online accounts?": "Use strong, unique passwords for each account, enable two-factor authentication, and be cautious of suspicious emails or links.",
    "What is a firewall?": "A firewall is a network security device that monitors and filters incoming and outgoing network traffic based on security rules. It acts as a barrier between a trusted network and an untrusted network.",
    "What is ransomware?": "Ransomware is a type of malware that encrypts a victim's files. The attacker then demands a ransom from the victim to restore access to the data.",
    "What are the basics of cyberhygiene?": """Here are the essential basics of cyber hygiene:
1. Use strong, unique passwords for all accounts
2. Enable two-factor authentication whenever possible
3. Keep software and systems updated regularly
4. Use antivirus software and keep it updated
5. Back up important data regularly
6. Be cautious with email attachments and links
7. Use encrypted connections (HTTPS) for sensitive activities
8. Regularly monitor accounts for suspicious activity
9. Use a password manager
10. Be careful with public Wi-Fi networks""",
    "What is social engineering?": "Social engineering is a manipulation technique that exploits human psychology to gain unauthorized access to systems or information. Attackers trick people into breaking normal security procedures, often by impersonating trusted contacts or creating a sense of urgency.",
    "How can I protect against social engineering?": "To protect against social engineering: 1) Be skeptical of unsolicited communications 2) Verify the identity of anyone requesting sensitive information 3) Never give out passwords or personal details 4) Use multi-factor authentication 5) Educate yourself and your team about common social engineering tactics.",
}

# Encode knowledge base questions for similarity search
knowledge_base_questions = list(knowledge_base.keys())
knowledge_base_embeddings = model.encode(knowledge_base_questions, convert_to_tensor=True)

def generate_claude_response(user_query):
    """Generate a response using Anthropic Claude with proper error handling"""
    try:
        if not ANTHROPIC_API_KEY:
            logger.error("Anthropic API key not configured")
            return "The chatbot is not properly configured. Please check the API settings."

        system_prompt = """You are a cybersecurity expert assistant providing clear, accurate, and helpful information about cybersecurity topics. 
Respond to questions with concise, professional, and informative answers. Focus on practical advice and current best practices."""

        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Using the Haiku model
            max_tokens=250,
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": f"Please provide a detailed and helpful response to this cybersecurity question: {user_query}"
                }
            ]
        )
        
        return response.content[0].text.strip()
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {str(e)}")
        return "I'm currently experiencing a technical issue with the AI service. Please try using one of our predefined cybersecurity questions."
    except Exception as e:
        logger.error(f"Unexpected error in Claude response generation: {str(e)}")
        return "An unexpected error occurred. Please try again or use one of our predefined cybersecurity questions."

def handle_query(user_query):
    """Handle user queries with improved error handling and fallback options"""
    try:
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
            logger.info(f"Best match found in knowledge base with score: {best_match_score}")
            return knowledge_base[best_match_question]
        
        # If no good match in knowledge base, use Claude
        return generate_claude_response(user_query)
        
    except Exception as e:
        logger.error(f"Error in query handling: {str(e)}")
        return "I encountered an error processing your question. Please try asking about specific cybersecurity topics like 'What is phishing?' or 'How can I protect my online accounts?'"

# Existing fetch_latest_vulnerabilities and scrape_cybersecurity_articles functions remain the same
def fetch_latest_vulnerabilities():
    """Fetch the latest vulnerabilities from the NVD API"""
    url = "https://services.nvd.disa.mil/rest/v1/cve/1.0"
    params = {
        "resultsPerPage": 5,
        "apiKey": NVD_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        vulnerabilities = []
        for item in data.get("result", {}).get("vulnerabilities", []):
            vuln_info = f"{item['cveId']}: {item['cveDescription']}"
            vulnerabilities.append(vuln_info)
        logger.info("Vulnerabilities fetched successfully.")
        return vulnerabilities
    except requests.RequestException as e:
        logger.error(f"Error fetching vulnerabilities: {e}")
        return ["Unable to fetch the latest vulnerabilities."]

def scrape_cybersecurity_articles():
    """Fetch cybersecurity news articles using News API"""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "cybersecurity",
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevance",
        "pageSize": 5
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = []
        for article in data.get("articles", []):
            articles.append(article["title"])
        logger.info("Articles fetched successfully.")
        return articles
    except requests.RequestException as e:
        logger.error(f"Error fetching articles: {e}")
        return ["Unable to fetch the latest articles."]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
            
        logger.info("Received user input: %s", user_input)

        if "latest vulnerabilities" in user_input.lower():
            vulnerabilities = fetch_latest_vulnerabilities()
            return jsonify({"response": "Here are some of the latest vulnerabilities:\n" + "\n".join(vulnerabilities)})
    
        elif "latest cybersecurity news" in user_input.lower():
            articles = scrape_cybersecurity_articles()
            return jsonify({"response": "Here are some of the latest cybersecurity articles:\n" + "\n".join(articles)})

        response = handle_query(user_input)
        return jsonify({"response": response})
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == "__main__":
    # Verify API keys
    if not ANTHROPIC_API_KEY:
        logger.warning("Anthropic API key not set. Claude responses will not be available.")
    
    app.run(debug=False)  # Set debug=False for production