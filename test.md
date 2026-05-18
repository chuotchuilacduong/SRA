# Default query
export API_BASE="http://localhost:11434/v1"
export MODEL="qwen2.5:7b"
python test_rrf_bfsrag_query.py

# Custom query
QUERY="What is the gradient descent algorithm?" \
API_BASE="http://localhost:11434/v1" MODEL="qwen2.5:7b" \
python test_rrf_bfsrag_query.py

# Try larger model for better accuracy
MODEL="qwen2.5:14b-instruct-q5_K_M" \
API_BASE="http://localhost:11434/v1" \
python test_rrf_bfsrag_query.py
