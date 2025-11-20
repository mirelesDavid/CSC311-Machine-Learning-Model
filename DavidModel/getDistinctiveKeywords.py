import pandas as pd
import numpy as np
import re
from collections import Counter

# 1. Load Data
df = pd.read_csv('training_data.csv')

# 2. Preprocessing
text_cols = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?"
]
df['all_text'] = df[text_cols].apply(lambda row: ' '.join(row.values.astype(str)).lower(), axis=1)

# --- IMPROVEMENT 1: Aggressive Stopword List ---
# Filters out common English words AND domain-specific filler words
STOPWORDS = set([
    'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'for', 'that', 'i', 'on', 'with', 'this', 'model', 
    'response', 'use', 'would', 'be', 'as', 'but', 'not', 'or', 'are', 'have', 'from', 'if', 'can', 
    'my', 'at', 'about', 'so', 'like', 'an', 'just', 'what', 'do', 'me', 'which', 'when', 'out', 'up', 
    'by', 'more', 'was', 'get', 'some', 'make', 'how', 'time', 'very', 'good', 'better', 'much', 'really', 
    'also', 'don', 'know', 'think', 'usually', 'check', 'find', 'other', 'using', 'will', 'has', 'give', 
    'gives', 'does', 'answers', 'answer', 'asking', 'ask', 'question', 'questions', 'task', 'tasks',
    'assignment', 'assignments', 'help', 'helping', 'able', 'need', 'want', 'tried', 'try', 'output',
    'example', 'examples', 'different', 'provide', 'provided', 'correct', 'incorrect', 'right', 'wrong',
    'understand', 'understanding', 'similar', 'however', 'because', 'since', 'then', 'than', 'even',
    'first', 'second', 'always', 'often', 'sometimes', 'never', 'likely', 'unlikely', 'sure', 'unsure',
    'create', 'creating', 'write', 'writing', 'generated', 'generating', 'look', 'looking', 'see', 'seeing',
    'thing', 'things', 'something', 'anything', 'everything', 'lot', 'bit', 'way', 'ways', 'part', 'parts',
    'point', 'points', 'result', 'results', 'work', 'working', 'works', 'problem', 'problems', 'issue', 'issues',
    'error', 'errors', 'mistake', 'mistakes', 'information', 'info', 'data', 'content', 'text', 'word', 'words',
    'sentence', 'sentences', 'paragraph', 'paragraphs', 'page', 'pages', 'paper', 'papers', 'essay', 'essays',
    'report', 'reports', 'email', 'emails', 'code', 'coding', 'program', 'programming', 'script', 'scripts',
    'language', 'languages', 'math', 'mathematics', 'equation', 'equations', 'formula', 'formulas', 'calculate',
    'calculating', 'calculation', 'calculations', 'solve', 'solving', 'solution', 'solutions', 'explain',
    'explaining', 'explanation', 'explanations', 'convert', 'converting', 'conversion', 'conversions',
    'format', 'formatting', 'formatted', 'style', 'styles', 'tone', 'tones', 'voice', 'voices',
    'above', 'below', 'necessary', 'effectively', 'resulting', 'edits', 'normal', 'taught', 'rephrasing'
])

def get_improved_keywords(df, target_class, top_n=10):
    """
    Finds distinctive keywords using a Balance Score: Lift * log(Frequency).
    This ensures words are both distinctive (high lift) AND common enough to be useful features.
    """
    
    # Split text by class
    target_text = ' '.join(df[df['label'] == target_class]['all_text'])
    other_text = ' '.join(df[df['label'] != target_class]['all_text'])
    
    def tokenize(text):
        # Get only words with 3+ chars
        words = re.findall(r'\b[a-z]{3,}\b', text)
        # Filter stopwords
        return [w for w in words if w not in STOPWORDS]
    
    target_tokens = tokenize(target_text)
    other_tokens = tokenize(other_text)
    
    target_counts = Counter(target_tokens)
    other_counts = Counter(other_tokens)
    
    # Total word counts for normalization
    total_target = sum(target_counts.values())
    total_other = sum(other_counts.values())
    
    scores = []
    
    # --- IMPROVEMENT 2: Robust Frequency Threshold ---
    # Only consider words that appear at least 5 times in the target class
    for word, count in target_counts.items():
        if count < 5: continue 
        
        # P(word | class)
        p_target = count / total_target
        
        # P(word | NOT class)
        # Add smoothing (+1) to avoid division by zero
        p_other = (other_counts[word] + 1) / total_other 
        
        # --- IMPROVEMENT 3: Weighted Score ---
        # Lift = How much more likely is it in Target vs Others?
        lift = p_target / p_other
        
        # Score = Lift * log(Count)
        # This penalizes rare words even if they have high lift
        score = lift * np.log(count)
        
        scores.append((word, score))
        
    # Sort by Score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N words
    return [x[0] for x in scores[:top_n]]

# 3. Generate the Dictionary
auto_keywords = {
    'ChatGPT': get_improved_keywords(df, 'ChatGPT'),
    'Gemini': get_improved_keywords(df, 'Gemini'),
    'Claude': get_improved_keywords(df, 'Claude')
}

print("Automatically Discovered Dictionary:")
print(auto_keywords)