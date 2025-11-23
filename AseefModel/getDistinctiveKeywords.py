import pandas as pd
import numpy as np
import re
from collections import Counter


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


def tokenizeText(text):
    words = re.findall(r'\b[a-z]{3,}\b', text)
    return [word for word in words if word not in STOPWORDS]


def getDistinctiveKeywords(dataframe, targetClass, topN=10):
    targetText = ' '.join(dataframe[dataframe['label'] == targetClass]['allText'])
    otherText = ' '.join(dataframe[dataframe['label'] != targetClass]['allText'])

    targetTokens = tokenizeText(targetText)
    otherTokens = tokenizeText(otherText)

    targetCounts = Counter(targetTokens)
    otherCounts = Counter(otherTokens)

    totalTargetWords = sum(targetCounts.values())
    totalOtherWords = sum(otherCounts.values())

    keywordScores = []

    for word, count in targetCounts.items():
        if count < 5:
            continue

        probabilityInTarget = count / totalTargetWords
        probabilityInOther = (otherCounts[word] + 1) / totalOtherWords

        lift = probabilityInTarget / probabilityInOther
        score = lift * np.log(count)

        keywordScores.append((word, score))

    keywordScores.sort(key=lambda x: x[1], reverse=True)

    return [word for word, score in keywordScores[:topN]]


def main():
    trainingData = pd.read_csv('training_data.csv')

    textColumns = [
        "In your own words, what kinds of tasks would you use this model for?",
        "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
        "When you verify a response from this model, how do you usually go about it?"
    ]

    trainingData['allText'] = trainingData[textColumns].apply(
        lambda row: ' '.join(row.values.astype(str)).lower(),
        axis=1
    )
    
    automaticKeywords = {
        'ChatGPT': getDistinctiveKeywords(trainingData, 'ChatGPT'),
        'Gemini': getDistinctiveKeywords(trainingData, 'Gemini'),
        'Claude': getDistinctiveKeywords(trainingData, 'Claude')
    }

    print("Discovered Keywords:")
    for modelName, keywords in automaticKeywords.items():
        print(f"\n{modelName}:")
        print(f"  {keywords}")


if __name__ == "__main__":
    main()
