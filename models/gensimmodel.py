from gensim.summarization import summarize
from gensim.summarization import keywords
import requests
import re

def tester(inp):
    """Switches between HTTPS and plain text formats"""
    if re.match(r'^https://', inp):
        text = requests.get(inp).text
        return (text)
    else:
        text = inp
        return (text)
    

entry = input("Enter any text. ")
text = tester(entry)
print("Summary: ")

print(summarize(text, ratio=0.5, word_count=500))