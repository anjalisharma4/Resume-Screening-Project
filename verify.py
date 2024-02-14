import os

#file_size = os.path.getsize('tfidf.pkl')
import pickle

with open('clf.pkl', 'rb') as f:
    data = f.read()

print(data)
