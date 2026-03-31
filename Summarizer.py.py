from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import PyPDF2  
nltk.download('punkt', quiet=True)

PDF_FILE_PATH = "your_document.pdf"  

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except FileNotFoundError:
        print(f"Error: File '{pdf_path}' not found!")
        return ""
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

text = extract_text_from_pdf(PDF_FILE_PATH)

if not text:
    print("No text extracted from PDF. Please check the file path.")
    exit()

print(f"Successfully extracted {len(text)} characters from PDF.\n")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Remove special characters and numbers
    return text

cleaned_text = clean_text(text)

words = word_tokenize(cleaned_text)
stop_words = {"is", "the", "in", "and", "of", "to", "it", "also", "a", "an", "for", "on", "with", "by"}

filtered_words = [w for w in words if w not in stop_words]

freq = {}
for word in filtered_words:
    if word in freq:
        freq[word] += 1
    else:
        freq[word] = 1

sorted_words = sorted(freq, key=freq.get, reverse=True)
keywords = sorted_words[:5]

sentences = sent_tokenize(text)
scores = {}

for sent in sentences:
    score = 0
    for word in word_tokenize(sent.lower()):
        if word in freq:
            score += freq[word]
    scores[sent] = score

if scores:
    best_sentence = max(scores, key=scores.get)
else:
    best_sentence = "No sentences found."

print("Loading BART model... This may take a while the first time.")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=150,
    min_length=50,
    num_beams=4,
    length_penalty=2.0,
    early_stopping=True
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

original_words = len(word_tokenize(text))
summary_words = len(word_tokenize(summary))
reduction = (original_words - summary_words) / original_words * 100 if original_words > 0 else 0

print("\n" + "="*60)
print("ORIGINAL TEXT (first 500 chars):")
print(text[:500] + "..." if len(text) > 500 else text)

print("\n" + "="*60)
print("TRANSFORMER (ABSTR ACTIVE) SUMMARY:")
print(summary)

print("\n" + "="*60)
print("EXTRACTIVE BEST SENTENCE:")
print(best_sentence)

print("\n" + "="*60)
print("KEYWORDS:")
print(", ".join(keywords))

print("\n" + "="*60)
print("STATISTICS:")
print(f"Original Words : {original_words}")
print(f"Summary Words  : {summary_words}")
print(f"Reduction      : {round(reduction, 2)}%")
