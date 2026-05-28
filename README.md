# Wiki-Recommender
A dynamic, automated web application built with Python and Streamlit that fetches live Wikipedia articles across any user-specified category, vectorizes the text using TF-IDF, and recommends semantically related pages using Cosine Similarity.

## Features
- **Dynamic Data Scraping:** Connects directly to the MediaWiki REST API via `wikipedia-api`.
- **Natural Language Processing:** Implements automated text vectorization using Scikit-Learn's `TfidfVectorizer`.
- **Interactive UI:** A streamlined, responsive frontend dashboard built using Streamlit.

## How to Run Locally
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
