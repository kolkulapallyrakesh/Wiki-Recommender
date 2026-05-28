%%writefile main.py
import streamlit as st
import pandas as pd
import wikipediaapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(page_title="Dynamic Wiki Recommender", page_icon="🌐", layout="centered")
st.title("Wikipedia Article Recommendation Engine")
st.markdown("Type any valid Wikipedia category below")
@st.cache_resource
def init_wiki_api():
    return wikipediaapi.Wikipedia(
        user_agent="DynamicWikiRecSystem/2.0 (rakeshraki6806@gmail.com)",
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

wiki_wiki = init_wiki_api()
@st.cache_data(show_spinner="Fetching live Wikipedia pages...")
def fetch_and_vectorize_category(category_name, max_articles=30):
    if not category_name.startswith("Category:"):
        category_name = f"Category:{category_name.strip()}"
        
    category = wiki_wiki.page(category_name)
    
    if not category.exists():
        return None, None
        
    articles_data = []
    count = 0
  
    for member in category.categorymembers.values():
        if count >= max_articles:
            break
        if member.ns == wikipediaapi.Namespace.MAIN:
            articles_data.append({
                "title": member.title,
                "text": member.text if member.text else "",
                "url": member.fullurl
            })
            count += 1
            
    if not articles_data:
        return None, None
    df = pd.DataFrame(articles_data)
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['text'].fillna(''))
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, similarity_matrix
with st.sidebar:
    st.header("Engine Category details")
    target_category = st.text_input("Enter Wikipedia Category:", value="Machine learning")
    max_pages = st.slider("Max pages to auto-fetch:", min_value=10, max_value=100, value=30, step=10)
    num_recommendations = st.slider("Number of Recommendations:", min_value=2, max_value=10, value=5)

if target_category:
    df, similarity_matrix = fetch_and_vectorize_category(target_category, max_pages)
    
    if df is not None and similarity_matrix is not None:
        st.success(f"✅ Successfully auto-fetched {len(df)} live articles from **{target_category}**!")
        all_titles = df['title'].tolist()
        selected_title = st.selectbox("Select an article to explore similarities:", options=all_titles)
        
        if selected_title:
            idx = df[df['title'] == selected_title].index[0]
            similarity_scores = list(enumerate(similarity_matrix[idx]))
            sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            top_recommendations = sorted_scores[1:num_recommendations + 1]
            
            st.subheader(f"✨ Top Recommendations for '{selected_title}'")
            st.write("---")
            
            for rank, (i, score) in enumerate(top_recommendations, 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"### {rank}. [{df['title'].iloc[i]}]({df['url'].iloc[i]})")
                    snippet = df['text'].iloc[i][:150] + "..." if len(df['text'].iloc[i]) > 150 else df['text'].iloc[i]
                    st.write(f"*{snippet}*")
                with col2:
                    st.metric(label="Match Score", value=f"{score*100:.1f}%")
                st.write("---")
    else:
        st.error(f"❌ Could not retrieve articles for '{target_category}'. Ensure the spelling matches a valid Wikipedia category.")
