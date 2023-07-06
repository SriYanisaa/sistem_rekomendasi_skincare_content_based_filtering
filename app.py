from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd

app = Flask(__name__)

# Load data from CSV
data = pd.read_csv('data_final_clean_femaledaily.csv')

# Representasi Fitur
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['description_processed'] + ' ' + data['review_processed'])

cosine_similarities = cosine_similarity(tfidf_matrix)

similarities = {}

for i in range(len(cosine_similarities)):
    similar_indices = cosine_similarities[i].argsort()[:-50:-1]
    similarities[data['product_name'].iloc[i]] = [(cosine_similarities[i][x], data['product_name'][x], data['product_brand'][x]) for x in similar_indices][1:]

class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def recommend(self, recommendation):
        selected_products = recommendation['product_names']
        num_recommendations = recommendation['sum_skincare']

        all_recom_skincare = []
        for skincare in selected_products:
            if skincare in self.matrix_similar:
                recom_skincare = self.matrix_similar[skincare][:num_recommendations]
                all_recom_skincare.extend(recom_skincare)

        all_recom_skincare = sorted(all_recom_skincare, key=lambda x: x[0], reverse=True)
        all_recom_skincare = all_recom_skincare[:num_recommendations]

        recommendations = []
        for product in all_recom_skincare:
            category = data.loc[data['product_name'] == product[1], 'category'].values[0]
            subcategory = data.loc[data['product_name'] == product[1], 'subcategory'].values[0]
            is_recommend = data.loc[data['product_name'] == product[1], 'is_recommend'].values[0]
            cosine_similarity_value = product[0]
            star_rating = data.loc[data['product_name'] == product[1], 'star_rating'].values[0]
            recommendations.append({
                'product_name': product[1],
                'product_brand': product[2],
                'category': category,
                'subcategory': subcategory,
                'is_recommend': is_recommend,
                'cosine_similarity': cosine_similarity_value,
                'star_rating': star_rating
            })

        return recommendations

recommender = ContentBasedRecommender(similarities)

@app.route('/')
def index():
    return render_template('index.html', products=data['product_name'].tolist())

@app.route('/recommend', methods=['POST'])
def recommend():
    recommendation = request.get_json()

    recommendations = recommender.recommend(recommendation)

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
