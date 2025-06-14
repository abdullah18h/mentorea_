# Enable build tools for lightfm compilation

import numpy as np
from flask import Flask, request, jsonify
import pickle
from pyngrok import ngrok
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize Flask app
app = Flask(__name__)

# Load LightFM objects (for mentees with interactions)

with open('./new_hybird_v2.pkl', 'rb') as f:
    model = pickle.load(f)
with open('./new_mentee_encoder_v2.pkl', 'rb') as f:
    mentee_encoder = pickle.load(f)
with open('./new_mentor_encoder_v2.pkl', 'rb') as f:
    mentor_encoder = pickle.load(f)
with open('./new_mentee_features_v2.pkl', 'rb') as f:
    mentee_features = pickle.load(f)
with open('./new_mentor_features_v2.pkl', 'rb') as f:
    mentor_features = pickle.load(f)
with open('./new_mentor_df_hybird_v2.pkl', 'rb') as f:
    mentor_df = pickle.load(f)


# Load content-based objects (for cold-start mentees)

with open('./new_tfidf_vectorizer_v2_content.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open('./new_mentee_skills_matrix_v2_content.pkl', 'rb') as f:
    mentee_skills_matrix = pickle.load(f)
with open('./new_mentor_skills_matrix_v2_content.pkl', 'rb') as f:
    mentor_skills_matrix = pickle.load(f)
with open('./new_mentee_df_v2_content.pkl', 'rb') as f:
     mentee_df = pickle.load(f)
with open('./new_mentor_df_v2_content.pkl', 'rb') as f:
     mentor_df = pickle.load(f)

# Load interaction data to check for cold-start mentees
with open('./new_interaction_df_v2.pkl', 'rb') as f:
    interaction_df = pickle.load(f)



# LightFM recommendation function (for mentees with interactions)
def recommend_mentors(mentee_id, top_k=50):
    try:
        # Attempt to transform the mentee_id directly (assuming UUID is encoded)
        mentee_idx = mentee_encoder.transform(np.array([mentee_id]).reshape(-1, 1))[0]
        mentee_idx = np.array(mentee_idx)

        # Validate the index range (0 to 149 based on your check)
        if not (0 <= mentee_idx[0] < 150):
            raise ValueError(f"Mentee_idx {mentee_idx[0]} is out of bounds for matrix with 150 mentees")

        n_mentors = 80
        mentor_indices = np.arange(n_mentors)
        mentee_indices = np.repeat(mentee_idx, len(mentor_indices))
        scores = model.predict(
            mentee_indices,
            mentor_indices,
            user_features=mentee_features,
            item_features=mentor_features
        )
        top_mentor_indices = np.argsort(-scores)[:top_k]
        top_mentor_ids = np.array(mentor_encoder.inverse_transform(top_mentor_indices.reshape(-1,1)))
        top_scores = scores[top_mentor_indices]
        top_mentor_ids= top_mentor_ids.reshape(-1)

        # Get mentor details
        recommendations = []
        for mentor_id, score in zip(top_mentor_ids, top_scores):
            mentor_info = mentor_df[mentor_df['Id'] == mentor_id]
            if not mentor_info.empty:
                mentor_details = {
                    'mentor_id': str(mentor_id),  # Ensure string format
                }
                recommendations.append(mentor_details)
        return recommendations
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return str(e)


# Content-based recommendation function (for cold-start mentees)
def recommend_mentors_content_based(mentee_id, top_k=50, weights=None):
    if weights is None:
        weights = {
            'skills': 0.4,
            'location': 0.1,
            'avg_rate': 0.25,
            'avg_availability': 0.1,
            'experience_years': 0.15
        }

    try:
        mentee_idx = mentee_df.index[mentee_df['Id'] == mentee_id].tolist()
        if not mentee_idx:
            raise ValueError(f"Mentee ID {mentee_id} not found in the dataset")
        mentee_idx = mentee_idx[0]

        mentee_skills_vector = mentee_skills_matrix[mentee_idx]
        mentee_location = mentee_df.loc[mentee_idx, 'location_encoded']

        skill_similarities = cosine_similarity(mentee_skills_vector, mentor_skills_matrix).flatten()
        location_similarities = (mentor_df['location_encoded'] == mentee_location).astype(int).values

        avg_rate_boost = mentor_df['avg_rate_normalized'].values
        avg_availability_boost = mentor_df['avg_availability_normalized'].values
        experience_years_boost = mentor_df['experience_years_normalized'].values

        combined_scores = (
            weights['skills'] * skill_similarities +
            weights['location'] * location_similarities
        )

        final_scores = combined_scores * (1 + weights['avg_rate'] * avg_rate_boost) * (1 + weights['avg_availability'] * avg_availability_boost) * (1 + weights['experience_years'] * experience_years_boost)

        top_mentor_indices = np.argsort(-final_scores)[:top_k]
        top_mentor_ids = mentor_df.iloc[top_mentor_indices]['Id'].values
        top_scores = final_scores[top_mentor_indices]

        recommendations = []
        for mentor_id, score in zip(top_mentor_ids, top_scores):
            mentor_info = mentor_df[mentor_df['Id'] == mentor_id]
            mentor_details = {
                    'mentor_id': str(mentor_id),  # Ensure string format
                }
            recommendations.append(mentor_details)
        return recommendations
    except Exception as e:
        return str(e)

def recommend_mentors_content_based_skills(m_skills, top_k=6, weights=None):
    if weights is None:
        weights = {
            'skills': 0.4,
            'location': 0.1,
            'avg_rate': 0.25,
            'avg_availability': 0.1,
            'experience_years': 0.15
        }

    try:
        # Validate m_skills
        if not isinstance(m_skills, list):
            raise ValueError("m_skills must be a list")
        if not all(isinstance(skill, str) for skill in m_skills):
            raise ValueError("All m_skills must be strings")
        if not m_skills:
            raise ValueError("m_skills cannot be empty")

        # Transform the input skills into a TF-IDF vector
        skills_text = " ".join(m_skills)
        mentee_skills_vector = tfidf_vectorizer.transform([skills_text]).toarray()

        # Since we don't have a specific mentee_id, use a default or null location (e.g., 0)
        mentee_location = 0  # Assuming location_encoded is 0 for unknown location

        # Compute similarities
        skill_similarities = cosine_similarity(mentee_skills_vector, mentor_skills_matrix).flatten()
        location_similarities = np.zeros_like(skill_similarities)  # Default to 0 for unknown location

        # Use normalized values from mentor_df for boosts
        avg_rate_boost = mentor_df['avg_rate_normalized'].values
        avg_availability_boost = mentor_df['avg_availability_normalized'].values
        experience_years_boost = mentor_df['experience_years_normalized'].values

        # Combine scores
        combined_scores = (
            weights['skills'] * skill_similarities +
            weights['location'] * location_similarities
        )

        final_scores = combined_scores * (1 + weights['avg_rate'] * avg_rate_boost) * (1 + weights['avg_availability'] * avg_availability_boost) * (1 + weights['experience_years'] * experience_years_boost)

        # Get top_k mentors
        top_mentor_indices = np.argsort(-final_scores)[:top_k]
        top_mentor_ids = mentor_df.iloc[top_mentor_indices]['Id'].values
        top_scores = final_scores[top_mentor_indices]

        # Prepare recommendations
        recommendations = []
        for mentor_id, score in zip(top_mentor_ids, top_scores):
            mentor_info = mentor_df[mentor_df['Id'] == mentor_id]
            mentor_details = {
                'mentor_id': str(mentor_id),
            }
            recommendations.append(mentor_details)
        return recommendations

    except Exception as e:
        return str(e)


#### ---> API endpoint
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    # Get the JSON data from the request body
    data = request.get_json()
    print(data)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    mentee_id = data.get('mentee_id')  # Already a string (UUID)
    m_skills = data.get('m_skills')  # Expecting a list of strings
    top_k = 50

    # Validate mentee_id
    if mentee_id is None:
        return jsonify({'error': 'mentee_id is required'}), 400

    # Validate m_skills
    if m_skills is None:
        return jsonify({'error': 'm_skills is required'}), 400
    if not isinstance(m_skills, list):
        return jsonify({'error': 'm_skills must be a list'}), 400
    if not all(isinstance(skill, str) for skill in m_skills):
        return jsonify({'error': 'All m_skills must be strings'}), 400

    try:
        
        at_mentee_df = mentee_id in mentee_df['Id'].values

        if at_mentee_df:
            recommendations = recommend_mentors_content_based(mentee_id, top_k)
            method = "Content-Based (Skills, Location, Avg_rate, Avg_availability, Experience_years)"

        else:
            recommendations = recommend_mentors_content_based_skills(m_skills)
            method = "skils"

        if isinstance(recommendations, str):
            return jsonify({'error': recommendations}), 400

        return jsonify({
            'mentee_id': mentee_id,
            'method': method,
            'recommendations': recommendations
        })
    except Exception as e:

        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
  # Run the Flask app
  port = int(os.getenv("PORT", 5000))
  app.run(host='0.0.0.0', port=port)
