#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Web pour la recherche biblique avec synthèse IA
Utilise Flask pour le backend et une interface moderne en HTML/CSS/JS
"""

from flask import Flask, render_template, request, jsonify
import cohere
from qdrant_client import QdrantClient
import os

# Configuration
COHERE_API_KEY = "wmJt4hjoKPb73nYu6aYs0juZ837vlurSxaRVc5I0"
COHERE_EMBEDDING_MODEL = "embed-multilingual-v3.0"
COHERE_GENERATION_MODEL = "command-r-08-2024"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "bible_verses"

# Initialiser Flask
app = Flask(__name__)

# Initialiser les clients (globaux pour éviter de les recréer à chaque requête)
co = cohere.Client(COHERE_API_KEY)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def search_verses(query, top_k=10):
    """Recherche des versets pertinents"""
    # Créer l'embedding de la requête
    query_embedding = co.embed(
        texts=[query],
        model=COHERE_EMBEDDING_MODEL,
        input_type="search_query"
    ).embeddings[0]

    # Rechercher dans Qdrant
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )

    return results


def format_results_for_synthesis(results):
    """Formate les résultats pour la synthèse IA"""
    context = ""
    for i, result in enumerate(results, 1):
        context += f"{i}. {result.payload['reference']}: {result.payload['text']}\n\n"
    return context


def generate_synthesis(query, results):
    """Génère une synthèse basée sur les versets trouvés"""
    context = format_results_for_synthesis(results)

    prompt = f"""Tu es un assistant biblique expert. Un utilisateur recherche des informations sur : "{query}"

Voici les versets bibliques pertinents trouvés :

{context}

En te basant uniquement sur ces versets, rédige une synthèse claire et structurée qui :
1. Répond à la question ou au thème recherché
2. Cite les références bibliques appropriées (livre chapitre:verset)
3. Organise les informations de manière cohérente
4. Reste fidèle au texte biblique sans ajouter d'interprétations personnelles

Synthèse :"""

    response = co.chat(
        model=COHERE_GENERATION_MODEL,
        message=prompt,
        temperature=0.3,
        max_tokens=1000
    )

    return response.text


def translate_to_malagasy(french_text):
    """Traduit la synthèse en malgache"""
    prompt = f"""Tu es un traducteur expert français-malgache.

Traduis le texte suivant en malgache (langue de Madagascar).
Garde les références bibliques en format original (exemple: Genèse 1:1).
Assure-toi que la traduction soit naturelle et fidèle au sens original.

Texte en français :
{french_text}

Traduction en malgache :"""

    response = co.chat(
        model=COHERE_GENERATION_MODEL,
        message=prompt,
        temperature=0.3,
        max_tokens=1000
    )

    return response.text


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """Endpoint pour la recherche"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        top_k = int(data.get('top_k', 10))
        show_verses = data.get('show_verses', True)

        if not query:
            return jsonify({'error': 'Veuillez entrer une recherche'}), 400

        # Rechercher les versets
        results = search_verses(query, top_k)

        if not results:
            return jsonify({'error': 'Aucun verset trouvé pour cette recherche'}), 404

        # Formater les versets
        verses = []
        for result in results:
            verses.append({
                'reference': result.payload['reference'],
                'text': result.payload['text'],
                'score': round(result.score, 4)
            })

        # Générer la synthèse
        synthesis = generate_synthesis(query, results)

        return jsonify({
            'success': True,
            'query': query,
            'synthesis': synthesis,
            'verses': verses if show_verses else [],
            'total_verses': len(verses)
        })

    except Exception as e:
        return jsonify({'error': f'Erreur: {str(e)}'}), 500


@app.route('/health')
def health():
    """Endpoint de santé pour vérifier que l'API fonctionne"""
    try:
        # Vérifier la connexion à Qdrant
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        return jsonify({
            'status': 'ok',
            'collection': COLLECTION_NAME,
            'points_count': collection_info.points_count
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # Créer le dossier templates s'il n'existe pas
    os.makedirs('templates', exist_ok=True)

    print("=" * 80)
    print("🌐 Interface Web - Recherche Biblique")
    print("=" * 80)
    print(f"📍 Serveur démarré sur http://localhost:5000")
    print(f"📊 Collection Qdrant: {COLLECTION_NAME}")
    print("=" * 80)

    app.run(debug=True, host='0.0.0.0', port=5000)