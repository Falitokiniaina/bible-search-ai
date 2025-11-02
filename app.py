#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Web pour la recherche biblique avec synthèse IA
Utilise Flask pour le backend et une interface moderne en HTML/CSS/JS
Version avec Qdrant Cloud
"""

from flask import Flask, render_template, request, jsonify
import cohere
from qdrant_client import QdrantClient
from deep_translator import GoogleTranslator
import os

# Configuration
COHERE_API_KEY = "wmJt4hjoKPb73nYu6aYs0juZ837vlurSxaRVc5I0"
COHERE_EMBEDDING_MODEL = "embed-multilingual-v3.0"
COHERE_GENERATION_MODEL = "command-r-08-2024"

# Configuration Qdrant Cloud
QDRANT_URL = os.environ.get('QDRANT_URL',
                            'https://93e9e040-840e-4a40-bab0-4d3074ccea48.europe-west3-0.gcp.cloud.qdrant.io')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY',
                                'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-Byh88ThNz3_9eHw_guu2TPzfbWe56DVSnl8nVAFrT8')
COLLECTION_NAME = "bible_verses"

# Initialiser Flask
app = Flask(__name__, static_folder='static', static_url_path='')

# Initialiser les clients (globaux pour éviter de les recréer à chaque requête)
co = cohere.Client(COHERE_API_KEY)
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


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
        max_tokens=800  # Réduit de 1000 à 800 pour accélérer
    )

    return response.text


def translate_to_malagasy(french_text):
    """Traduit la synthèse en malgache"""
    try:
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
            max_tokens=1200  # Réduit de 1500 à 1200
        )

        return response.text

    except Exception as e:
        print(f"❌ Erreur traduction malgache: {e}")
        return f"Erreur de traduction: {str(e)}"


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
        print(f"🔍 Recherche: '{query}' (top_k={top_k})")
        results = search_verses(query, top_k)

        if not results:
            return jsonify({'error': 'Aucun verset trouvé pour cette recherche'}), 404

        print(f"✅ {len(results)} versets trouvés")

        # Formater les versets
        verses = []
        for result in results:
            verses.append({
                'reference': result.payload['reference'],
                'text': result.payload['text'],
                'score': round(result.score, 4)
            })

        # Générer la synthèse en français
        print("🔄 Génération de la synthèse française...")
        synthesis = generate_synthesis(query, results)
        print(f"✅ Synthèse française générée ({len(synthesis)} caractères)")

        # Ne pas traduire immédiatement (sera fait dans une 2ème requête)
        return jsonify({
            'success': True,
            'query': query,
            'synthesis': synthesis,
            'synthesis_malagasy': None,  # Sera rempli par /translate
            'verses': verses if show_verses else [],
            'total_verses': len(verses)
        })

    except Exception as e:
        print(f"❌ Erreur dans /search: {e}")
        return jsonify({'error': f'Erreur: {str(e)}'}), 500


@app.route('/translate', methods=['POST'])
def translate():
    """Endpoint pour la traduction en malgache (séparé pour éviter timeout)"""
    try:
        data = request.json
        french_text = data.get('text', '').strip()

        if not french_text:
            return jsonify({'error': 'Texte manquant'}), 400

        print("🔄 Traduction en malgache...")
        synthesis_malagasy = translate_to_malagasy(french_text)
        print(f"✅ Traduction générée ({len(synthesis_malagasy)} caractères)")

        return jsonify({
            'success': True,
            'translation': synthesis_malagasy
        })

    except Exception as e:
        print(f"❌ Erreur traduction: {e}")
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
            'points_count': collection_info.points_count,
            'qdrant_url': QDRANT_URL.split('@')[-1] if '@' in QDRANT_URL else QDRANT_URL
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/robots.txt')
def robots():
    """Serve robots.txt for SEO"""
    return """User-agent: *
Allow: /
Disallow: /static/
Disallow: /admin/

Sitemap: https://bible-search-ai.onrender.com/sitemap.xml

Crawl-delay: 1

User-agent: Googlebot
Allow: /

User-agent: Bingbot
Allow: /

User-agent: Slurp
Allow: /""", 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.route('/sitemap.xml')
def sitemap():
    """Serve sitemap.xml for SEO"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://bible-search-ai.onrender.com/</loc>
        <lastmod>2025-11-01</lastmod>
        <changefreq>daily</changefreq>
        <priority>1.0</priority>
    </url>
    <url>
        <loc>https://bible-search-ai.onrender.com/health</loc>
        <lastmod>2025-11-01</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.3</priority>
    </url>
</urlset>""", 200, {'Content-Type': 'application/xml; charset=utf-8'}

@app.route('/BingSiteAuth.xml')
def bing_verification():
    """Bing Webmaster Tools verification file"""
    # Copiez le contenu exact du fichier que Bing vous a fourni
    xml_content = """<?xml version="1.0"?>
<users>
    <user>C3C8E7B19260FF3B7227C09FD8578406</user>
</users>"""
    return xml_content, 200, {'Content-Type': 'application/xml; charset=utf-8'}

@app.route('/googlecdf45738476f5de6.html')
def google_verification():
    """Google Search Console verification file"""
    return "google-site-verification: googlecdf45738476f5de6.html", 200, {'Content-Type': 'text/html; charset=utf-8'}


if __name__ == '__main__':
    # Créer le dossier templates s'il n'existe pas
    os.makedirs('templates', exist_ok=True)

    # Récupérer le port
    port = int(os.environ.get('PORT', 5000))

    # Afficher les informations de démarrage
    print("=" * 80)
    print("🌐 Interface Web - Recherche Biblique")
    print("=" * 80)
    print(f"📍 Serveur démarré sur http://0.0.0.0:{port}")
    print(f"📊 Collection Qdrant: {COLLECTION_NAME}")
    print(f"☁️  Qdrant Cloud: {QDRANT_URL[:50]}...")
    print("=" * 80)

    # Lancer l'application
    app.run(host='0.0.0.0', port=port)