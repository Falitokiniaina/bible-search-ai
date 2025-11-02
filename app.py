#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Web pour la recherche biblique avec synthèse IA
Utilise Flask pour le backend et une interface moderne en HTML/CSS/JS
Version avec Qdrant Cloud + Analytics + Admin Dashboard
"""

from flask import Flask, render_template, request, jsonify
import cohere
from qdrant_client import QdrantClient
from deep_translator import GoogleTranslator
import os
from datetime import datetime
import json

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

# Fichier de stats (simple JSON)
STATS_FILE = 'stats.json'


def load_stats():
    """Charge les statistiques depuis le fichier"""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {'searches': [], 'translations': 0, 'total_searches': 0}
    return {'searches': [], 'translations': 0, 'total_searches': 0}


def save_stats(stats):
    """Sauvegarde les statistiques"""
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Erreur sauvegarde stats: {e}")


def log_search(query, results_count, ip_address=None):
    """Enregistre une recherche"""
    stats = load_stats()
    stats['total_searches'] = stats.get('total_searches', 0) + 1

    search_log = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'results_count': results_count,
        'ip': ip_address or 'unknown'
    }

    # Garder seulement les 1000 dernières recherches
    stats['searches'].append(search_log)
    if len(stats['searches']) > 1000:
        stats['searches'] = stats['searches'][-1000:]

    save_stats(stats)


def log_translation():
    """Enregistre une traduction"""
    stats = load_stats()
    stats['translations'] = stats.get('translations', 0) + 1
    save_stats(stats)


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
        max_tokens=800
    )

    return response.text


def translate_to_malagasy(french_text):
    """Traduit la synthèse en malgache via Google Translate (deep-translator)"""
    try:
        print("🔄 Traduction via Google Translate (deep-translator)...")

        # Limiter la longueur pour éviter les timeouts
        max_length = 4500

        if len(french_text) > max_length:
            print(f"⚠️ Texte trop long ({len(french_text)} car), découpage...")
            chunks = []
            words = french_text.split()
            current_chunk = []
            current_length = 0

            for word in words:
                word_length = len(word) + 1
                if current_length + word_length > max_length:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    current_chunk.append(word)
                    current_length += word_length

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            # Traduire chaque morceau
            translated_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"   Traduction morceau {i + 1}/{len(chunks)}...")
                translated = GoogleTranslator(source='fr', target='mg').translate(chunk)
                translated_chunks.append(translated)

            result = ' '.join(translated_chunks)
        else:
            result = GoogleTranslator(source='fr', target='mg').translate(french_text)

        print(f"✅ Traduction réussie ({len(result)} caractères)")
        return result

    except Exception as e:
        print(f"❌ Erreur traduction: {e}")
        return f"Traduction indisponible: {str(e)[:100]}..."


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

        # Logger la recherche
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
        log_search(query, len(verses), ip_address)

        # Ne pas traduire immédiatement (sera fait dans une 2ème requête)
        return jsonify({
            'success': True,
            'query': query,
            'synthesis': synthesis,
            'synthesis_malagasy': None,
            'verses': verses if show_verses else [],
            'total_verses': len(verses)
        })

    except Exception as e:
        print(f"❌ Erreur dans /search: {e}")
        return jsonify({'error': f'Erreur: {str(e)}'}), 500


@app.route('/translate', methods=['POST'])
def translate():
    """Endpoint pour la traduction en malgache"""
    try:
        data = request.json
        french_text = data.get('text', '').strip()

        if not french_text:
            return jsonify({'error': 'Texte manquant'}), 400

        print("🔄 Traduction en malgache...")
        synthesis_malagasy = translate_to_malagasy(french_text)
        print(f"✅ Traduction générée ({len(synthesis_malagasy)} caractères)")

        # Logger la traduction
        log_translation()

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
    xml_content = """<?xml version="1.0"?>
<users>
    <user>C3C8E7B19260FF3B7227C09FD8578406</user>
</users>"""
    return xml_content, 200, {'Content-Type': 'application/xml; charset=utf-8'}


@app.route('/googlecdf45738476f5de6.html')
def google_verification():
    """Google Search Console verification file"""
    return "google-site-verification: googlecdf45738476f5de6.html", 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/admin/stats')
def admin_stats():
    """Page admin des statistiques - PROTÉGÉE PAR MOT DE PASSE"""
    # Vérifier l'authentification basique
    auth = request.authorization
    if not auth or auth.username != 'admin' or auth.password != 'BibleAI2025!':
        return ('Authentification requise', 401, {
            'WWW-Authenticate': 'Basic realm="Admin Stats"'
        })

    stats = load_stats()

    # Calculer des statistiques
    from collections import Counter
    from datetime import datetime, timedelta

    # Top recherches
    search_queries = [s['query'] for s in stats['searches']]
    top_searches = Counter(search_queries).most_common(10)

    # Recherches aujourd'hui
    today = datetime.now().date()
    searches_today = [s for s in stats['searches']
                      if datetime.fromisoformat(s['timestamp']).date() == today]

    # Recherches 7 derniers jours
    week_ago = datetime.now() - timedelta(days=7)
    searches_week = [s for s in stats['searches']
                     if datetime.fromisoformat(s['timestamp']) >= week_ago]

    # Créer HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Admin - Statistiques Bible Search AI</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                margin: 0;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 40px;
                color: #333;
            }}
            h1 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 15px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
            }}
            .stat-number {{
                font-size: 3em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .stat-label {{
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .top-searches {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            .search-item {{
                padding: 10px;
                border-bottom: 1px solid #dee2e6;
                display: flex;
                justify-content: space-between;
            }}
            .recent-searches {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                max-height: 400px;
                overflow-y: auto;
            }}
            .recent-item {{
                padding: 8px;
                border-bottom: 1px solid #dee2e6;
                font-size: 0.9em;
            }}
            .timestamp {{
                color: #666;
                font-size: 0.85em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔐 Tableau de bord Admin - Bible Search AI</h1>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">📊 Total Recherches</div>
                    <div class="stat-number">{stats['total_searches']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">🔍 Aujourd'hui</div>
                    <div class="stat-number">{len(searches_today)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">📅 Cette semaine</div>
                    <div class="stat-number">{len(searches_week)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">🇲🇬 Traductions</div>
                    <div class="stat-number">{stats.get('translations', 0)}</div>
                </div>
            </div>

            <h2>🔥 Top 10 des recherches</h2>
            <div class="top-searches">
                {''.join([f'<div class="search-item"><span>"{query}"</span><span><strong>{count}</strong> fois</span></div>'
                          for query, count in top_searches]) or '<p>Aucune recherche pour le moment</p>'}
            </div>

            <h2>📝 Dernières recherches (50)</h2>
            <div class="recent-searches">
                {''.join([f'<div class="recent-item"><strong>"{s["query"]}"</strong> <span class="timestamp">({datetime.fromisoformat(s["timestamp"]).strftime("%d/%m/%Y %H:%M")})</span></div>'
                          for s in reversed(stats['searches'][-50:])]) or '<p>Aucune recherche pour le moment</p>'}
            </div>
        </div>
    </body>
    </html>
    """

    return html


if __name__ == '__main__':
    # Créer le dossier templates s'il n'existe pas
    os.makedirs('templates', exist_ok=True)

    # Récupérer le port
    port = int(os.environ.get('PORT', 5000))

    # Afficher les informations de démarrage
    print("=" * 80)
    print("🌐 Interface Web - Recherche Biblique + Admin Dashboard")
    print("=" * 80)
    print(f"📍 Serveur démarré sur http://0.0.0.0:{port}")
    print(f"📊 Collection Qdrant: {COLLECTION_NAME}")
    print(f"☁️  Qdrant Cloud: {QDRANT_URL[:50]}...")
    print(f"🔐 Admin: https://bible-search-ai.onrender.com/admin/stats")
    print(f"   User: admin | Pass: BibleAI2025!")
    print("=" * 80)

    # Lancer l'application
    app.run(host='0.0.0.0', port=port)