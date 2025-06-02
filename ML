import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import re
import math
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
import json

# Configurazione della pagina
st.set_page_config(
    page_title="Stock Sentiment Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Sentiment Analysis & Price Prediction")
st.markdown("Analizza l'andamento dei titoli REALI e prevedi le variazioni basandoti sul sentiment delle news")

# Sidebar per configurazione
st.sidebar.header("Configurazione")

# Input ticker
ticker = st.sidebar.text_input("Inserisci il ticker (es: AAPL, MSFT, TSLA)", value="AAPL").upper()

# API Key per Alpha Vantage (opzionale)
alpha_vantage_key = st.sidebar.text_input(
    "Alpha Vantage API Key (opzionale)", 
    type="password",
    help="Ottieni una chiave gratuita su alphavantage.co per dati finanziari reali"
)

# Threshold per le variazioni
threshold = st.sidebar.slider("Soglia variazione (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

@st.cache_data(ttl=3600)  # Cache per 1 ora
def get_real_stock_data(ticker, api_key=None):
    """Ottiene dati finanziari reali"""
    
    if api_key:
        # Usa Alpha Vantage per dati reali
        try:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                
                # Converti in DataFrame
                df_data = []
                for date_str, values in time_series.items():
                    df_data.append({
                        'Date': pd.to_datetime(date_str),
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Volume': int(values['5. volume'])
                    })
                
                df = pd.DataFrame(df_data)
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                
                # Prendi ultimi 2 anni
                two_years_ago = datetime.now() - timedelta(days=730)
                df = df[df.index >= two_years_ago]
                
                return df
            else:
                st.warning(f"Errore API Alpha Vantage: {data.get('Note', 'Limite richieste raggiunto')}")
                return None
                
        except Exception as e:
            st.error(f"Errore nel recupero dati da Alpha Vantage: {str(e)}")
            return None
    
    else:
        # Fallback: usa Yahoo Finance API gratuita (non ufficiale)
        try:
            # Calcola timestamp per 2 anni fa
            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(days=730)).timestamp())
            
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
            params = {
                'period1': start_time,
                'period2': end_time,
                'interval': '1d',
                'events': 'history'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                # Leggi CSV direttamente
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                # Rinomina colonne per consistenza
                df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                return df
            else:
                return None
                
        except Exception as e:
            st.error(f"Errore nel recupero dati Yahoo Finance: {str(e)}")
            return None

@st.cache_data(ttl=1800)  # Cache per 30 minuti
def get_news_from_google_rss(ticker, days=30):
    """Ottiene news reali da Google RSS"""
    try:
        # Cerca news per il ticker
        query = f"{ticker} stock OR {ticker} earnings OR {ticker} news"
        encoded_query = quote(query)
        
        # URL Google News RSS
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        news_items = []
        for item in root.findall('.//item')[:50]:  # Primi 50 articoli
            try:
                title = item.find('title').text if item.find('title') is not None else ""
                description = item.find('description').text if item.find('description') is not None else ""
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
                
                # Parse data pubblicazione
                try:
                    if pub_date:
                        pub_datetime = pd.to_datetime(pub_date)
                    else:
                        pub_datetime = datetime.now()
                except:
                    pub_datetime = datetime.now()
                
                # Combina titolo e descrizione
                full_text = f"{title} {description}".strip()
                
                if full_text and len(full_text) > 20:  # Filtra testi troppo corti
                    news_items.append({
                        'date': pub_datetime,
                        'text': full_text,
                        'title': title
                    })
                    
            except Exception as e:
                continue
        
        # Ordina per data
        news_items.sort(key=lambda x: x['date'], reverse=True)
        
        return news_items
        
    except Exception as e:
        st.error(f"Errore nel recupero news: {str(e)}")
        return []

def analyze_sentiment_simple(text):
    """Analizza sentiment usando word-based approach"""
    # Liste di parole positive e negative per il contesto finanziario
    positive_words = {
        'gain', 'gains', 'rise', 'rises', 'rising', 'up', 'surge', 'surges', 'surge', 
        'beat', 'beats', 'beating', 'strong', 'strength', 'growth', 'growing', 'grew',
        'profit', 'profits', 'profitable', 'earnings', 'revenue', 'sales', 'increase',
        'increased', 'increases', 'bull', 'bullish', 'positive', 'optimistic', 'upgrade',
        'upgraded', 'buy', 'recommend', 'target', 'outperform', 'exceed', 'exceeds',
        'breakthrough', 'innovation', 'partnership', 'deal', 'success', 'successful',
        'rally', 'rallies', 'boost', 'boosted', 'improve', 'improved', 'improvement'
    }
    
    negative_words = {
        'fall', 'falls', 'falling', 'fell', 'drop', 'drops', 'dropping', 'dropped',
        'decline', 'declines', 'declining', 'down', 'plunge', 'plunges', 'crash',
        'weak', 'weakness', 'loss', 'losses', 'lose', 'losing', 'lost', 'bear',
        'bearish', 'negative', 'pessimistic', 'downgrade', 'downgraded', 'sell',
        'concern', 'concerns', 'worried', 'worry', 'risk', 'risks', 'challenge',
        'challenges', 'problem', 'problems', 'issue', 'issues', 'struggle', 'struggles',
        'disappointing', 'disappointed', 'miss', 'missed', 'below', 'underperform',
        'recession', 'inflation', 'uncertainty', 'volatile', 'volatility'
    }
    
    # Converti in lowercase e tokenizza
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Conta parole positive e negative
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # Determina sentiment
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

def match_news_to_dates(stock_data, news_items):
    """Associa news alle date dei dati stock"""
    matched_data = []
    
    for date in stock_data.index:
        # Trova news per questa data (±1 giorno)
        date_news = []
        for news in news_items:
            news_date = news['date'].date() if hasattr(news['date'], 'date') else news['date']
            stock_date = date.date() if hasattr(date, 'date') else date
            
            # Cerca news dello stesso giorno o del giorno prima
            if abs((news_date - stock_date).days) <= 1:
                date_news.append(news['text'])
        
        # Se non ci sono news specifiche, usa un campione generale
        if not date_news and news_items:
            date_news = [news_items[0]['text']]  # Usa la news più recente
        
        # Combina tutte le news del giorno
        combined_text = " ".join(date_news) if date_news else f"No specific news for {ticker}"
        sentiment = analyze_sentiment_simple(combined_text)
        
        matched_data.append({
            'date': date,
            'news_text': combined_text[:500],  # Limita lunghezza
            'sentiment': sentiment,
            'news_count': len(date_news)
        })
    
    return matched_data

def calculate_daily_changes(data, threshold):
    """Calcola le variazioni giornaliere e le etichette"""
    data['Daily_Change'] = ((data['Close'] - data['Open']) / data['Open']) * 100
    
    # Crea le etichette basate sulla soglia
    def get_label(change, threshold):
        if change > threshold:
            return "Positive"
        elif change < -threshold:
            return "Negative"
        else:
            return "No-change"
    
    data['Label'] = data['Daily_Change'].apply(lambda x: get_label(x, threshold))
    return data

def simple_text_vectorizer(texts, max_features=200):
    """Semplice vettorizzatore di testo basato su word frequency"""
    # Combina tutti i testi e trova le parole più comuni
    all_words = []
    for text in texts:
        # Semplice tokenizzazione
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        all_words.extend(words)
    
    # Trova le parole più comuni (escluse stopwords comuni)
    stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'with', 'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'}
    
    word_counts = Counter([w for w in all_words if w not in stopwords and len(w) > 2])
    top_words = [word for word, count in word_counts.most_common(max_features)]
    
    # Vettorizza ogni testo
    vectors = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_count = Counter(words)
        
        vector = []
        for word in top_words:
            vector.append(word_count.get(word, 0))
        
        vectors.append(vector)
    
    return np.array(vectors), top_words

def simple_classifier(X_train, y_train, X_test):
    """Classificatore semplice basato su Naive Bayes"""
    classes = list(set(y_train))
    class_probs = {}
    feature_probs = {}
    
    # Calcola probabilità delle classi
    total_samples = len(y_train)
    for cls in classes:
        class_count = sum(1 for y in y_train if y == cls)
        class_probs[cls] = class_count / total_samples
    
    # Calcola probabilità delle feature per classe
    for cls in classes:
        cls_indices = [i for i, y in enumerate(y_train) if y == cls]
        cls_features = X_train[cls_indices]
        
        # Media delle feature per questa classe (con smoothing)
        feature_means = np.mean(cls_features, axis=0) + 0.1
        feature_probs[cls] = feature_means
    
    # Predizioni
    predictions = []
    for sample in X_test:
        scores = {}
        for cls in classes:
            # Log probabilità per evitare underflow
            score = math.log(class_probs[cls])
            for i, feature_val in enumerate(sample):
                if feature_val > 0:  # Solo se la feature è presente
                    score += math.log(feature_probs[cls][i])
            scores[cls] = score
        
        # Scegli la classe con score più alto
        predicted_class = max(scores.keys(), key=lambda x: scores[x])
        predictions.append(predicted_class)
    
    return predictions

def evaluate_model(y_true, y_pred):
    """Valuta le performance del modello"""
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)
    
    # Calcola metriche per classe
    classes = list(set(y_true))
    class_metrics = {}
    
    for cls in classes:
        true_pos = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        false_pos = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
        false_neg = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return accuracy, class_metrics

def create_ml_model(news_data, labels):
    """Crea e addestra un modello di machine learning"""
    if len(set(labels)) < 2:
        return None
    
    # Split train/test (80/20)
    n_train = int(0.8 * len(news_data))
    indices = list(range(len(news_data)))
    np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train_text = [news_data[i] for i in train_indices]
    X_test_text = [news_data[i] for i in test_indices]
    y_train = [labels[i] for i in train_indices]
    y_test = [labels[i] for i in test_indices]
    
    # Vettorizza il testo
    X_combined = X_train_text + X_test_text
    X_vectors, vocab = simple_text_vectorizer(X_combined)
    
    X_train = X_vectors[:len(X_train_text)]
    X_test = X_vectors[len(X_train_text):]
    
    # Addestra classificatore
    predictions = simple_classifier(X_train, y_train, X_test)
    
    # Valuta modello
    accuracy, class_metrics = evaluate_model(y_test, predictions)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': y_test,
        'class_metrics': class_metrics,
        'vocabulary': vocab[:20]  # Top 20 parole più importanti
    }

def main():
    if st.sidebar.button("🚀 Avvia Analisi Reale", type="primary"):
        if not ticker:
            st.error("Inserisci un ticker valido!")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Ottieni dati storici REALI
        status_text.text("📊 Recupero dati finanziari reali...")
        progress_bar.progress(20)
        
        stock_data = get_real_stock_data(ticker, alpha_vantage_key)
        if stock_data is None or stock_data.empty:
            st.error(f"❌ Impossibile recuperare dati reali per {ticker}. Verifica il ticker o prova più tardi.")
            return
        
        st.success(f"✅ Recuperati {len(stock_data)} giorni di dati reali per {ticker}")
        
        # 2. Calcola variazioni e etichette
        status_text.text("📈 Calcolo variazioni giornaliere...")
        progress_bar.progress(40)
        
        stock_data = calculate_daily_changes(stock_data, threshold)
        
        # 3. Ottieni news REALI
        status_text.text("📰 Recupero news reali da Google RSS...")
        progress_bar.progress(60)
        
        news_items = get_news_from_google_rss(ticker)
        if not news_items:
            st.warning("⚠️ Nessuna news trovata. L'analisi continuerà con dati limitati.")
            return
        
        st.success(f"✅ Recuperate {len(news_items)} news reali per {ticker}")
        
        # 4. Associa news alle date
        status_text.text("🔗 Associazione news alle date...")
        progress_bar.progress(70)
        
        matched_news = match_news_to_dates(stock_data, news_items)
        
        # Aggiungi news al dataframe
        for i, news_data in enumerate(matched_news):
            if i < len(stock_data):
                stock_data.iloc[i, stock_data.columns.get_loc('Label')] = stock_data.iloc[i]['Label']
        
        # Crea DataFrame per ML con ultimi dati disponibili
        recent_data = stock_data.tail(min(200, len(matched_news))).copy()
        
        news_texts = [item['news_text'] for item in matched_news[-len(recent_data):]]
        sentiments = [item['sentiment'] for item in matched_news[-len(recent_data):]]
        
        recent_data['News'] = news_texts
        recent_data['News_Sentiment'] = sentiments
        
        # 5. Training ML
        status_text.text("🤖 Training modello con dati reali...")
        progress_bar.progress(85)
        
        if len(news_texts) > 10:
            ml_results = create_ml_model(news_texts, recent_data['Label'].tolist())
        else:
            ml_results = None
        
        progress_bar.progress(100)
        status_text.text("✅ Analisi con dati reali completata!")
        
        # 6. Visualizza risultati
        st.success(f"🎉 Analisi completata per {ticker} con dati e news REALI!")
        
        # Info sui dati
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Giorni di dati", len(stock_data))
        
        with col2:
            positive_days = len(stock_data[stock_data['Label'] == 'Positive'])
            st.metric("📈 Giorni positivi", positive_days)
        
        with col3:
            negative_days = len(stock_data[stock_data['Label'] == 'Negative'])
            st.metric("📉 Giorni negativi", negative_days)
        
        with col4:
            st.metric("📰 News trovate", len(news_items))
        
        # Info sul periodo
        start_date = stock_data.index.min().strftime('%Y-%m-%d')
        end_date = stock_data.index.max().strftime('%Y-%m-%d')
        avg_change = stock_data['Daily_Change'].mean()
        
        st.info(f"📅 **Periodo analizzato**: {start_date} → {end_date} | **Variazione media**: {avg_change:.2f}%")
        
        # Grafici
        st.subheader("📈 Andamento del Prezzo Reale")
        
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['Close'],
            mode='lines',
            name='Prezzo di Chiusura',
            line=dict(color='blue', width=2)
        ))
        
        fig_price.update_layout(
            title=f"Andamento reale del prezzo di {ticker}",
            xaxis_title="Data",
            yaxis_title="Prezzo ($)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Distribuzione delle etichette
        st.subheader("📊 Distribuzione delle Variazioni Reali")
        
        col1, col2 = st.columns(2)
        
        with col1:
            label_counts = stock_data['Label'].value_counts()
            
            fig_pie = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title=f"Distribuzione variazioni reali - {ticker}",
                color_discrete_map={
                    'Positive': '#00CC44',
                    'Negative': '#FF4444',
                    'No-change': '#888888'
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Sentiment distribution
            if 'News_Sentiment' in recent_data.columns:
                sentiment_counts = recent_data['News_Sentiment'].value_counts()
                
                fig_sentiment = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title=f"Distribuzione sentiment news - {ticker}",
                    color_discrete_map={
                        'Positive': '#00CC44',
                        'Negative': '#FF4444',
                        'Neutral': '#888888'
                    }
                )
                fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Risultati ML
        if ml_results:
            st.subheader("🤖 Risultati del Modello con Dati Reali")
            
            accuracy = ml_results['accuracy']
            
            st.success(f"🏆 **Accuratezza con dati reali: {accuracy:.1%}**")
            
            # Interpretazione risultati
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if accuracy > 0.65:
                    st.success(f"""
                    🎉 **Risultati eccellenti con dati reali!** 
                    
                    Il modello mostra un'accuratezza del **{accuracy:.1%}** usando news e prezzi reali, 
                    indicando una **correlazione significativa** tra sentiment delle news e variazioni di {ticker}.
                    
                    ✅ Le notizie reali possono essere un **predittore affidabile** per gli andamenti del titolo.
                    """)
                elif accuracy > 0.45:
                    st.warning(f"""
                    ⚠️ **Risultati moderati con dati reali**
                    
                    Il modello ha un'accuratezza del **{accuracy:.1%}** con dati reali, 
                    suggerendo una **correlazione parziale** tra sentiment e variazioni di prezzo.
                    
                    📊 Le news hanno un impatto, ma altri fattori di mercato sono ugualmente influenti.
                    """)
                else:
                    st.info(f"""
                    📊 **Correlazione limitata nei dati reali**
                    
                    L'accuratezza del **{accuracy:.1%}** indica che il sentiment delle news 
                    ha un **impatto limitato** sulle variazioni di {ticker}.
                    
                    💡 I mercati reali sono complessi e influenzati da molti fattori oltre alle news.
                    """)
            
            with col2:
                # Gauge chart per l'accuratezza
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = accuracy * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Accuratezza %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 45], 'color': "lightgray"},
                            {'range': [45, 65], 'color': "yellow"},
                            {'range': [65, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Analisi correlazione
            if 'News_Sentiment' in recent_data.columns:
                st.subheader("📊 Correlazione Sentiment vs Performance (Dati Reali)")
                
                sentiment_performance = pd.crosstab(recent_data['News_Sentiment'], recent_data['Label'])
                
                fig_heatmap = px.imshow(
                    sentiment_performance.values,
                    labels=dict(x="Variazione Prezzo", y="Sentiment News", color="Frequenza"),
                    x=sentiment_performance.columns,
                    y=sentiment_performance.index,
                    title="Correlazione reale tra Sentiment delle News e Variazioni di Prezzo",
                    color_continuous_scale="Blues",
                    text_auto=True
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        else:
            st.warning("⚠️ Dati insufficienti per il training del modello ML.")
        
        # News recenti reali
        st.subheader("📰 News Reali Recenti")
        
        # Mostra le ultime 5 news con sentiment
        recent_news = news_items[:5]
        for i, news in enumerate(recent_news):
            sentiment = analyze_sentiment_simple(news['text'])
            sentiment_color = {'Positive': '🟢', 'Negative': '🔴', 'Neutral': '🟡'}
            
            with st.expander(f"{sentiment_color.get(sentiment, '🟡')} {news['title'][:100]}..."):
                st.write(f"**Data**: {news['date'].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Sentiment**: {sentiment}")
                st.write(f"**Testo**: {news['text'][:500]}...")
        
        # Dati recenti
        st.subheader("📋 Dati Finanziari Recenti")
        
        display_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Change', 'Label']].tail(10)
        display_data['Daily_Change'] = display_data['Daily_Change'].round(2)
        display_data.columns = ['Apertura', 'Massimo', 'Minimo', 'Chiusura', 'Volume', 'Variazione %', 'Etichetta']
        
        st.dataframe(display_data, use_container_width=True)

# Sidebar con informazioni
st.sidebar.markdown("""
---
### 📋 Dati Reali Utilizzati

**📊 Dati Finanziari:**
- **Con API Key**: Alpha Vantage (dati ufficiali)
- **Senza API Key**: Yahoo Finance (gratuito)
- **Periodo**: Ultimi 2 anni
- **Frequenza**: Dati giornalieri OHLC

**📰 News:**
- **Fonte**: Google News RSS (gratuito)
- **Tipo**: News reali correlate al ticker
- **Sentiment**: Analisi basata su parole chiave finanziarie
- **Aggiornamento**: In tempo reale

### 🔑 API Keys (Opzionali)

**Alpha Vantage** (Consigliata):
- Registrati su [alphavantage.co](https://alphavantage.co)
- 25 richieste gratuite al giorno
- Dati di alta qualità

### 🎯 Interpretazione Risultati

**Con Dati Reali:**
- **🟢 > 65%**: Forte correlazione sentiment-prezzi
- **🟡 45-65%**: Correlazione moderata, fattori multipli
- **⚪ < 45%**: Correlazione debole, mercati complessi

### ⚠️ Disclaimer

- **Dati reali** ma **solo per scopi educativi**
- I mercati sono influenzati da molti fattori
- **Non utilizzare** per decisioni di investimento
- Le performance passate non garantiscono risultati futuri

### 🔄 Limiti e Note

- Google RSS può avere limitazioni temporali
- Yahoo Finance è un'API non ufficiale
- Il sentiment è basato su analisi testuale semplice
- Risultati migliori con API keys ufficiali
""")

if __name__ == "__main__":
    main()
