import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import feedparser
import requests
from bs4 import BeautifulSoup
import time
import warnings
warnings.filterwarnings('ignore')

# Configurazione della pagina
st.set_page_config(
    page_title="Financial Sentiment Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Analizzatore Predittivo di Titoli Finanziari")
st.markdown("*Analisi sentiment delle news e predizione degli andamenti azionari*")

# Sidebar per input utente
st.sidebar.header("Configurazione")
ticker = st.sidebar.text_input("Inserisci il ticker (es. AAPL, MSFT):", value="AAPL").upper()
threshold = st.sidebar.slider("Soglia di variazione (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

if st.sidebar.button("Avvia Analisi"):
    if ticker:
        # Fase 1: Raccolta dati finanziari
        st.header("📊 Dati Storici del Titolo")
        
        with st.spinner("Scaricamento dati finanziari..."):
            # Calcola le date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 anni
            
            try:
                # Scarica i dati con yfinance
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if stock_data.empty:
                    st.error(f"Nessun dato trovato per il ticker {ticker}")
                    st.stop()
                
                # Calcola le variazioni giornaliere
                stock_data['Daily_Change'] = stock_data['Close'].pct_change() * 100
                stock_data['Price_Direction'] = stock_data['Daily_Change'].apply(
                    lambda x: 'Positive' if x > threshold else ('Negative' if x < -threshold else 'No-change')
                )
                
                # Rimuovi i valori NaN
                stock_data = stock_data.dropna()
                
                # Mostra statistiche
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Giorni analizzati", len(stock_data))
                with col2:
                    st.metric("Variazione media", f"{stock_data['Daily_Change'].mean():.2f}%")
                with col3:
                    positive_days = len(stock_data[stock_data['Price_Direction'] == 'Positive'])
                    st.metric("Giorni positivi", positive_days)
                with col4:
                    negative_days = len(stock_data[stock_data['Price_Direction'] == 'Negative'])
                    st.metric("Giorni negativi", negative_days)
                
                # Grafico del prezzo
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                ax1.plot(stock_data.index, stock_data['Close'], linewidth=2)
                ax1.set_title(f'Prezzo di chiusura - {ticker}')
                ax1.set_ylabel('Prezzo ($)')
                ax1.grid(True, alpha=0.3)
                
                ax2.bar(stock_data.index, stock_data['Daily_Change'], 
                       color=['green' if x > threshold else 'red' if x < -threshold else 'gray' 
                              for x in stock_data['Daily_Change']])
                ax2.set_title('Variazioni giornaliere')
                ax2.set_ylabel('Variazione (%)')
                ax2.axhline(y=threshold, color='green', linestyle='--', alpha=0.7)
                ax2.axhline(y=-threshold, color='red', linestyle='--', alpha=0.7)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Distribuzione delle direzioni
                direction_counts = stock_data['Price_Direction'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['green', 'red', 'gray']
                ax.pie(direction_counts.values, labels=direction_counts.index, autopct='%1.1f%%', colors=colors)
                ax.set_title('Distribuzione delle direzioni di prezzo')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Errore nel download dei dati: {str(e)}")
                st.stop()
        
        # Fase 2: Raccolta news
        st.header("📰 Raccolta News")
        
        def get_google_news_rss(ticker, days_back=730):
            """Raccoglie news da Google News RSS"""
            news_data = []
            
            # URL per Google News RSS con ricerca per ticker
            rss_url = f"https://news.google.com/rss/search?q={ticker}&hl=it&gl=IT&ceid=IT:it"
            
            try:
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries:
                    try:
                        # Estrai la data
                        pub_date = datetime(*entry.published_parsed[:6])
                        
                        # Filtra per gli ultimi 2 anni
                        if pub_date >= start_date:
                            news_data.append({
                                'date': pub_date.date(),
                                'title': entry.title,
                                'summary': entry.get('summary', ''),
                                'link': entry.link
                            })
                    except:
                        continue
                        
            except Exception as e:
                st.warning(f"Errore nella raccolta news RSS: {str(e)}")
            
            return news_data
        
        def simple_sentiment_analysis(text):
            """Analisi sentiment semplificata"""
            from textblob import TextBlob
            
            # Parole chiave positive e negative per il contesto finanziario
            positive_words = ['crescita', 'aumento', 'profitto', 'guadagno', 'successo', 'miglioramento', 
                            'positivo', 'rialzo', 'boom', 'record', 'ottimismo', 'growth', 'profit', 
                            'gain', 'success', 'positive', 'rise', 'bull', 'optimism']
            
            negative_words = ['perdita', 'calo', 'diminuzione', 'crisi', 'fallimento', 'negativo',
                            'ribasso', 'crollo', 'pessimismo', 'rischio', 'loss', 'decline', 'fall',
                            'crisis', 'failure', 'negative', 'bear', 'pessimism', 'risk']
            
            text_lower = text.lower()
            
            positive_score = sum(1 for word in positive_words if word in text_lower)
            negative_score = sum(1 for word in negative_words if word in text_lower)
            
            # Usa anche TextBlob come supporto
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Combina i punteggi
            final_score = (positive_score - negative_score) + polarity
            
            if final_score > 0.1:
                return 'Positive'
            elif final_score < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        
        with st.spinner("Raccolta news in corso..."):
            news_data = get_google_news_rss(ticker)
            
            if not news_data:
                st.warning("Nessuna news trovata. Procedo con dati simulati per la demo.")
                # Crea dati simulati per la demo
                news_data = []
                for i, date in enumerate(stock_data.index[-100:]):  # Ultime 100 date
                    sentiment = np.random.choice(['Positive', 'Negative', 'Neutral'], 
                                               p=[0.4, 0.3, 0.3])
                    news_data.append({
                        'date': date.date(),
                        'title': f"News simulata {i+1} per {ticker}",
                        'summary': f"Contenuto simulato con sentiment {sentiment}",
                        'sentiment': sentiment
                    })
            else:
                # Analizza sentiment delle news reali
                for news in news_data:
                    full_text = f"{news['title']} {news['summary']}"
                    news['sentiment'] = simple_sentiment_analysis(full_text)
            
            # Converti in DataFrame
            news_df = pd.DataFrame(news_data)
            news_df['date'] = pd.to_datetime(news_df['date'])
            
            st.success(f"Raccolte {len(news_df)} news")
            
            # Mostra alcune news
            if len(news_df) > 0:
                st.subheader("Esempi di News Raccolte")
                sample_news = news_df.head(5)
                for _, news in sample_news.iterrows():
                    with st.expander(f"{news['date'].strftime('%Y-%m-%d')} - {news['title'][:100]}..."):
                        st.write(f"**Sentiment:** {news['sentiment']}")
                        st.write(f"**Sommario:** {news['summary'][:300]}...")
        
        # Fase 3: Preparazione dati per ML
        st.header("🤖 Modello di Machine Learning")
        
        with st.spinner("Preparazione dati e training del modello..."):
            # Allinea i dati
            stock_data_reset = stock_data.reset_index()
            stock_data_reset['date'] = stock_data_reset['Date'].dt.date
            
            # Merge dei dati
            if len(news_df) > 0:
                # Aggrega news per data
                daily_sentiment = news_df.groupby('date')['sentiment'].apply(
                    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Neutral'
                ).reset_index()
                
                # Merge con dati finanziari
                merged_data = pd.merge(stock_data_reset, daily_sentiment, on='date', how='left')
                merged_data['sentiment'] = merged_data['sentiment'].fillna('Neutral')
            else:
                # Se non ci sono news, usa sentiment casuale
                merged_data = stock_data_reset.copy()
                merged_data['sentiment'] = np.random.choice(['Positive', 'Negative', 'Neutral'], 
                                                          size=len(merged_data), p=[0.4, 0.3, 0.3])
            
            # Prepara features per ML
            # Encoding del sentiment
            sentiment_mapping = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
            merged_data['sentiment_encoded'] = merged_data['sentiment'].map(sentiment_mapping)
            
            # Features tecniche
            merged_data['prev_close'] = merged_data['Close'].shift(1)
            merged_data['price_change'] = merged_data['Close'] - merged_data['prev_close']
            merged_data['volume_ma'] = merged_data['Volume'].rolling(window=5).mean()
            merged_data['price_ma'] = merged_data['Close'].rolling(window=5).mean()
            
            # Rimuovi NaN
            ml_data = merged_data.dropna()
            
            if len(ml_data) < 50:
                st.error("Dati insufficienti per il training del modello")
                st.stop()
            
            # Prepara X e y
            features = ['sentiment_encoded', 'prev_close', 'Volume', 'volume_ma', 'price_ma']
            X = ml_data[features]
            y = ml_data['Price_Direction']
            
            # Split dei dati
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Training di diversi modelli
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True)
            }
            
            results = {}
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred
                }
            
            # Trova il modello migliore
            best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            best_model = results[best_model_name]['model']
            best_accuracy = results[best_model_name]['accuracy']
            
            st.success(f"Training completato!")
        
        # Fase 4: Risultati
        st.header("📊 Risultati dell'Analisi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance dei Modelli")
            performance_df = pd.DataFrame({
                'Modello': list(results.keys()),
                'Accuratezza': [results[name]['accuracy'] for name in results.keys()]
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(performance_df['Modello'], performance_df['Accuratezza'], 
                         color=['gold' if name == best_model_name else 'lightblue' 
                               for name in performance_df['Modello']])
            ax.set_title('Accuratezza dei Modelli')
            ax.set_ylabel('Accuratezza')
            ax.set_ylim(0, 1)
            
            # Aggiungi valori sulle barre
            for bar, acc in zip(bars, performance_df['Accuratezza']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Matrice di Confusione - Modello Migliore")
            y_pred_best = results[best_model_name]['predictions']
            cm = confusion_matrix(y_test, y_pred_best)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=best_model.classes_, 
                       yticklabels=best_model.classes_, ax=ax)
            ax.set_title(f'Matrice di Confusione - {best_model_name}')
            ax.set_ylabel('Valori Reali')
            ax.set_xlabel('Predizioni')
            st.pyplot(fig)
        
        # Riepilogo finale
        st.header("🎯 Conclusioni")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Modello Migliore", best_model_name)
        with col2:
            st.metric("Accuratezza Massima", f"{best_accuracy:.1%}")
        with col3:
            reliability = "Alta" if best_accuracy > 0.7 else "Media" if best_accuracy > 0.6 else "Bassa"
            st.metric("Affidabilità", reliability)
        
        # Analisi dettagliata
        st.subheader("Analisi Dettagliata")
        
        if best_accuracy > 0.7:
            st.success(f"""
            ✅ **Il modello {best_model_name} mostra buone performance predittive**
            
            - Accuratezza: {best_accuracy:.1%}
            - Il modello può essere utilizzato come supporto decisionale
            - Le news sentiment sembrano avere correlazione con i movimenti di prezzo
            """)
        elif best_accuracy > 0.6:
            st.warning(f"""
            ⚠️ **Il modello {best_model_name} mostra performance moderate**
            
            - Accuratezza: {best_accuracy:.1%}
            - Il modello può fornire indicazioni generali
            - Consigliabile utilizzare altre analisi tecniche come supporto
            """)
        else:
            st.error(f"""
            ❌ **Il modello {best_model_name} mostra performance limitate**
            
            - Accuratezza: {best_accuracy:.1%}
            - Le news sentiment potrebbero non essere sufficientemente predittive
            - Necessarie analisi più approfondite o features aggiuntive
            """)
        
        # Report dettagliato
        with st.expander("Report Dettagliato del Modello"):
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred_best))
            
            st.subheader("Importanza delle Features (Random Forest)")
            if best_model_name == 'Random Forest':
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(feature_importance['Feature'], feature_importance['Importance'])
                ax.set_title('Importanza delle Features')
                ax.set_xlabel('Importanza')
                plt.tight_layout()
                st.pyplot(fig)
        
        # Download dei risultati
        st.subheader("📥 Download Risultati")
        
        # Prepara dati per download
        results_summary = {
            'Ticker': ticker,
            'Periodo_Analisi': f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}",
            'Giorni_Analizzati': len(stock_data),
            'News_Raccolte': len(news_df),
            'Modello_Migliore': best_model_name,
            'Accuratezza': f"{best_accuracy:.3f}",
            'Soglia_Variazione': f"{threshold}%"
        }
        
        results_df = pd.DataFrame([results_summary])
        csv = results_df.to_csv(index=False)
        
        st.download_button(
            label="Scarica Riepilogo Analisi (CSV)",
            data=csv,
            file_name=f"{ticker}_analisi_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Inserisci un ticker nella sidebar e clicca 'Avvia Analisi' per iniziare")
    
    # Informazioni sull'app
    st.markdown("""
    ## Come funziona l'app:
    
    1. **📊 Raccolta Dati Finanziari**: Scarica 2 anni di dati storici da Yahoo Finance
    2. **🏷️ Classificazione Movimenti**: Etichetta i giorni come Positive/Negative/No-change basandosi sulla soglia impostata
    3. **📰 Raccolta News**: Ottiene news correlate al ticker tramite Google News RSS
    4. **🤖 Analisi Sentiment**: Classifica il sentiment delle news usando tecniche di NLP
    5. **📈 Modello Predittivo**: Addestra modelli ML per predire i movimenti di prezzo
    6. **📊 Valutazione**: Mostra l'accuratezza e l'affidabilità del modello
    
    ### Modelli utilizzati:
    - **Random Forest**: Ensemble method robusto
    - **Logistic Regression**: Modello lineare interpretabile  
    - **SVM**: Support Vector Machine per classificazione
    
    ### Features utilizzate:
    - Sentiment delle news
    - Prezzo di chiusura precedente
    - Volume di trading
    - Medie mobili di prezzo e volume
    """)
