"""
Фокусированный RAG чат с реальными эмбеддингами через OpenRouter
Только на основе файла filerag.txt
"""

import streamlit as st
import numpy as np
import os
import requests
import json
import re
from typing import List, Dict

# Настройка страницы
st.set_page_config(
    page_title="Национальная стратегия ИИ",
    page_icon="📄",
    layout="centered"
)

# Скрываем лишние элементы
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {margin-top: -50px;}
    .block-container {padding-top: 2rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

class SmartStrategyRAG:
    """RAG с реальными эмбеддингами через OpenRouter API"""
    
    def __init__(self, file_path: str, api_key: str = None):
        self.file_path = file_path
        self.chunks = []
        self.embeddings = None
        self.sources = []
        
        # API ключ (можно получить бесплатно на openrouter.ai)
        self.api_key = api_key or st.secrets.get("OPENROUTER_API_KEY", None)
        
        if not self.api_key:
            st.warning("⚠️ API ключ не найден. Работа будет в ограниченном режиме.")
        
        # Загружаем документ
        self._load_document()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Получение реальных эмбеддингов через OpenRouter API"""
        if not self.api_key:
            # Улучшенный fallback с учетом контекста
            return self._get_fallback_embedding(text)
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sentence-transformers/all-mpnet-base-v2",
                    "input": text[:1000]  # Ограничиваем длину
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()['data'][0]['embedding']
            else:
                return self._get_fallback_embedding(text)
                
        except Exception:
            return self._get_fallback_embedding(text)
    
    def _get_fallback_embedding(self, text: str) -> List[float]:
        """Улучшенный fallback с учетом важности слов"""
        words = text.lower().split()
        
        # Важные термины для юридических документов
        legal_terms = {
            'федеральный закон': 3.0,
            'конституция': 3.0,
            'статья': 2.5,
            'правовой': 2.0,
            'регулирование': 2.0,
            'нормативный': 2.0
        }
        
        dim = 768  # Стандартная размерность для эмбеддингов
        features = np.zeros(dim)
        
        for word in words:
            idx = abs(hash(word)) % dim
            features[idx] += 1
        
        # Усиление для юридических терминов
        for term, weight in legal_terms.items():
            if term in text.lower():
                term_idx = abs(hash(term)) % dim
                features[term_idx] += weight
        
        # Нормализация
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.tolist()
    
    def _chunk_by_articles(self, text: str) -> List[Dict]:
        """Разбиение документа по статьям с сохранением номеров"""
        # Находим все статьи (пункты)
        article_pattern = r'(\d+\.\s+[^\n]+(?:\n[^0-9][^\n]*)*)'
        articles = re.findall(article_pattern, text, re.MULTILINE)
        
        chunks = []
        for article in articles:
            # Извлекаем номер статьи
            article_num = re.match(r'(\d+)\.', article)
            num = article_num.group(1) if article_num else "0"
            
            # Разбиваем длинные статьи на подчасти
            if len(article) > 800:
                sentences = article.replace('!', '.').replace('?', '.').split('.')
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) < 600:
                        current_chunk += sent + ". "
                    else:
                        if current_chunk:
                            chunks.append({
                                'text': current_chunk.strip(),
                                'article': num,
                                'full_text': article
                            })
                        current_chunk = sent + ". "
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'article': num,
                        'full_text': article
                    })
            else:
                chunks.append({
                    'text': article.strip(),
                    'article': num,
                    'full_text': article
                })
        
        return chunks
    
    def _load_document(self):
        """Загрузка и обработка документа"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Разбиваем на чанки по статьям
            chunks = self._chunk_by_articles(text)
            
            if not chunks:
                st.error("❌ Не удалось создать чанки")
                return
            
            # Прогресс бар
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Генерируем эмбеддинги
            chunk_texts = []
            chunk_embeddings = []
            
            for i, chunk in enumerate(chunks):
                status_text.text(f"Обработка чанка {i+1}/{len(chunks)}...")
                
                chunk_texts.append({
                    'text': chunk['text'],
                    'article': chunk['article'],
                    'full_text': chunk['full_text']
                })
                
                emb = self._get_embedding(chunk['text'])
                chunk_embeddings.append(emb)
                
                progress_bar.progress((i + 1) / len(chunks))
            
            progress_bar.empty()
            status_text.empty()
            
            # Сохраняем
            self.chunks = chunk_texts
            self.embeddings = np.array(chunk_embeddings)
            
            st.sidebar.success(f"✅ Загружено {len(chunks)} чанков из {len(set(c['article'] for c in chunk_texts))} статей")
            
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Семантический поиск"""
        if not self.chunks:
            return []
        
        query_emb = np.array(self._get_embedding(query))
        
        # Косинусное сходство
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        
        if norms.all() and query_norm:
            similarities = np.dot(self.embeddings, query_emb) / (norms * query_norm)
        else:
            similarities = np.zeros(len(self.chunks))
        
        # Буст для статей, номера которых упомянуты в запросе
        numbers = re.findall(r'\d+', query)
        for i, chunk in enumerate(self.chunks):
            if chunk['article'] in numbers:
                similarities[i] += 0.5
        
        # Топ результаты
        top_indices = np.argsort(similarities)[-k*2:][::-1]
        
        results = []
        seen_articles = set()
        
        for idx in top_indices:
            if similarities[idx] > 0.2:  # Порог релевантности
                article = self.chunks[idx]['article']
                
                # Берем не больше 2 чанков из одной статьи
                if article in seen_articles:
                    article_count = sum(1 for r in results if r['article'] == article)
                    if article_count >= 2:
                        continue
                
                results.append({
                    'text': self.chunks[idx]['text'],
                    'full_text': self.chunks[idx]['full_text'],
                    'article': article,
                    'similarity': float(similarities[idx])
                })
                seen_articles.add(article)
            
            if len(results) >= k:
                break
        
        return results
    
    def query(self, question: str) -> str:
        """Ответ на вопрос"""
        if not self.chunks:
            return "❌ Документ не загружен."
        
        # Поиск релевантных чанков
        relevant = self.search(question, k=5)
        
        if not relevant:
            return "❌ В документе не найдена информация по вашему вопросу."
        
        # Группируем по статьям
        articles = {}
        for r in relevant:
            if r['article'] not in articles:
                articles[r['article']] = []
            articles[r['article']].append(r['full_text'])
        
        # Формируем ответ
        answer = "📄 **На основе Национальной стратегии развития ИИ:**\n\n"
        
        # Сортируем статьи по релевантности
        sorted_articles = sorted(articles.items(), 
                               key=lambda x: max(r['similarity'] for r in relevant if r['article'] == x[0]), 
                               reverse=True)
        
        for article_num, texts in sorted_articles[:3]:
            answer += f"**Статья {article_num}**\n"
            
            # Берем первый текст как представителя статьи
            full_text = texts[0]
            
            # Для юридических вопросов показываем полный текст статьи
            if any(word in question.lower() for word in ['закон', 'правов', 'федеральн', 'конституц']):
                # Обрезаем до разумной длины
                if len(full_text) > 600:
                    answer += full_text[:600] + "...\n\n"
                else:
                    answer += full_text + "\n\n"
            else:
                # Для общих вопросов показываем релевантную часть
                if len(full_text) > 400:
                    answer += full_text[:400] + "...\n\n"
                else:
                    answer += full_text + "\n\n"
        
        # Список статей
        article_list = ', '.join([f"Статья {a}" for a, _ in sorted_articles[:3]])
        answer += f"\n*Источник: Национальная стратегия развития ИИ, {article_list}*"
        
        return answer

def main():
    st.title("📄 Национальная стратегия развития ИИ")
    st.markdown("*Чат на основе официального документа (с изменениями 2024 г.)*")
    
    # Путь к файлу
    file_path = "filerag.txt"
    
    # Проверяем наличие файла
    if not os.path.exists(file_path):
        st.error(f"❌ Файл {file_path} не найден!")
        st.info("Пожалуйста, убедитесь, что файл filerag.txt находится в той же папке")
        return
    
    # Инициализация RAG
    if 'rag' not in st.session_state:
        st.session_state.rag = SmartStrategyRAG(file_path)
    
    rag = st.session_state.rag
    
    # Сайдбар
    with st.sidebar:
        st.header("📚 О документе")
        st.markdown("""
        **Национальная стратегия развития ИИ**  
        *до 2030 года (с изменениями 2024)*
        
        **Разделы:**
        - I. Общие положения (ст. 1-5)
        - II. Развитие ИИ в России и мире (ст. 6-23)
        - III. Основные принципы (ст. 24)
        - IV. Цели и задачи (ст. 25-41)
        - V. Механизмы реализации (ст. 42-50)
        """)
        
        if rag.chunks:
            articles = set(c['article'] for c in rag.chunks)
            st.metric("Всего статей", len(articles))
            st.metric("Всего чанков", len(rag.chunks))
        
        # Получение API ключа
        with st.expander("🔑 Настройка API (для лучших результатов)"):
            st.markdown("""
            1. Получите бесплатный ключ на [openrouter.ai](https://openrouter.ai/)
            2. Вставьте ключ ниже или добавьте в secrets
            """)
            api_key = st.text_input("API ключ OpenRouter", type="password")
            if api_key:
                st.session_state.rag.api_key = api_key
                st.success("✅ Ключ сохранен!")
                st.rerun()
    
    # История чата
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Отображение истории
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Ввод вопроса
    if prompt := st.chat_input("Задайте вопрос о стратегии развития ИИ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Ищу в документе..."):
                response = rag.query(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
