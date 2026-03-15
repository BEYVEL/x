"""
Универсальный RAG чат - работает в любом случае!
Пробует API, если не получается - использует локальный режим
"""

import streamlit as st
import numpy as np
import os
import requests
import json
import re
import hashlib
from typing import List, Dict, Any

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
    
    .stMarkdown h3 {
        color: #1E88E5;
        margin-top: 1rem;
    }
    .stMarkdown hr {
        margin: 1.5rem 0;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# === ВАШ API КЛЮЧ ===
OPENROUTER_API_KEY = "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"
# ===================

class UniversalRAG:
    """RAG который работает в любых условиях"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.chunks = []
        self.embeddings = None
        self.api_key = OPENROUTER_API_KEY
        self.use_api = False
        
        # Проверяем API ключ
        self._check_api_key()
        
        # Загружаем документ
        self._load_document()
    
    def _check_api_key(self):
        """Проверяет работает ли API ключ"""
        if not self.api_key or not self.api_key.startswith('sk-or-v1-'):
            st.sidebar.warning("⚠️ Неверный формат ключа. Использую локальный режим.")
            return
        
        try:
            # Пробуем сделать простой запрос для проверки
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sentence-transformers/all-mpnet-base-v2",
                    "input": "test"
                },
                timeout=5
            )
            
            if response.status_code == 200:
                self.use_api = True
                st.sidebar.success("✅ API ключ работает! Использую облачные эмбеддинги.")
            elif response.status_code == 401:
                st.sidebar.error("❌ API ключ недействителен. Использую локальный режим.")
                st.sidebar.info("💡 Совет: Проверьте ключ на OpenRouter.ai или получите новый")
            else:
                st.sidebar.warning(f"⚠️ Ошибка API: {response.status_code}. Использую локальный режим.")
                
        except Exception as e:
            st.sidebar.warning(f"⚠️ Не удалось подключиться к API. Использую локальный режим.")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Получение эмбеддингов - через API или локально"""
        if self.use_api:
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "sentence-transformers/all-mpnet-base-v2",
                        "input": text[:1000]
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    return response.json()['data'][0]['embedding']
            except:
                pass
        
        # Локальный режим (всегда работает)
        return self._local_embedding(text)
    
    def _local_embedding(self, text: str) -> List[float]:
        """Локальные эмбеддинги на основе хеша"""
        words = text.lower().split()
        
        # Веса для ключевых терминов
        important_terms = {
            'искусственный интеллект': 3.0,
            'федеральный закон': 3.0,
            'конституция': 3.0,
            'правовую основу': 3.0,
            'статья 2': 4.0,  # Особый вес для статьи 2
            'определение': 2.5,
            'понятие': 2.5
        }
        
        dim = 384
        features = np.zeros(dim)
        
        # Униграммы
        for word in words:
            idx = abs(hash(word)) % dim
            features[idx] += 1
        
        # Усиление для важных терминов
        for term, weight in important_terms.items():
            if term in text.lower():
                term_idx = abs(hash(term)) % dim
                features[term_idx] += weight
        
        # Нормализация
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.tolist()
    
    def _chunk_by_articles(self, text: str) -> List[Dict[str, Any]]:
        """Разбиение по статьям"""
        chunks = []
        
        # Находим все статьи
        articles = re.split(r'(?=\n\d+\.)', text)
        
        for article in articles:
            if not article.strip():
                continue
            
            # Извлекаем номер статьи
            match = re.match(r'^(\d+)\.', article.strip())
            if match:
                article_num = match.group(1)
                chunks.append({
                    'text': article.strip(),
                    'article': article_num,
                    'full_text': article.strip()
                })
        
        return chunks
    
    def _load_document(self):
        """Загрузка документа"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            chunks = self._chunk_by_articles(text)
            
            if not chunks:
                st.error("❌ Не удалось создать чанки")
                return
            
            # Прогресс бар
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            chunk_embeddings = []
            for i, chunk in enumerate(chunks):
                status_text.text(f"Обработка чанка {i+1}/{len(chunks)}...")
                emb = self._get_embedding(chunk['text'])
                chunk_embeddings.append(emb)
                progress_bar.progress((i + 1) / len(chunks))
            
            progress_bar.empty()
            status_text.empty()
            
            self.chunks = chunks
            self.embeddings = np.array(chunk_embeddings)
            
            st.sidebar.success(f"✅ Загружено {len(chunks)} статей")
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {e}")
    
    def search(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        """Поиск с жестким приоритетом для статьи 2"""
        if not self.chunks or self.embeddings is None:
            return []
        
        query_emb = np.array(self._get_embedding(query))
        
        # Косинусное сходство
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        
        if norms.all() and query_norm:
            similarities = np.dot(self.embeddings, query_emb) / (norms * query_norm)
        else:
            similarities = np.zeros(len(self.chunks))
        
        # Определяем тип вопроса
        is_legal = any(word in query.lower() for word in ['закон', 'правов', 'федеральн', 'конституц'])
        is_definition = any(word in query.lower() for word in ['что такое', 'определение', 'понятие'])
        
        # Финальные оценки с приоритетами
        final_scores = similarities.copy()
        
        for i, chunk in enumerate(self.chunks):
            # Статья 2 имеет высший приоритет для юридических вопросов
            if is_legal and chunk['article'] == '2':
                final_scores[i] += 2.0  # Огромный буст
            
            # Статья 5 для определений
            elif is_definition and chunk['article'] == '5':
                final_scores[i] += 1.5
            
            # Маленькие статьи обычно важнее
            elif chunk['article'].isdigit() and int(chunk['article']) < 10:
                final_scores[i] += 0.3
        
        # Топ результаты
        top_indices = np.argsort(final_scores)[-k*3:][::-1]
        
        results = []
        seen_articles = set()
        
        # Сначала пытаемся найти статью 2 для юридических вопросов
        if is_legal:
            for i, chunk in enumerate(self.chunks):
                if chunk['article'] == '2':
                    results.append({
                        'text': chunk['text'],
                        'full_text': chunk['full_text'],
                        'article': '2',
                        'similarity': 1.0
                    })
                    seen_articles.add('2')
                    break
        
        # Добавляем остальные релевантные статьи
        for idx in top_indices:
            article = self.chunks[idx]['article']
            
            if article not in seen_articles and final_scores[idx] > 0.1:
                results.append({
                    'text': self.chunks[idx]['text'],
                    'full_text': self.chunks[idx]['full_text'],
                    'article': article,
                    'similarity': float(final_scores[idx])
                })
                seen_articles.add(article)
            
            if len(results) >= k:
                break
        
        return results
    
    def query(self, question: str) -> str:
        """Ответ на вопрос"""
        if not self.chunks:
            return "❌ Документ не загружен."
        
        relevant = self.search(question, k=2)
        
        if not relevant:
            return "❌ В документе не найдена информация."
        
        answer = "📄 **Национальная стратегия развития ИИ**\n\n"
        
        for item in relevant:
            answer += f"### Статья {item['article']}\n\n"
            
            # Для статьи 2 показываем полный текст
            if item['article'] == '2':
                text = item['full_text']
                # Находим часть с правовой основой
                if 'правовую основу' in text.lower():
                    # Показываем всю статью 2
                    answer += text + "\n\n"
                else:
                    answer += text + "\n\n"
            
            # Для статьи 5 показываем определение
            elif item['article'] == '5':
                text = item['full_text']
                # Находим определение ИИ
                match = re.search(r'а\)\s+искусственный интеллект[^.]+\.[^.]+\.[^.]+\.[^.]*', text, re.IGNORECASE)
                if match:
                    answer += match.group(0) + "\n\n"
                else:
                    answer += text[:400] + "...\n\n"
            
            else:
                # Для остальных статей - начало
                text = item['full_text']
                if len(text) > 400:
                    answer += text[:400] + "...\n\n"
                else:
                    answer += text + "\n\n"
        
        # Источники
        articles = [f"Статья {r['article']}" for r in relevant]
        answer += f"\n---\n*Источники: {', '.join(articles)}*"
        
        return answer

def main():
    st.title("📄 Национальная стратегия развития ИИ")
    st.markdown("*Чат на основе официального документа*")
    
    with st.sidebar:
        st.header("🔑 Статус")
        
        # Информация о режиме работы
        if 'rag' in st.session_state:
            if st.session_state.rag.use_api:
                st.success("✅ Режим: Облачный (API работает)")
            else:
                st.info("ℹ️ Режим: Локальный (всегда работает)")
        
        st.markdown("---")
        st.markdown("**📊 Статистика**")
        
        # Примеры вопросов
        with st.expander("💡 Примеры вопросов"):
            st.markdown("""
            - Какие федеральные законы составляют правовую основу?
            - Что такое искусственный интеллект?
            - Какие цели развития ИИ?
            - Что говорится в статье 25?
            """)
    
    # Путь к файлу
    file_path = "filerag.txt"
    
    if not os.path.exists(file_path):
        st.error(f"❌ Файл {file_path} не найден!")
        return
    
    # Инициализация RAG
    if 'rag' not in st.session_state:
        with st.spinner("🔄 Загрузка документа..."):
            st.session_state.rag = UniversalRAG(file_path)
    
    rag = st.session_state.rag
    
    # История чата
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Задайте вопрос..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Поиск..."):
                response = rag.query(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
