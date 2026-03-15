"""
ПРОФЕССИОНАЛЬНЫЙ RAG с OpenRouter LLM
НАСТОЯЩЕЕ ПОНИМАНИЕ ТЕКСТА
"""

import streamlit as st
import numpy as np
import os
import requests
import re
import json
from typing import List, Dict, Any

# Настройка страницы
st.set_page_config(
    page_title="Национальная стратегия ИИ",
    page_icon="📄",
    layout="centered"
)

# Стили
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .question-btn {
        margin: 5px 0;
        width: 100%;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
        line-height: 1.6;
    }
    .source-box {
        color: #666;
        font-size: 0.9rem;
        margin-top: 1rem;
        padding-top: 0.5rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# === ВАШ OPENROUTER КЛЮЧ ===
OPENROUTER_API_KEY = "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"
# ==========================

class OpenRouterRAG:
    """RAG с настоящей LLM через OpenRouter"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.articles = {}
        self.embeddings = {}
        self.api_key = OPENROUTER_API_KEY
        
        # Проверяем ключ
        if not self.api_key.startswith('sk-or-v1-'):
            st.error("❌ Неверный формат ключа OpenRouter")
        else:
            st.sidebar.success("✅ OpenRouter ключ загружен")
        
        # Загружаем документ
        self._load_document()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Эмбеддинги для поиска"""
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
        
        # Запасной вариант
        words = text.lower().split()
        dim = 384
        features = np.zeros(dim)
        for word in words:
            features[abs(hash(word)) % dim] += 1
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features.tolist()
    
    def _generate_with_llm(self, context: str, question: str) -> str:
        """
        Генерация ответа через OpenRouter с моделью Mistral
        """
        
        prompt = f"""Ты - эксперт по Национальной стратегии развития искусственного интеллекта РФ.

На основе фрагментов документа ниже, ответь на вопрос.
ОТВЕЧАЙ СВОИМИ СЛОВАМИ, НЕ КОПИРУЙ ТЕКСТ ИЗ ДОКУМЕНТА.
Дай полный, структурированный ответ.

ФРАГМЕНТЫ ДОКУМЕНТА:
{context}

ВОПРОС: {question}

ОТВЕТ:"""
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://localhost:8501",
                    "X-Title": "National AI Strategy Chat"
                },
                json={
                    "model": "mistralai/mistral-7b-instruct:free",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            elif response.status_code == 402:
                return "❌ Требуется пополнить баланс OpenRouter (минимум $10)"
            else:
                return f"❌ Ошибка API: {response.status_code}"
                
        except Exception as e:
            return f"❌ Ошибка: {str(e)}"
    
    def _load_document(self):
        """Загрузка документа"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Разбиваем по статьям
            lines = text.split('\n')
            current_article = ""
            current_num = ""
            
            for line in lines:
                match = re.match(r'^(\d+)\.\s+(.*)', line.strip())
                if match:
                    if current_num and current_article:
                        self.articles[current_num] = current_article.strip()
                    current_num = match.group(1)
                    current_article = line + "\n"
                elif current_num:
                    current_article += line + "\n"
            
            if current_num and current_article:
                self.articles[current_num] = current_article.strip()
            
            # Индексация
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (num, text) in enumerate(self.articles.items()):
                status_text.text(f"Индексация статьи {num}...")
                self.embeddings[num] = self._get_embedding(text)
                progress_bar.progress((i + 1) / len(self.articles))
            
            progress_bar.empty()
            status_text.empty()
            
            st.sidebar.success(f"✅ Загружено {len(self.articles)} статей")
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {e}")
    
    def search(self, query: str, k: int = 3) -> List[tuple]:
        """Поиск релевантных статей"""
        if not self.articles:
            return []
        
        query_emb = np.array(self._get_embedding(query))
        
        # Вычисляем сходство
        similarities = []
        for num, emb in self.embeddings.items():
            emb_array = np.array(emb)
            norm_product = np.linalg.norm(emb_array) * np.linalg.norm(query_emb)
            if norm_product > 0:
                similarity = np.dot(emb_array, query_emb) / norm_product
            else:
                similarity = 0
            similarities.append((num, similarity))
        
        # Сортируем
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Приоритет для статьи 2 в юридических вопросах
        if any(word in query.lower() for word in ['закон', 'правов', 'федеральн', 'конституц']):
            for i, (num, sim) in enumerate(similarities):
                if num == '2':
                    # Поднимаем статью 2 наверх
                    similarities.pop(i)
                    similarities.insert(0, ('2', sim + 1.0))
                    break
        
        return similarities[:k]
    
    def query(self, question: str) -> str:
        """Полный RAG цикл"""
        if not self.articles:
            return "❌ Документ не загружен."
        
        # Поиск релевантных статей
        relevant = self.search(question, k=3)
        
        if not relevant:
            return "❌ В документе не найдена информация."
        
        # Собираем контекст
        context_parts = []
        articles_used = []
        
        for num, sim in relevant:
            if sim > 0.1:
                article_text = self.articles[num]
                if len(article_text) > 800:
                    article_text = article_text[:800] + "..."
                context_parts.append(f"[Статья {num}]:\n{article_text}")
                articles_used.append(num)
        
        context = "\n\n".join(context_parts)
        
        # Генерируем ответ через LLM
        answer = self._generate_with_llm(context, question)
        
        # Добавляем источники
        answer += f"\n\n<div class='source-box'>📚 Источники: Статьи {', '.join(articles_used)}</div>"
        
        return f'<div class="answer-box">{answer}</div>'

def main():
    st.title("📄 Национальная стратегия развития ИИ")
    st.markdown("*Анализ с настоящей LLM через OpenRouter*")
    
    with st.sidebar:
        st.header("❓ Примеры вопросов")
        
        examples = {
            "📌 Какие федеральные законы?": "Какие федеральные законы составляют правовую основу стратегии?",
            "📌 Что такое ИИ?": "Что такое искусственный интеллект по определению стратегии?",
            "📌 Большие фундаментальные модели": "Что такое большие фундаментальные модели и какой порог параметров?",
            "📌 Цели развития": "Какие цели развития ИИ указаны в стратегии?",
            "📌 Доверенные технологии": "Что такое доверенные технологии ИИ?"
        }
        
        for btn_text, question in examples.items():
            if st.button(btn_text, use_container_width=True):
                st.session_state.prompt = question
        
        st.markdown("---")
        st.info("""
        **Как получить ответы:**
        1. Ключ OpenRouter загружен
        2. Нужен баланс минимум $10
        3. Модель: Mistral-7B
        """)
    
    # Путь к файлу
    file_path = "filerag.txt"
    
    if not os.path.exists(file_path):
        st.error(f"❌ Файл {file_path} не найден!")
        return
    
    # Инициализация
    if 'rag' not in st.session_state:
        with st.spinner("🔄 Загрузка документа..."):
            st.session_state.rag = OpenRouterRAG(file_path)
    
    # История
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Обработка预设 вопроса
    if "prompt" in st.session_state:
        prompt = st.session_state.prompt
        del st.session_state.prompt
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Анализирую документ через LLM..."):
                response = st.session_state.rag.query(prompt)
                st.markdown(response, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Отображение
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.markdown(message["content"])
    
    # Ввод
    if prompt := st.chat_input("Задайте вопрос о стратегии..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Анализирую документ через LLM..."):
                response = st.session_state.rag.query(prompt)
                st.markdown(response, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
