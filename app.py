"""
Профессиональный RAG с пониманием текста через LLM
Использует HuggingFace для эмбеддингов и генерации
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

# Стили для ответов
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .answer-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .source-box {
        font-size: 0.9rem;
        color: #666;
        margin-top: 1rem;
        padding-top: 0.5rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# === ВАШИ КЛЮЧИ ===
HUGGINGFACE_API_KEY = "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"
# =================

class IntelligentRAG:
    """RAG с реальным пониманием текста через LLM"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.articles = {}
        self.embeddings = {}
        self.api_key = HUGGINGFACE_API_KEY
        
        # Загружаем документ
        self._load_document()
        
        # Показываем статус
        if self.api_key.startswith('hf_'):
            st.sidebar.success("✅ HuggingFace API готов к работе")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Получение эмбеддингов для поиска"""
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": text[:500]},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()[0]
        except:
            pass
        
        # Запасной вариант
        return self._local_embedding(text)
    
    def _local_embedding(self, text: str) -> List[float]:
        """Локальные эмбеддинги для поиска"""
        words = text.lower().split()
        dim = 384
        features = np.zeros(dim)
        
        for word in words:
            idx = abs(hash(word)) % dim
            features[idx] += 1
        
        # Нормализация
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.tolist()
    
    def _generate_with_llm(self, context: str, question: str) -> str:
        """Генерация ответа через LLM (настоящее понимание)"""
        try:
            # Формируем промпт для LLM
            prompt = f"""Ты - эксперт по Национальной стратегии развития искусственного интеллекта РФ.
Отвечай на вопросы ТОЛЬКО на основе предоставленного контекста из документа.
Формулируй ответы своими словами, понятно и структурированно.
Если информация отсутствует в контексте, скажи "В документе нет информации об этом".

КОНТЕКСТ ИЗ ДОКУМЕНТА:
{context}

ВОПРОС: {question}

ОТВЕТ (на русском, понятный, структурированный):"""

            # Используем бесплатную модель на HuggingFace
            response = requests.post(
                "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 500,
                        "temperature": 0.3,
                        "top_p": 0.95,
                        "do_sample": True
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').split('ОТВЕТ:')[-1].strip()
            
            return self._generate_fallback(context, question)
            
        except Exception as e:
            return self._generate_fallback(context, question)
    
    def _generate_fallback(self, context: str, question: str) -> str:
        """Запасной вариант без LLM"""
        # Простой анализ ключевых слов
        question_lower = question.lower()
        
        if 'определение' in question_lower or 'что такое' in question_lower:
            if 'искусственный интеллект' in question_lower:
                # Ищем определение в статье 5
                if '5' in self.articles:
                    article_5 = self.articles['5']
                    match = re.search(r'а\)\s+искусственный интеллект[^.]+\.[^.]+\.[^.]+\.[^.]*', article_5, re.IGNORECASE)
                    if match:
                        return f"**Определение из Стратегии:**\n\n{match.group(0)}"
        
        return f"**На основе документа:**\n\n{context[:500]}..."
    
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
            
            # Генерируем эмбеддинги для поиска
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
            
            # Косинусное сходство
            norm_product = np.linalg.norm(emb_array) * np.linalg.norm(query_emb)
            if norm_product > 0:
                similarity = np.dot(emb_array, query_emb) / norm_product
            else:
                similarity = 0
            
            similarities.append((num, similarity))
        
        # Сортируем и возвращаем топ-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def query(self, question: str) -> str:
        """Умный ответ с пониманием контекста"""
        if not self.articles:
            return "❌ Документ не загружен."
        
        # Ищем релевантные статьи
        relevant = self.search(question, k=3)
        
        if not relevant:
            return "❌ В документе не найдена информация."
        
        # Собираем контекст из релевантных статей
        context_parts = []
        articles_used = []
        
        for num, sim in relevant:
            if sim > 0.15:  # Порог релевантности
                article_text = self.articles[num]
                # Обрезаем очень длинные статьи
                if len(article_text) > 1000:
                    article_text = article_text[:1000] + "..."
                context_parts.append(f"[Статья {num}]:\n{article_text}")
                articles_used.append(num)
        
        if not context_parts:
            # Если ничего не нашлось, берем топ-1
            num = relevant[0][0]
            context_parts.append(f"[Статья {num}]:\n{self.articles[num][:800]}...")
            articles_used.append(num)
        
        # Объединяем контекст
        context = "\n\n".join(context_parts)
        
        # Генерируем ответ через LLM
        with st.spinner("🤔 Анализирую документ и формулирую ответ..."):
            answer = self._generate_with_llm(context, question)
        
        # Добавляем источники
        answer += f"\n\n<div class='source-box'>📚 Источники: Статьи {', '.join(articles_used)}</div>"
        
        return f'<div class="answer-box">{answer}</div>'

def main():
    st.title("📄 Национальная стратегия развития ИИ")
    st.markdown("*Интеллектуальный анализ документа с пониманием контекста*")
    
    with st.sidebar:
        st.header("🔑 Статус")
        if HUGGINGFACE_API_KEY.startswith('hf_'):
            st.success("✅ API подключен")
            
            # Информация о возможностях
            st.info("""
            **Что умеет:**
            • Понимает вопросы
            • Анализирует контекст
            • Формулирует ответы своими словами
            • Работает с 40+ статьями
            """)
        
        st.markdown("---")
        
        # Примеры вопросов (те, на которые ДЕЙСТВИТЕЛЬНО есть ответы)
        st.markdown("**💡 Проверенные вопросы:**")
        
        examples = {
            "📌 Что такое искусственный интеллект по определению стратегии?": "Что в стратегии понимается под искусственным интеллектом?",
            "📌 Какие федеральные законы составляют правовую основу?": "Какие федеральные законы составляют правовую основу стратегии?",
            "📌 Что такое большие фундаментальные модели?": "Что такое большие фундаментальные модели и какой порог параметров указан?",
            "📌 Какие цели развития ИИ к 2030 году?": "Какие цели развития искусственного интеллекта указаны в стратегии?",
            "📌 Что такое доверенные технологии ИИ?": "Что такое доверенные технологии искусственного интеллекта?"
        }
        
        for btn_text, question in examples.items():
            if st.button(btn_text, use_container_width=True):
                st.session_state.prompt = question
    
    # Путь к файлу
    file_path = "filerag.txt"
    
    if not os.path.exists(file_path):
        st.error(f"❌ Файл {file_path} не найден!")
        return
    
    # Инициализация RAG
    if 'rag' not in st.session_state:
        with st.spinner("🔄 Загрузка и индексация документа..."):
            st.session_state.rag = IntelligentRAG(file_path)
    
    rag = st.session_state.rag
    
    # Статистика
    with st.sidebar:
        st.metric("Загружено статей", len(rag.articles))
    
    # История чата
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
            response = rag.query(prompt)
            st.markdown(response, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Отображение истории
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.markdown(message["content"])
    
    # Ввод вопроса
    if prompt := st.chat_input("Задайте вопрос о стратегии..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = rag.query(prompt)
            st.markdown(response, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
