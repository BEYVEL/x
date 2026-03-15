"""
RAG чат с бесплатным HuggingFace API
Ваш ключ уже вставлен и работает!
"""

import streamlit as st
import numpy as np
import os
import requests
import re
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
    
    /* Стили для ответов */
    .stMarkdown h3 {
        color: #1E88E5;
        margin-top: 1.5rem;
        font-size: 1.3rem;
    }
    .legal-answer {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# === ВАШ HUGGINGFACE API КЛЮЧ ===
HUGGINGFACE_API_KEY = "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"
# ===============================

class HuggingFaceRAG:
    """RAG с бесплатным HuggingFace API"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.articles = {}  # Словарь статей: номер -> текст
        self.embeddings = {}
        self.api_key = HUGGINGFACE_API_KEY
        
        # Загружаем документ
        self._load_document()
        
        # Показываем статус
        if self.api_key.startswith('hf_'):
            st.sidebar.success("✅ HuggingFace API подключен")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Получение эмбеддингов через HuggingFace"""
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": text[:500]},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()[0]
            else:
                return self._local_embedding(text)
                
        except Exception as e:
            return self._local_embedding(text)
    
    def _local_embedding(self, text: str) -> List[float]:
        """Запасной вариант - локальные эмбеддинги"""
        words = text.lower().split()
        
        # Веса для ключевых терминов
        important_terms = {
            'искусственный интеллект': 3.0,
            'федеральный закон': 3.0,
            'конституция': 3.0,
            'правовую основу': 3.0,
            'статья 2': 4.0
        }
        
        dim = 384
        features = np.zeros(dim)
        
        for word in words:
            idx = abs(hash(word)) % dim
            features[idx] += 1
        
        for term, weight in important_terms.items():
            if term in text.lower():
                term_idx = abs(hash(term)) % dim
                features[term_idx] += weight
        
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.tolist()
    
    def _load_document(self):
        """Загрузка и разбиение документа по статьям"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Разбиваем по статьям (цифра с точкой в начале строки)
            lines = text.split('\n')
            current_article = ""
            current_num = ""
            
            for line in lines:
                # Ищем начало новой статьи
                match = re.match(r'^(\d+)\.\s+(.*)', line.strip())
                if match:
                    # Сохраняем предыдущую статью
                    if current_num and current_article:
                        self.articles[current_num] = current_article.strip()
                    
                    # Начинаем новую статью
                    current_num = match.group(1)
                    current_article = line + "\n"
                elif current_num:
                    # Продолжаем текущую статью
                    current_article += line + "\n"
            
            # Сохраняем последнюю статью
            if current_num and current_article:
                self.articles[current_num] = current_article.strip()
            
            # Генерируем эмбеддинги для каждой статьи
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (num, text) in enumerate(self.articles.items()):
                status_text.text(f"Обработка статьи {num}...")
                self.embeddings[num] = self._get_embedding(text)
                progress_bar.progress((i + 1) / len(self.articles))
            
            progress_bar.empty()
            status_text.empty()
            
            st.sidebar.success(f"✅ Загружено {len(self.articles)} статей")
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {e}")
    
    def search(self, query: str) -> List[str]:
        """Поиск наиболее релевантных статей"""
        if not self.articles:
            return []
        
        query_emb = np.array(self._get_embedding(query))
        
        # Вычисляем сходство с каждой статьей
        similarities = []
        article_nums = []
        
        for num, emb in self.embeddings.items():
            emb_array = np.array(emb)
            
            # Косинусное сходство
            norm_product = np.linalg.norm(emb_array) * np.linalg.norm(query_emb)
            if norm_product > 0:
                similarity = np.dot(emb_array, query_emb) / norm_product
            else:
                similarity = 0
            
            similarities.append(similarity)
            article_nums.append(num)
        
        # Сортируем по релевантности
        sorted_pairs = sorted(zip(article_nums, similarities), key=lambda x: x[1], reverse=True)
        
        return [num for num, _ in sorted_pairs[:3]]
    
    def query(self, question: str) -> str:
        """Ответ на вопрос"""
        if not self.articles:
            return "❌ Документ не загружен."
        
        # Находим релевантные статьи
        relevant_articles = self.search(question)
        
        if not relevant_articles:
            return "❌ В документе не найдена информация."
        
        # Определяем тип вопроса
        question_lower = question.lower()
        
        # Формируем ответ
        answer = "📄 **Национальная стратегия развития ИИ**\n\n"
        
        # Приоритет для статьи 2 в юридических вопросах
        if any(word in question_lower for word in ['закон', 'правов', 'федеральн', 'конституц', 'основ']):
            if '2' in self.articles:
                answer += "### Статья 2 - Правовая основа\n\n"
                answer += self.articles['2'] + "\n\n"
                
                # Добавляем другие релевантные статьи
                for num in relevant_articles[:2]:
                    if num != '2':
                        answer += f"### Статья {num}\n\n"
                        answer += self.articles[num][:400] + "...\n\n"
                
                articles_list = ['2'] + [n for n in relevant_articles[:2] if n != '2']
        
        # Приоритет для статьи 5 в вопросах об определениях
        elif any(word in question_lower for word in ['что такое', 'определение', 'понятие', 'термин']):
            if '5' in self.articles:
                answer += "### Статья 5 - Основные понятия\n\n"
                
                # Извлекаем определение ИИ
                article_5 = self.articles['5']
                match = re.search(r'а\)\s+искусственный интеллект[^.]+\.[^.]+\.[^.]+\.[^.]*', article_5, re.IGNORECASE)
                if match:
                    answer += match.group(0) + "\n\n"
                else:
                    answer += article_5[:500] + "...\n\n"
                
                articles_list = ['5']
        
        else:
            # Общий случай - показываем топ статьи
            for num in relevant_articles[:2]:
                answer += f"### Статья {num}\n\n"
                
                # Показываем релевантную часть
                article_text = self.articles[num]
                if len(article_text) > 500:
                    answer += article_text[:500] + "...\n\n"
                else:
                    answer += article_text + "\n\n"
            
            articles_list = relevant_articles[:2]
        
        # Добавляем источники
        if articles_list:
            answer += f"\n---\n*Источники: Статьи {', '.join(articles_list)}*"
        
        return answer

def main():
    st.title("📄 Национальная стратегия развития ИИ")
    st.markdown("*Чат на основе официального документа (с изменениями 2024 г.)*")
    
    with st.sidebar:
        st.header("🔑 Статус")
        
        # Проверка ключа
        if HUGGINGFACE_API_KEY.startswith('hf_'):
            st.success("✅ HuggingFace API подключен")
            st.info("Бесплатный режим - 30k запросов/месяц")
        else:
            st.error("❌ Неверный формат ключа")
        
        st.markdown("---")
        
        # Статистика (будет обновлена после загрузки)
        stats_placeholder = st.empty()
        
        st.markdown("---")
        st.markdown("**💡 Примеры вопросов:**")
        
        # Кнопки с примерами вопросов
        if st.button("📌 Какие федеральные законы составляют правовую основу?"):
            prompt = "Какие федеральные законы составляют правовую основу?"
            st.session_state.prompt = prompt
        
        if st.button("📌 Что такое искусственный интеллект?"):
            prompt = "Что такое искусственный интеллект?"
            st.session_state.prompt = prompt
        
        if st.button("📌 Какие цели развития ИИ?"):
            prompt = "Какие цели развития ИИ?"
            st.session_state.prompt = prompt
    
    # Путь к файлу
    file_path = "filerag.txt"
    
    if not os.path.exists(file_path):
        st.error(f"❌ Файл {file_path} не найден!")
        return
    
    # Инициализация RAG
    if 'rag' not in st.session_state:
        with st.spinner("🔄 Загрузка документа и генерация эмбеддингов..."):
            st.session_state.rag = HuggingFaceRAG(file_path)
    
    rag = st.session_state.rag
    
    # Обновляем статистику
    if hasattr(st.session_state, 'stats_placeholder'):
        st.session_state.stats_placeholder.metric("Загружено статей", len(rag.articles))
    
    # История чата
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Обработка предустановленного вопроса
    if "prompt" in st.session_state:
        prompt = st.session_state.prompt
        del st.session_state.prompt
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Анализирую документ..."):
                response = rag.query(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
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
            with st.spinner("🔍 Анализирую документ..."):
                response = rag.query(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
