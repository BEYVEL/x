"""
Профессиональный RAG чат с реальными эмбеддингами
OpenRouter API ключ встроен в код
"""

import streamlit as st
import numpy as np
import os
import requests
import json
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
    
    /* Улучшенное форматирование ответов */
    .stMarkdown h3 {
        color: #1E88E5;
        margin-top: 1rem;
    }
    .stMarkdown hr {
        margin: 1.5rem 0;
    }
    .stMarkdown ul {
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# === ВСТАВЬТЕ ВАШ API КЛЮЧ ЗДЕСЬ ===
OPENROUTER_API_KEY = "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"  # Замените на ваш реальный ключ
# =================================

class ProfessionalRAG:
    """Профессиональная RAG система с реальными эмбеддингами"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.chunks = []
        self.embeddings = None
        self.api_key = OPENROUTER_API_KEY
        
        # Загружаем документ
        self._load_document()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Получение реальных эмбеддингов через OpenRouter API"""
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://localhost:8501",
                    "X-Title": "National AI Strategy Chat"
                },
                json={
                    "model": "sentence-transformers/all-mpnet-base-v2",
                    "input": text[:1000]
                },
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()['data'][0]['embedding']
            else:
                st.warning(f"⚠️ Ошибка API: {response.status_code}, использую локальный режим")
                return self._get_fallback_embedding(text)
                
        except Exception as e:
            st.warning(f"⚠️ Ошибка подключения: {e}, использую локальный режим")
            return self._get_fallback_embedding(text)
    
    def _get_fallback_embedding(self, text: str) -> List[float]:
        """Улучшенный fallback с семантическими весами"""
        words = text.lower().split()
        
        # Веса для ключевых терминов
        term_weights = {
            'искусственный интеллект': 4.0,
            'определение': 3.0,
            'понятие': 3.0,
            'термин': 2.5,
            'технологии': 1.5,
            'система': 1.0
        }
        
        dim = 384
        features = np.zeros(dim)
        
        for word in words:
            idx = abs(hash(word)) % dim
            features[idx] += 1
        
        # Усиление для ключевых фраз
        for term, weight in term_weights.items():
            if term in text.lower():
                term_idx = abs(hash(term)) % dim
                features[term_idx] += weight
        
        # Нормализация
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.tolist()
    
    def _chunk_by_articles(self, text: str) -> List[Dict[str, Any]]:
        """Интеллектуальное разбиение на статьи"""
        # Находим все статьи
        article_pattern = r'(\d+\.\s+[^0-9]+(?:\n[^0-9][^\n]*)*)'
        articles = re.findall(article_pattern, text, re.MULTILINE)
        
        chunks = []
        for article_text in articles:
            # Извлекаем номер статьи
            num_match = re.match(r'(\d+)\.', article_text)
            if num_match:
                article_num = num_match.group(1)
                
                # Извлекаем заголовок (первая строка)
                first_line = article_text.split('\n')[0]
                title = re.sub(r'^\d+\.\s+', '', first_line)[:100]
                
                # Разбиваем длинные статьи на логические части
                if len(article_text) > 800:
                    # Разбиваем по подпунктам (а), б), в) и т.д.)
                    sub_sections = re.split(r'\n\s*([а-я]\))\s*', article_text)
                    
                    if len(sub_sections) > 1:
                        # Есть подпункты
                        for j in range(1, len(sub_sections), 2):
                            sub_marker = sub_sections[j]
                            sub_text = sub_sections[j+1] if j+1 < len(sub_sections) else ""
                            chunk_text = f"{article_num}. {title}\n{sub_marker} {sub_text}"
                            chunks.append({
                                'text': chunk_text.strip(),
                                'article': article_num,
                                'title': title,
                                'full_text': article_text.strip(),
                                'is_definition': 'поняти' in article_text.lower() or 'определ' in article_text.lower()
                            })
                    else:
                        # Нет подпунктов, разбиваем по предложениям
                        sentences = re.split(r'[.!?]+', article_text)
                        current_chunk = ""
                        for sent in sentences:
                            if len(current_chunk) + len(sent) < 500:
                                current_chunk += sent + ". "
                            else:
                                if current_chunk:
                                    chunks.append({
                                        'text': current_chunk.strip(),
                                        'article': article_num,
                                        'title': title,
                                        'full_text': article_text.strip(),
                                        'is_definition': 'поняти' in article_text.lower() or 'определ' in article_text.lower()
                                    })
                                current_chunk = sent + ". "
                        if current_chunk:
                            chunks.append({
                                'text': current_chunk.strip(),
                                'article': article_num,
                                'title': title,
                                'full_text': article_text.strip(),
                                'is_definition': 'поняти' in article_text.lower() or 'определ' in article_text.lower()
                            })
                else:
                    chunks.append({
                        'text': article_text.strip(),
                        'article': article_num,
                        'title': title,
                        'full_text': article_text.strip(),
                        'is_definition': 'поняти' in article_text.lower() or 'определ' in article_text.lower()
                    })
        
        return chunks
    
    def _load_document(self):
        """Загрузка и обработка документа"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Разбиваем на чанки
            chunks = self._chunk_by_articles(text)
            
            if not chunks:
                st.error("❌ Не удалось создать чанки")
                return
            
            # Прогресс бар
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Генерируем эмбеддинги
            chunk_embeddings = []
            
            for i, chunk in enumerate(chunks):
                status_text.text(f"Обработка чанка {i+1}/{len(chunks)}...")
                emb = self._get_embedding(chunk['text'])
                chunk_embeddings.append(emb)
                progress_bar.progress((i + 1) / len(chunks))
            
            progress_bar.empty()
            status_text.empty()
            
            # Сохраняем
            self.chunks = chunks
            self.embeddings = np.array(chunk_embeddings)
            
            # Статистика
            unique_articles = set(c['article'] for c in chunks)
            st.sidebar.success(f"✅ Загружено: {len(unique_articles)} статей, {len(chunks)} чанков")
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {e}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Умный поиск с контекстным ранжированием"""
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
        question_type = self._detect_question_type(query)
        
        # Финальные оценки
        final_scores = similarities.copy()
        
        for i, chunk in enumerate(self.chunks):
            # Буст для определений, если вопрос о понятиях
            if question_type == 'definition' and chunk.get('is_definition', False):
                final_scores[i] *= 2.0  # Удваиваем релевантность для статей с определениями
            
            # Буст для точного совпадения номера статьи
            numbers = re.findall(r'\d+', query)
            if chunk['article'] in numbers:
                final_scores[i] += 0.5
            
            # Буст для статей с маленькими номерами (основные понятия)
            if chunk['article'].isdigit() and int(chunk['article']) < 10:
                if any(word in query.lower() for word in ['что', 'как', 'определение', 'понятие']):
                    final_scores[i] += 0.3
        
        # Топ результаты
        top_indices = np.argsort(final_scores)[-k*2:][::-1]
        
        # Собираем уникальные статьи
        results = []
        seen_articles = set()
        
        for idx in top_indices:
            if final_scores[idx] > 0.15:
                article = self.chunks[idx]['article']
                
                if article not in seen_articles:
                    results.append({
                        'text': self.chunks[idx]['text'],
                        'full_text': self.chunks[idx]['full_text'],
                        'article': article,
                        'title': self.chunks[idx].get('title', ''),
                        'similarity': float(final_scores[idx]),
                        'is_definition': self.chunks[idx].get('is_definition', False)
                    })
                    seen_articles.add(article)
            
            if len(results) >= k:
                break
        
        # Сортируем по релевантности
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results
    
    def _detect_question_type(self, query: str) -> str:
        """Определение типа вопроса для контекстного ранжирования"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['что такое', 'определение', 'понятие', 'что понимается', 'термин']):
            return 'definition'
        elif any(word in query_lower for word in ['какие', 'перечисли', 'назови']):
            return 'enumeration'
        elif any(word in query_lower for word in ['почему', 'зачем', 'какова цель']):
            return 'explanation'
        elif any(word in query_lower for word in ['статья', 'пункт']):
            return 'article_reference'
        else:
            return 'general'
    
    def _format_answer(self, question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Форматирование красивого, читаемого ответа"""
        question_type = self._detect_question_type(question)
        
        answer = "📄 **Национальная стратегия развития ИИ**\n\n"
        
        if question_type == 'definition' and any(c['is_definition'] for c in relevant_chunks):
            # Для вопросов об определениях показываем сначала статью 5
            def_chunks = [c for c in relevant_chunks if c['article'] == '5']
            other_chunks = [c for c in relevant_chunks if c['article'] != '5']
            
            if def_chunks:
                answer += "### 📌 Определение из Стратегии:\n\n"
                for chunk in def_chunks:
                    # Извлекаем только определение из текста
                    text = chunk['full_text']
                    # Находим определение искусственного интеллекта
                    match = re.search(r'а\)\s+искусственный интеллект[^.]+\.[^.]+\.[^.]+\.[^.]+\.[^.]*', text, re.IGNORECASE)
                    if match:
                        answer += match.group(0).strip() + "\n\n"
                    else:
                        # Если не нашли точное совпадение, показываем часть текста
                        sentences = text.split('.')
                        def_sentences = []
                        for sent in sentences:
                            if 'искусственный интеллект' in sent.lower():
                                def_sentences.append(sent.strip())
                        if def_sentences:
                            answer += ". ".join(def_sentences[:3]) + ".\n\n"
            
            # Добавляем другие релевантные статьи
            if other_chunks:
                answer += "### 📚 Дополнительно:\n\n"
                for chunk in other_chunks[:1]:
                    answer += f"**Статья {chunk['article']}**"
                    if chunk.get('title'):
                        answer += f" - {chunk['title']}"
                    answer += "\n\n"
                    
                    # Показываем релевантную часть
                    sentences = chunk['full_text'].split('.')
                    relevant_sents = []
                    for sent in sentences:
                        if any(word in sent.lower() for word in question.lower().split()):
                            relevant_sents.append(sent.strip())
                    
                    if relevant_sents:
                        answer += ". ".join(relevant_sents[:2]) + ".\n\n"
        
        else:
            # Для общих вопросов
            for i, chunk in enumerate(relevant_chunks[:2]):
                answer += f"### Статья {chunk['article']}"
                if chunk.get('title'):
                    answer += f" - {chunk['title']}"
                answer += "\n\n"
                
                # Показываем релевантные части
                text = chunk['full_text']
                sentences = text.split('.')
                
                # Находим предложения, релевантные вопросу
                question_words = set(question.lower().split())
                relevant_sentences = []
                
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    sent_words = set(sent.lower().split())
                    # Если есть пересечение с вопросом
                    if len(question_words & sent_words) > 0:
                        relevant_sentences.append(sent)
                
                if relevant_sentences:
                    # Показываем до 3 релевантных предложений
                    shown = relevant_sentences[:3]
                    for sent in shown:
                        # Выделяем ключевые термины жирным
                        for word in question_words:
                            if len(word) > 3:
                                pattern = re.compile(re.escape(word), re.IGNORECASE)
                                sent = pattern.sub(f"**{word}**", sent)
                        answer += f"• {sent}.\n"
                    answer += "\n"
                else:
                    # Если не нашли релевантных предложений, показываем начало
                    answer += text[:300] + "...\n\n"
        
        # Источники
        articles_used = [f"Статья {c['article']}" for c in relevant_chunks[:2]]
        if articles_used:
            answer += f"\n---\n*Источники: {', '.join(articles_used)}*"
        
        return answer
    
    def query(self, question: str) -> str:
        """Полный цикл RAG"""
        if not self.chunks:
            return "❌ Документ не загружен."
        
        # Поиск
        relevant = self.search(question, k=5)
        
        if not relevant:
            return "❌ В документе не найдена информация по вашему вопросу."
        
        # Форматируем ответ
        return self._format_answer(question, relevant)

def main():
    st.title("📄 Национальная стратегия развития ИИ")
    st.markdown("*Чат на основе официального документа (с изменениями 2024 г.)*")
    
    # Информация об API
    if OPENROUTER_API_KEY != "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":
        st.sidebar.success("✅ API ключ подключен")
    else:
        st.sidebar.warning("⚠️ Вставьте ваш API ключ в код")
    
    # Путь к файлу
    file_path = "filerag.txt"
    
    # Проверяем наличие файла
    if not os.path.exists(file_path):
        st.error(f"❌ Файл {file_path} не найден!")
        return
    
    # Инициализация RAG
    if 'rag' not in st.session_state:
        with st.spinner("🔄 Загрузка документа и генерация эмбеддингов..."):
            st.session_state.rag = ProfessionalRAG(file_path)
    
    rag = st.session_state.rag
    
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
            with st.spinner("🔍 Анализирую документ..."):
                response = rag.query(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
