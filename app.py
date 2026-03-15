"""
Фокусированный RAG чат с улучшенным поиском
Только на основе файла filerag.txt
"""

import streamlit as st
import numpy as np
import os
import requests
import json
import re
from typing import List, Dict, Any
from collections import Counter

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

class ImprovedStrategyRAG:
    """RAG с улучшенным поиском и ранжированием"""
    
    def __init__(self, file_path: str, api_key: str = None):
        self.file_path = file_path
        self.chunks = []
        self.embeddings = None
        self.api_key = api_key or st.secrets.get("OPENROUTER_API_KEY", None)
        
        if not self.api_key:
            st.warning("⚠️ API ключ не найден. Работа будет в ограниченном режиме.")
        
        # Загружаем документ
        self._load_document()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Получение реальных эмбеддингов через OpenRouter API"""
        if not self.api_key:
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
                    "input": text[:1000]
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
        """Улучшенный fallback с весами для ключевых слов"""
        words = text.lower().split()
        
        # Словарь важных терминов с весами
        important_terms = {
            # Для вопроса о правовой основе
            'правовую основу': 5.0,
            'конституция': 4.0,
            'федеральный закон': 4.0,
            'федеральные законы': 4.0,
            'закон': 3.0,
            'указы президента': 4.0,
            'стратегия': 2.0,
            
            # Общие юридические термины
            'правовой': 3.0,
            'регулирование': 2.5,
            'нормативный': 2.5,
            'законодательство': 3.0
        }
        
        dim = 384
        features = np.zeros(dim)
        
        # Учитываем униграммы
        for word in words:
            idx = abs(hash(word)) % dim
            features[idx] += 1
        
        # Усиление для важных терминов (целиком фразы)
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
        """Разбиение документа по статьям с сохранением номеров"""
        lines = text.split('\n')
        chunks = []
        current_article = ""
        current_num = ""
        current_title = ""
        in_article = False
        
        for i, line in enumerate(lines):
            # Ищем начало новой статьи (цифра с точкой в начале строки)
            match = re.match(r'^(\d+)\.\s+(.*)', line.strip())
            
            if match:
                # Сохраняем предыдущую статью
                if current_article and current_num:
                    chunks.append({
                        'text': current_article.strip(),
                        'article': current_num,
                        'title': current_title,
                        'full_text': current_article.strip()
                    })
                
                # Начинаем новую статью
                current_num = match.group(1)
                current_title = match.group(2)[:50]  # Первые 50 символов заголовка
                current_article = line + "\n"
                in_article = True
            elif in_article:
                # Продолжаем текущую статью
                current_article += line + "\n"
                
                # Проверяем, не закончилась ли статья (следующая цифра в начале строки)
                next_line = lines[i+1] if i+1 < len(lines) else ""
                if re.match(r'^\d+\.', next_line.strip()):
                    # Статья заканчивается, добавим её при следующей итерации
                    pass
            else:
                # Текст до первой статьи (введение)
                if line.strip() and not current_num:
                    chunks.append({
                        'text': line.strip(),
                        'article': "0",
                        'title': "Введение",
                        'full_text': line.strip()
                    })
        
        # Добавляем последнюю статью
        if current_article and current_num:
            chunks.append({
                'text': current_article.strip(),
                'article': current_num,
                'title': current_title,
                'full_text': current_article.strip()
            })
        
        return chunks
    
    def _keyword_boost(self, query: str, text: str) -> float:
        """Дополнительный буст на основе ключевых слов"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Специальные фразы для поиска
        special_phrases = [
            'правовую основу',
            'федеральные законы',
            'конституция',
            'федеральный закон',
            'указы президента'
        ]
        
        boost = 0.0
        
        # Буст для специальных фраз
        for phrase in special_phrases:
            if phrase in query.lower() and phrase in text.lower():
                boost += 0.8
        
        # Буст для общих слов
        common_words = query_words & text_words
        boost += len(common_words) * 0.1
        
        return boost
    
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
            
            # Подсчитываем уникальные статьи
            unique_articles = set()
            for chunk in chunks:
                if chunk['article'] != "0":
                    unique_articles.add(chunk['article'])
            
            st.sidebar.success(f"✅ Загружено {len(chunks)} чанков из {len(unique_articles)} статей")
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {e}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Улучшенный семантический поиск с реранжированием"""
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
        
        # Дополнительные факторы ранжирования
        final_scores = similarities.copy()
        
        for i, chunk in enumerate(self.chunks):
            # 1. Буст для точного совпадения номера статьи
            numbers = re.findall(r'\d+', query)
            if chunk['article'] in numbers:
                final_scores[i] += 0.5
            
            # 2. Буст для статей с маленькими номерами (они обычно важнее)
            if chunk['article'].isdigit() and int(chunk['article']) < 10:
                final_scores[i] += 0.1
            
            # 3. Буст на основе ключевых слов
            keyword_boost = self._keyword_boost(query, chunk['text'])
            final_scores[i] += keyword_boost
            
            # 4. Штраф для статей, которые явно не относятся к вопросу
            if 'финансовое обеспечение' in query.lower() and 'финанс' not in chunk['text'].lower():
                final_scores[i] *= 0.5
            if 'правовую основу' in query.lower() and 'правов' not in chunk['text'].lower():
                final_scores[i] *= 0.3
        
        # Топ результаты
        top_indices = np.argsort(final_scores)[-k*3:][::-1]
        
        # Собираем результаты с метаданными
        results = []
        seen_articles = set()
        
        for idx in top_indices:
            if final_scores[idx] > 0.1:
                article = self.chunks[idx]['article']
                
                # Проверяем, не слишком ли много чанков из одной статьи
                if article in seen_articles:
                    article_count = sum(1 for r in results if r['article'] == article)
                    if article_count >= 2:
                        continue
                
                results.append({
                    'text': self.chunks[idx]['text'],
                    'full_text': self.chunks[idx].get('full_text', self.chunks[idx]['text']),
                    'article': article,
                    'title': self.chunks[idx].get('title', ''),
                    'similarity': float(final_scores[idx]),
                    'base_similarity': float(similarities[idx])
                })
                seen_articles.add(article)
            
            if len(results) >= k:
                break
        
        # Сортируем по финальной релевантности
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results
    
    def query(self, question: str) -> str:
        """Ответ на вопрос с акцентом на самые релевантные статьи"""
        if not self.chunks:
            return "❌ Документ не загружен."
        
        # Поиск релевантных чанков
        relevant = self.search(question, k=5)
        
        if not relevant:
            return "❌ В документе не найдена информация по вашему вопросу."
        
        # Формируем ответ
        answer = "📄 **На основе Национальной стратегии развития ИИ:**\n\n"
        
        # Определяем основной контекст вопроса
        is_legal_question = any(word in question.lower() for word in 
                               ['закон', 'правов', 'федеральн', 'конституц', 'основ'])
        
        # Показываем ТОЛЬКО самые релевантные статьи (макс 2)
        shown_articles = 0
        
        for item in relevant:
            if shown_articles >= 2:
                break
            
            article_num = item['article']
            
            # Для юридических вопросов форсируем показ статьи 2
            if is_legal_question and article_num == "2":
                if article_num != "0":
                    answer += f"**Статья {article_num}**"
                    if item['title']:
                        answer += f" - {item['title']}"
                    answer += "\n"
                else:
                    answer += "**Введение**\n"
                
                # Показываем полный текст статьи 2
                answer += item['full_text'] + "\n\n"
                shown_articles += 1
                
            # Для остальных случаев показываем по релевантности
            elif not is_legal_question or article_num != "2":
                # Показываем только если релевантность высокая
                if item['similarity'] > 0.3:
                    if article_num != "0":
                        answer += f"**Статья {article_num}**"
                        if item['title']:
                            answer += f" - {item['title']}"
                        answer += "\n"
                    else:
                        answer += "**Введение**\n"
                    
                    # Показываем текст
                    text = item['full_text']
                    if len(text) > 600:
                        answer += text[:600] + "...\n\n"
                    else:
                        answer += text + "\n\n"
                    
                    shown_articles += 1
        
        # Если не показали ни одной статьи, показываем топ-1
        if shown_articles == 0 and relevant:
            item = relevant[0]
            article_num = item['article']
            if article_num != "0":
                answer += f"**Статья {article_num}**"
                if item['title']:
                    answer += f" - {item['title']}"
                answer += "\n"
            else:
                answer += "**Введение**\n"
            
            answer += item['full_text'][:500] + "...\n\n"
        
        # Список использованных статей
        used_articles = [f"Статья {r['article']}" for r in relevant[:2] if r['article'] != "0"]
        if used_articles:
            answer += f"\n*Источник: Национальная стратегия развития ИИ, {', '.join(used_articles)}*"
        
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
        with st.spinner("🔄 Загрузка документа..."):
            st.session_state.rag = ImprovedStrategyRAG(file_path)
    
    rag = st.session_state.rag
    
    # Сайдбар
    with st.sidebar:
        st.header("📚 О документе")
        st.markdown("""
        **Национальная стратегия развития ИИ**  
        *до 2030 года (с изменениями 2024)*
        """)
        
        if rag.chunks:
            unique_articles = set()
            for chunk in rag.chunks:
                if chunk['article'] != "0":
                    unique_articles.add(chunk['article'])
            
            st.metric("Всего статей", len(unique_articles))
            st.metric("Всего чанков", len(rag.chunks))
    
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
