"""
Минималистичный RAG чат - Только суть
Чистый интерфейс, полные ответы, никакого шума
"""

import streamlit as st
import numpy as np
import requests
from pathlib import Path
import tempfile
import os
from typing import List, Dict

# Настройка страницы - МИНИМАЛИЗМ
st.set_page_config(
    page_title="RAG Чат",
    page_icon="💬",
    layout="centered"  # Центрированный, не широкий
)

# Скрываем дефолтный Streamlit мусор
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {margin-top: -80px;}
    .stButton>button {width: 100%;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

class MinimalRAG:
    """Минималистичная RAG система - только необходимое"""
    
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.sources = []
        self.dim = 384
        
        # Пытаемся получить API ключ (опционально)
        self.openrouter_key = st.secrets.get("OPENROUTER_API_KEY", None)
    
    def _get_embedding(self, text: str) -> List[float]:
        """Улучшенный эмбеддинг с лучшей семантикой"""
        words = text.lower().split()
        
        # Используем не только униграммы, но и биграммы для лучшего контекста
        features = np.zeros(self.dim)
        
        # Униграммы
        for word in words[:200]:
            idx = abs(hash(word)) % self.dim
            features[idx] += 1
        
        # Биграммы (для лучшего понимания контекста)
        for i in range(len(words)-1):
            bigram = words[i] + " " + words[i+1]
            idx = abs(hash(bigram)) % self.dim
            features[idx] += 0.5  # Меньший вес для биграмм
        
        # Нормализация
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.tolist()
    
    def add_document(self, text: str, source: str = "документ"):
        """Добавление документа с умным чанкингом"""
        # Разбиваем на предложения, потом собираем в чанки
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение большое, разбиваем его
            if len(sentence) > 300:
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) < 300:
                        temp_chunk += word + " "
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
            else:
                # Добавляем к текущему чанку
                if len(current_chunk) + len(sentence) < 500:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Фильтруем пустые и слишком короткие чанки
        chunks = [c for c in chunks if len(c) > 50]
        
        if not chunks:
            return 0
        
        # Генерируем эмбеддинги
        chunk_embeddings = []
        for chunk in chunks:
            embedding = self._get_embedding(chunk)
            chunk_embeddings.append(embedding)
        
        # Сохраняем
        self.chunks.extend(chunks)
        self.sources.extend([source] * len(chunks))
        
        if self.embeddings is None:
            self.embeddings = np.array(chunk_embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, np.array(chunk_embeddings)])
        
        return len(chunks)
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Поиск с реранкингом для лучшей точности"""
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
        
        # Дополнительный реранкинг по ключевым словам
        query_words = set(query.lower().split())
        keyword_scores = []
        
        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words) / max(len(query_words), 1)
            keyword_scores.append(overlap * 0.3)  # 30% вес
        
        # Финальные оценки
        final_scores = similarities + np.array(keyword_scores)
        
        # Топ результаты
        top_indices = np.argsort(final_scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if final_scores[idx] > 0.1:  # Минимальный порог
                results.append({
                    'text': self.chunks[idx],
                    'source': self.sources[idx],
                    'score': float(final_scores[idx])
                })
        
        return results
    
    def generate_answer(self, question: str, context: List[Dict]) -> str:
        """Генерация ПОЛНОГО ответа без обрезания"""
        if not context:
            return "❌ Информация не найдена."
        
        # Собираем ВЕСЬ релевантный контекст
        full_context = "\n\n".join([c['text'] for c in context])
        
        # Если есть OpenRouter, используем его для лучших ответов
        if self.openrouter_key:
            try:
                prompt = f"""Контекст: {full_context}

Вопрос: {question}

Дай полный, развернутый ответ на русском языке, используя только информацию из контекста.
Не обрезай ответ, пиши все детали."""

                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "mistralai/mistral-7b-instruct:free",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 1000  # Увеличиваем лимит
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
            except:
                pass  # Если API не работает, используем fallback
        
        # Fallback: собираем полный контекст
        answer = "📄 **На основе документов:**\n\n"
        
        # Группируем по источникам для лучшей читаемости
        sources_dict = {}
        for item in context:
            if item['source'] not in sources_dict:
                sources_dict[item['source']] = []
            sources_dict[item['source']].append(item['text'])
        
        for source, texts in sources_dict.items():
            answer += f"**{source}:**\n"
            for text in texts:
                # Добавляем текст ПОЛНОСТЬЮ, не обрезаем
                answer += f"{text}\n\n"
        
        return answer

def main():
    # Заголовок - минимально
    st.title("💬 Чат с документами")
    
    # Инициализация
    if 'rag' not in st.session_state:
        st.session_state.rag = MinimalRAG()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    rag = st.session_state.rag
    
    # Минимальный сайдбар
    with st.sidebar:
        st.markdown("### 📁 Загрузка")
        
        uploaded_file = st.file_uploader(
            "Выберите файл",
            type=['txt'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as f:
                content = uploaded_file.read().decode('utf-8', errors='ignore')
                f.write(content)
                count = rag.add_document(content, source=uploaded_file.name)
                os.unlink(f.name)
            
            if count > 0:
                st.success(f"✅ Загружено: {uploaded_file.name}")
                st.rerun()
        
        if st.button("📝 Пример документа"):
            sample = """Стратегия развития искусственного интеллекта в России определяет основные направления развития технологий ИИ до 2030 года.

Правовую основу настоящей Стратегии составляют Конституция Российской Федерации, федеральные законы от 27 июля 2006 г. № 149-ФЗ "Об информации, информационных технологиях и о защите информации" и от 27 июля 2006 г. № 152-ФЗ "О персональных данных".

Основными источниками финансового обеспечения реализации настоящей Стратегии являются средства федерального бюджета, бюджетов субъектов Российской Федерации, а также внебюджетные источники.

В целях аналитической поддержки реализации настоящей Стратегии проводятся научные исследования, направленные на прогнозирование развития технологий искусственного интеллекта и оценку эффективности внедрения таких технологий в различные отрасли экономики.

Ключевыми направлениями развития ИИ в России являются: компьютерное зрение, обработка естественного языка, распознавание и синтез речи, рекомендательные системы и системы поддержки принятия решений."""
            
            rag.add_document(sample, source="стратегия.txt")
            st.success("✅ Пример загружен")
            st.rerun()
        
        # Минимальная статистика
        if rag.chunks:
            st.markdown("---")
            st.markdown(f"📊 **{len(rag.chunks)}** чанков")
            st.markdown(f"📚 **{len(set(rag.sources))}** документов")
    
    # Отображение истории чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Ввод вопроса
    if prompt := st.chat_input("Задайте вопрос..."):
        # Добавляем вопрос
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Получаем ответ
        with st.chat_message("assistant"):
            with st.spinner("🔍 Поиск..."):
                # Ищем релевантные чанки
                relevant = rag.search(prompt, k=5)
                
                if not relevant:
                    response = "❌ Информация не найдена."
                else:
                    # Генерируем полный ответ
                    response = rag.generate_answer(prompt, relevant)
                
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
