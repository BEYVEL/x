"""
Фокусированный RAG чат - Только на основе файла filerag.txt
Никаких загрузок от пользователей, только ваш документ
"""

import streamlit as st
import numpy as np
import os
from pathlib import Path
from typing import List, Dict
import hashlib

# Настройка страницы - минимализм
st.set_page_config(
    page_title="Национальная стратегия ИИ",
    page_icon="📄",
    layout="centered"
)

# Скрываем лишние элементы Streamlit
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

class StrategyRAG:
    """RAG система только для одного документа - Национальной стратегии ИИ"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.chunks = []
        self.embeddings = None
        self.sources = []
        self.dim = 512  # Увеличил размерность для лучшей точности
        
        # Загружаем и обрабатываем документ при инициализации
        self._load_document()
    
    def _smart_chunking(self, text: str) -> List[str]:
        """
        Умное разбиение на чанки по статьям и разделам документа
        Сохраняет структуру документа
        """
        # Разбиваем по номерам статей (пунктов)
        import re
        
        # Ищем все пункты (1., 2., 3., и т.д.)
        chunks = []
        
        # Разделяем документ на логические блоки по пунктам
        lines = text.split('\n')
        current_chunk = ""
        
        for line in lines:
            # Если строка начинается с цифры и точки (пункт документа)
            if re.match(r'^\d+\.', line.strip()):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Дополнительно разбиваем слишком большие чанки
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > 1000:  # Если чанк слишком большой
                # Разбиваем на подчасти по 500 символов с перекрытием
                words = chunk.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) < 800:
                        temp_chunk += word + " "
                    else:
                        if temp_chunk:
                            final_chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "
                if temp_chunk:
                    final_chunks.append(temp_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Улучшенный эмбеддинг с учетом важных терминов
        """
        words = text.lower().split()
        
        # Словарь важных терминов для ИИ стратегии (веса для ключевых слов)
        important_terms = {
            'искусственный интеллект': 3.0,
            'стратегия': 2.0,
            'технологии': 1.5,
            'развитие': 1.5,
            'национальный': 1.5,
            'федеральный': 1.5,
            'правительство': 1.5,
            'президент': 1.5,
            'российская федерация': 2.0,
            'данные': 1.5,
            'безопасность': 1.5,
            'этика': 1.5,
            'цифровой': 1.5,
            'экономика': 1.5
        }
        
        features = np.zeros(self.dim)
        
        # Униграммы с весами
        for word in words:
            idx = abs(hash(word)) % self.dim
            # Базовая частота
            features[idx] += 1
            
            # Проверяем, есть ли слово в важных терминах
            for term, weight in important_terms.items():
                if term in text.lower():
                    term_idx = abs(hash(term)) % self.dim
                    features[term_idx] += weight
        
        # Биграммы для контекста
        for i in range(len(words)-1):
            bigram = words[i] + " " + words[i+1]
            idx = abs(hash(bigram)) % self.dim
            features[idx] += 0.3
        
        # Нормализация
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.tolist()
    
    def _load_document(self):
        """Загрузка и обработка документа"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Умное разбиение на чанки
            chunks = self._smart_chunking(text)
            
            if not chunks:
                st.error("❌ Не удалось создать чанки из документа")
                return
            
            # Генерация эмбеддингов
            with st.spinner("🔄 Обработка документа..."):
                chunk_embeddings = []
                for chunk in chunks:
                    embedding = self._get_embedding(chunk)
                    chunk_embeddings.append(embedding)
                
                # Сохраняем чанки и эмбеддинги
                self.chunks = chunks
                self.sources = ["Национальная стратегия ИИ"] * len(chunks)
                self.embeddings = np.array(chunk_embeddings)
                
                st.sidebar.success(f"✅ Документ загружен: {len(chunks)} чанков")
                
        except Exception as e:
            st.error(f"❌ Ошибка загрузки документа: {e}")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Поиск релевантных чанков с порогом релевантности
        """
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
        
        # Усиление для чанков с номерами статей (прямые ссылки)
        for i, chunk in enumerate(self.chunks):
            # Если в чанке есть номер статьи, который совпадает с числом в запросе
            numbers = re.findall(r'\d+', query)
            for num in numbers:
                if num in chunk[:50]:  # Проверяем начало чанка
                    similarities[i] += 0.2
        
        # Получаем топ результаты выше порога
        top_indices = np.argsort(similarities)[-k*2:][::-1]
        
        results = []
        seen_content = set()
        
        for idx in top_indices:
            if similarities[idx] > 0.15:  # Порог релевантности
                # Дедупликация похожего контента
                chunk_preview = self.chunks[idx][:100]
                if chunk_preview not in seen_content:
                    results.append({
                        'text': self.chunks[idx],
                        'source': self.sources[idx],
                        'similarity': float(similarities[idx]),
                        'article_num': self._extract_article_number(self.chunks[idx])
                    })
                    seen_content.add(chunk_preview)
            
            if len(results) >= k:
                break
        
        return results
    
    def _extract_article_number(self, text: str) -> str:
        """Извлечение номера статьи из текста"""
        import re
        match = re.match(r'^(\d+)\.', text.strip())
        if match:
            return f"Статья {match.group(1)}"
        return ""
    
    def generate_answer(self, question: str, context: List[Dict]) -> str:
        """
        Генерация структурированного ответа на основе найденного контекста
        """
        if not context:
            return "❌ В документе не найдена информация по вашему вопросу."
        
        # Сортируем по релевантности
        context.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Формируем ответ
        answer = "📄 **На основе Национальной стратегии развития ИИ:**\n\n"
        
        # Добавляем самые релевантные части
        used_articles = set()
        
        for item in context[:3]:  # Берем топ-3
            article_num = self._extract_article_number(item['text'])
            
            if article_num and article_num not in used_articles:
                answer += f"**{article_num}**\n"
                used_articles.add(article_num)
            
            # Извлекаем основное содержание
            text = item['text']
            
            # Если текст слишком длинный, показываем релевантную часть
            if len(text) > 500:
                # Пытаемся найти предложения с ключевыми словами из вопроса
                sentences = text.replace('!', '.').replace('?', '.').split('.')
                question_words = set(question.lower().split())
                
                relevant_sentences = []
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    sent_words = set(sent.lower().split())
                    if len(question_words & sent_words) > 1:
                        relevant_sentences.append(sent)
                
                if relevant_sentences:
                    answer += ". ".join(relevant_sentences[:2]) + ".\n\n"
                else:
                    # Если не нашли специфичных предложений, показываем начало
                    answer += text[:300] + "...\n\n"
            else:
                answer += text + "\n\n"
        
        # Добавляем ссылки на статьи
        if used_articles:
            answer += f"\n*Источник: Национальная стратегия развития ИИ, {', '.join(used_articles)}*"
        
        return answer
    
    def query(self, question: str) -> str:
        """Полный цикл RAG"""
        if not self.chunks:
            return "❌ Документ не загружен."
        
        # Поиск релевантных чанков
        relevant = self.search(question, k=5)
        
        if not relevant:
            return "❌ В документе не найдена информация по вашему вопросу."
        
        # Генерация ответа
        return self.generate_answer(question, relevant)

def main():
    st.title("📄 Национальная стратегия развития ИИ")
    st.markdown("*Чат на основе официального документа (с изменениями 2024 г.)*")
    
    # Путь к вашему файлу (должен быть в той же папке, что и app.py)
    file_path = "filerag.txt"
    
    # Проверяем существование файла
    if not os.path.exists(file_path):
        st.error(f"❌ Файл {file_path} не найден!")
        st.info("Пожалуйста, убедитесь, что файл filerag.txt находится в той же папке, что и app.py")
        
        # Показываем содержимое папки для отладки
        st.write("Файлы в текущей папке:")
        for f in os.listdir('.'):
            st.write(f"- {f}")
        return
    
    # Инициализация RAG
    if 'rag' not in st.session_state:
        st.session_state.rag = StrategyRAG(file_path)
    
    rag = st.session_state.rag
    
    # Сайдбар с информацией о документе
    with st.sidebar:
        st.header("📚 О документе")
        st.markdown("""
        **Национальная стратегия развития ИИ**  
        *до 2030 года (с изменениями 2024)*
        
        **Структура документа:**
        - I. Общие положения (ст. 1-5)
        - II. Развитие ИИ в России и мире (ст. 6-23)
        - III. Основные принципы (ст. 24)
        - IV. Цели и задачи (ст. 25-41)
        - V. Механизмы реализации (ст. 42-50)
        """)
        
        if rag.chunks:
            st.metric("Всего чанков", len(rag.chunks))
            st.metric("Размер эмбеддингов", f"{rag.embeddings.shape[1]}d")
    
    # История чата
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Отображение истории
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Ввод вопроса
    if prompt := st.chat_input("Задайте вопрос о стратегии развития ИИ..."):
        # Добавляем вопрос
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Получаем ответ
        with st.chat_message("assistant"):
            with st.spinner("🔍 Ищу в документе..."):
                response = rag.query(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
