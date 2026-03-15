"""
Простое RAG приложение для вопросов по документам
Запуск: streamlit run app.py
"""

import os
import numpy as np
import hashlib
from pathlib import Path
import streamlit as st

# Настройка страницы
st.set_page_config(
    page_title="Чат с документами",
    page_icon="📚",
    layout="wide"
)

class SimpleRAG:
    def __init__(self, model_name="gemma:2b"):
        self.model_name = model_name
        self.chunks = []
        self.embeddings = None
        self.sources = []
        
        # Создаем папку для документов
        self.docs_folder = Path("./documents")
        self.docs_folder.mkdir(exist_ok=True)
    
    def _hash_features(self, text, dim=256):
        """Создание простых признаков на основе хеша"""
        words = text.lower().split()[:100]
        features = np.zeros(dim)
        
        for word in words:
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
            features[idx] += 1
        
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def add_text(self, text, source="документ"):
        """Добавление текста в базу знаний"""
        # Разбиваем на чанки
        chunk_size = 500
        overlap = 50
        chunks = []
        sources = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if chunk and len(chunk) > 50:  # Игнорируем слишком короткие чанки
                chunks.append(chunk)
                sources.append(source)
        
        if not chunks:
            return 0
        
        # Создаем эмбеддинги
        new_embeddings = np.array([self._hash_features(chunk) for chunk in chunks])
        
        # Сохраняем
        self.chunks.extend(chunks)
        self.sources.extend(sources)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        return len(chunks)
    
    def add_file(self, filepath):
        """Добавление текстового файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.add_text(text, source=os.path.basename(filepath))
        except Exception as e:
            st.error(f"Ошибка при чтении {filepath}: {e}")
            return 0
    
    def search(self, query, k=3):
        """Поиск наиболее релевантных чанков"""
        if not self.chunks:
            return []
        
        query_features = self._hash_features(query)
        similarities = np.dot(self.embeddings, query_features)
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks[idx],
                'source': self.sources[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def query(self, question, k=3):
        """Ответ на вопрос"""
        relevant = self.search(question, k=k)
        
        if not relevant:
            return "❌ Не найдено релевантной информации в документах."
        
        # Формируем ответ
        response = f"📚 **На основе документов:**\n\n"
        
        for i, r in enumerate(relevant, 1):
            # Обрезаем текст для предпросмотра
            preview = r['text'][:300] + "..." if len(r['text']) > 300 else r['text']
            response += f"**{i}. Источник: {r['source']}**\n"
            response += f"{preview}\n\n"
            response += f"---\n\n"
        
        return response

# Инициализация RAG
@st.cache_resource
def init_rag():
    rag = SimpleRAG()
    
    # Загружаем существующие документы
    docs_folder = Path("./documents")
    docs_folder.mkdir(exist_ok=True)
    
    for file in docs_folder.glob("*.txt"):
        rag.add_file(str(file))
    
    return rag

# Основное приложение
def main():
    # Заголовок
    st.title("📚 Чат с вашими документами")
    st.markdown("---")
    
    # Инициализируем RAG
    rag = init_rag()
    
    # Сайдбар для управления документами
    with st.sidebar:
        st.header("📁 Управление документами")
        
        # Загрузка файлов
        uploaded_files = st.file_uploader(
            "Загрузите текстовые файлы",
            type=['txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Сохраняем файл
                docs_folder = Path("./documents")
                docs_folder.mkdir(exist_ok=True)
                
                file_path = docs_folder / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Добавляем в RAG
                rag.add_file(str(file_path))
            
            st.success(f"✅ Загружено файлов: {len(uploaded_files)}")
            st.rerun()
        
        st.divider()
        
        # Список загруженных документов
        st.header("📋 Загруженные документы")
        docs_folder = Path("./documents")
        files = list(docs_folder.glob("*.txt"))
        
        if files:
            for file in files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {file.name}")
                with col2:
                    # Кнопка просмотра
                    if st.button("👁️", key=f"view_{file.name}"):
                        with open(file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        st.session_state[f"content_{file.name}"] = content
                
                with col3:
                    # Кнопка удаления
                    if st.button("🗑️", key=f"del_{file.name}"):
                        file.unlink()
                        st.rerun()
                
                # Показываем содержимое если есть
                if f"content_{file.name}" in st.session_state:
                    with st.expander(f"Содержимое {file.name}"):
                        st.text(st.session_state[f"content_{file.name}"][:500] + "...")
        else:
            st.info("📭 Нет загруженных документов")
        
        st.divider()
        
        # Статистика
        st.header("📊 Статистика")
        st.write(f"**Всего чанков:** {len(rag.chunks)}")
        st.write(f"**Документов:** {len(files)}")
        
        if rag.chunks:
            st.write(f"**Размер эмбеддингов:** {rag.embeddings.shape}")
    
    # Основной чат
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Задайте вопрос по документам")
        
        # Поле ввода вопроса
        question = st.text_input(
            "Введите ваш вопрос:",
            placeholder="Например: О чем говорится в документах?",
            key="question_input"
        )
        
        # Кнопка отправки
        if st.button("🔍 Спросить", type="primary"):
            if question:
                with st.spinner("🔎 Ищу ответ в документах..."):
                    answer = rag.query(question)
                    
                    # Сохраняем в историю
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    
                    st.session_state.history.append({
                        "question": question,
                        "answer": answer
                    })
    
    with col2:
        st.subheader("ℹ️ Информация")
        st.info(
            """
            **Как использовать:**
            1. Загрузите текстовые файлы в боковой панели
            2. Задайте вопрос по содержимому
            3. Получите ответ с указанием источников
            
            **Поддерживаемые форматы:**
            - Текстовые файлы (.txt)
            """
        )
    
    # История вопросов
    if "history" in st.session_state and st.session_state.history:
        st.markdown("---")
        st.subheader("📝 История вопросов")
        
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"❓ {item['question'][:50]}..."):
                st.markdown(item['answer'])

# Запуск приложения
if __name__ == "__main__":
    main()
