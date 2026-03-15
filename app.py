"""
Профессиональная RAG система с реальными эмбеддингами и LLM
Запуск: streamlit run app.py
"""

import os
import streamlit as st
import numpy as np
from pathlib import Path
import json
import requests
from typing import List, Dict, Optional
import tempfile

# Настройка страницы
st.set_page_config(
    page_title="Профессиональный RAG чат",
    page_icon="🤖",
    layout="wide"
)

class ProfessionalRAG:
    """
    Продвинутая RAG система с реальными эмбеддингами через Ollama
    """
    
    def __init__(self, embedding_model="mxbai-embed-large", llm_model="gemma2:2b"):
        """
        Инициализация с моделями Ollama
        
        Args:
            embedding_model: Модель для эмбеддингов (рекомендуется mxbai-embed-large)
            llm_model: Модель для генерации ответов
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunks = []
        self.embeddings = None
        self.sources = []
        self.metadata = []
        
        # Проверяем доступность Ollama
        self._check_ollama()
        
        # Создаем папку для документов
        self.docs_folder = Path("./documents")
        self.docs_folder.mkdir(exist_ok=True)
        
    def _check_ollama(self):
        """Проверка доступности Ollama и моделей"""
        try:
            # Проверяем работает ли Ollama
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                st.sidebar.info(f"✅ Ollama доступна. Модели: {', '.join(model_names)}")
                
                # Проверяем нужные модели
                if self.embedding_model not in str(model_names):
                    st.sidebar.warning(f"⚠️ Модель {self.embedding_model} не найдена. Установите: ollama pull {self.embedding_model}")
                
                if self.llm_model not in str(model_names):
                    st.sidebar.warning(f"⚠️ Модель {self.llm_model} не найдена. Установите: ollama pull {self.llm_model}")
            else:
                st.sidebar.error("❌ Ollama не отвечает. Запустите: ollama serve")
        except requests.exceptions.ConnectionError:
            st.sidebar.error("❌ Ollama не запущена. Установите и запустите: https://ollama.com")
        except Exception as e:
            st.sidebar.error(f"❌ Ошибка: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Получение эмбеддингов через Ollama API
        
        Args:
            text: Текст для эмбеддинга
            
        Returns:
            Вектор эмбеддинга
        """
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text[:1000]  # Ограничиваем длину
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['embedding']
            else:
                st.error(f"Ошибка эмбеддинга: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Ошибка подключения к Ollama: {e}")
            return None
    
    def smart_chunking(self, text: str, source: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
        """
        Умное разбиение текста на чанки с перекрытием
        
        Args:
            text: Исходный текст
            source: Источник
            chunk_size: Размер чанка в символах
            overlap: Перекрытие между чанками
            
        Returns:
            Список чанков с метаданными
        """
        # Разбиваем на предложения (простая реализация)
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_len = len(sentence)
            
            # Если предложение слишком большое, разбиваем его
            if sentence_len > chunk_size:
                # Разбиваем большое предложение на части
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= chunk_size:
                        temp_chunk += word + " "
                    else:
                        if temp_chunk:
                            chunks.append({
                                'text': temp_chunk.strip(),
                                'source': source,
                                'chunk_id': len(chunks)
                            })
                        temp_chunk = word + " "
                if temp_chunk:
                    chunks.append({
                        'text': temp_chunk.strip(),
                        'source': source,
                        'chunk_id': len(chunks)
                    })
            else:
                # Добавляем предложение к текущему чанку
                if current_size + sentence_len + 1 <= chunk_size:
                    current_chunk += sentence + ". "
                    current_size += sentence_len + 2
                else:
                    # Сохраняем текущий чанк и начинаем новый с перекрытием
                    if current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'source': source,
                            'chunk_id': len(chunks)
                        })
                    
                    # Перекрытие: берем последние несколько предложений
                    words = current_chunk.split()
                    overlap_text = " ".join(words[-20:]) if len(words) > 20 else current_chunk
                    current_chunk = overlap_text + " " + sentence + ". "
                    current_size = len(current_chunk)
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'source': source,
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def add_document(self, text: str, source: str = "document"):
        """
        Добавление документа с генерацией эмбеддингов
        
        Args:
            text: Текст документа
            source: Название источника
        """
        with st.spinner(f"🔄 Обработка {source}..."):
            # Умное разбиение на чанки
            chunks = self.smart_chunking(text, source)
            
            if not chunks:
                st.warning(f"⚠️ Не удалось создать чанки для {source}")
                return 0
            
            # Прогресс бар
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Генерация эмбеддингов для каждого чанка
            chunk_embeddings = []
            valid_chunks = []
            
            for i, chunk in enumerate(chunks):
                status_text.text(f"Генерация эмбеддинга {i+1}/{len(chunks)}...")
                
                embedding = self.get_embedding(chunk['text'])
                if embedding:
                    chunk_embeddings.append(embedding)
                    valid_chunks.append(chunk)
                
                progress_bar.progress((i + 1) / len(chunks))
            
            progress_bar.empty()
            status_text.empty()
            
            if not chunk_embeddings:
                st.error("❌ Не удалось создать эмбеддинги")
                return 0
            
            # Добавляем в хранилище
            for chunk in valid_chunks:
                self.chunks.append(chunk['text'])
                self.sources.append(chunk['source'])
                self.metadata.append({
                    'chunk_id': chunk['chunk_id'],
                    'source': chunk['source']
                })
            
            # Обновляем эмбеддинги
            new_embeddings = np.array(chunk_embeddings)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            st.success(f"✅ Добавлено {len(valid_chunks)} чанков из {source}")
            return len(valid_chunks)
    
    def add_file(self, filepath: str):
        """Добавление файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.add_document(text, source=os.path.basename(filepath))
        except Exception as e:
            st.error(f"❌ Ошибка чтения {filepath}: {e}")
            return 0
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Гибридный поиск (семантический + ключевой)
        
        Args:
            query: Запрос
            k: Количество результатов
            
        Returns:
            Список релевантных чанков
        """
        if not self.chunks:
            return []
        
        # Получаем эмбеддинг запроса
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
        
        query_embedding = np.array(query_embedding)
        
        # Семантический поиск (косинусное сходство)
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        if norms.all() and query_norm:
            similarities = np.dot(self.embeddings, query_embedding) / (norms * query_norm)
        else:
            similarities = np.zeros(len(self.chunks))
        
        # Ключевой поиск (простые ключевые слова)
        query_words = set(query.lower().split())
        keyword_scores = []
        
        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            if query_words:
                overlap = len(query_words & chunk_words) / len(query_words)
            else:
                overlap = 0
            keyword_scores.append(overlap)
        
        # Комбинируем оценки (70% семантика, 30% ключевые слова)
        combined_scores = 0.7 * similarities + 0.3 * np.array(keyword_scores)
        
        # Получаем топ-K результатов
        top_indices = np.argsort(combined_scores)[-k*2:][::-1]
        
        # Дедупликация и сортировка
        results = []
        seen_sources = set()
        
        for idx in top_indices:
            if np.isnan(combined_scores[idx]):
                continue
                
            source = self.sources[idx]
            
            # Ограничиваем количество чанков из одного источника
            if source in seen_sources:
                source_count = sum(1 for r in results if r['source'] == source)
                if source_count >= 2:  # Максимум 2 чанка из одного источника
                    continue
            
            results.append({
                'text': self.chunks[idx],
                'source': source,
                'similarity': float(combined_scores[idx]),
                'semantic_score': float(similarities[idx]) if not np.isnan(similarities[idx]) else 0,
                'keyword_score': float(keyword_scores[idx])
            })
            seen_sources.add(source)
            
            if len(results) >= k:
                break
        
        return results
    
    def generate_answer(self, question: str, context: List[Dict]) -> str:
        """
        Генерация ответа через Ollama LLM
        
        Args:
            question: Вопрос пользователя
            context: Найденный контекст
            
        Returns:
            Ответ
        """
        if not context:
            return "❌ Не найдено релевантной информации в документах."
        
        # Формируем контекст
        context_text = ""
        for i, ctx in enumerate(context, 1):
            context_text += f"\n[Документ {i} - {ctx['source']}]\n{ctx['text']}\n"
        
        # Создаем промпт на русском
        prompt = f"""Ты - ассистент, который отвечает на вопросы строго на основе предоставленного контекста.

КОНТЕКСТ:
{context_text}

ИНСТРУКЦИИ:
1. Отвечай ТОЛЬКО на русском языке
2. Используй информацию только из предоставленного контекста
3. Если в контексте нет информации для ответа, скажи "Я не нашел информации об этом в документах"
4. Будь точным и конкретным
5. Если уместно, указывай источники информации

ВОПРОС: {question}

ОТВЕТ (на русском языке):"""
        
        try:
            # Отправляем запрос в Ollama
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Низкая температура для точности
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"❌ Ошибка генерации: {response.status_code}"
                
        except Exception as e:
            return f"❌ Ошибка подключения к Ollama: {e}"
    
    def query(self, question: str, k: int = 5, show_scores: bool = False) -> str:
        """
        Полный цикл RAG: поиск + генерация
        
        Args:
            question: Вопрос
            k: Количество чанков для поиска
            show_scores: Показывать ли оценки релевантности
            
        Returns:
            Ответ
        """
        # Поиск релевантных чанков
        relevant = self.hybrid_search(question, k=k)
        
        if not relevant:
            return "❌ Не найдено релевантной информации."
        
        # Генерация ответа
        answer = self.generate_answer(question, relevant)
        
        # Добавляем информацию о скорах если нужно
        if show_scores and relevant:
            answer += "\n\n---\n**Оценки релевантности:**\n"
            for r in relevant:
                answer += f"- {r['source']}: {r['similarity']:.3f} (семантика: {r['semantic_score']:.3f}, ключи: {r['keyword_score']:.3f})\n"
        
        return answer

# Инициализация RAG в сессии
@st.cache_resource
def init_rag():
    """Инициализация RAG системы"""
    return ProfessionalRAG(
        embedding_model="mxbai-embed-large",  # Лучшая модель для эмбеддингов
        llm_model="gemma2:2b"  # Можно заменить на другую модель
    )

def main():
    st.title("🤖 Профессиональный RAG чат с документами")
    st.markdown("---")
    
    # Инициализация
    rag = init_rag()
    
    # Сайдбар с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Параметры поиска
        k_results = st.slider(
            "Количество результатов поиска",
            min_value=1,
            max_value=10,
            value=5,
            help="Больше результатов = больше контекста, но дороже"
        )
        
        show_scores = st.checkbox(
            "Показывать оценки релевантности",
            value=False,
            help="Отображать метрики качества поиска"
        )
        
        st.divider()
        
        # Модели
        st.subheader("🤖 Модели")
        st.info(f"**Эмбеддинги:** {rag.embedding_model}")
        st.info(f"**LLM:** {rag.llm_model}")
        
        st.divider()
        
        # Загрузка документов
        st.header("📁 Загрузка документов")
        
        uploaded_files = st.file_uploader(
            "Выберите текстовые файлы",
            type=['txt', 'md'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Сохраняем временно
                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as f:
                    content = uploaded_file.read().decode('utf-8')
                    f.write(content)
                    temp_path = f.name
                
                # Добавляем в RAG
                rag.add_file(temp_path)
                
                # Удаляем временный файл
                os.unlink(temp_path)
            
            st.rerun()
        
        st.divider()
        
        # Список загруженных документов
        st.header("📋 Документы в базе")
        if rag.sources:
            unique_sources = set(rag.sources)
            for source in unique_sources:
                count = rag.sources.count(source)
                st.write(f"📄 {source} ({count} чанков)")
            
            st.write(f"**Всего чанков:** {len(rag.chunks)}")
        else:
            st.info("📭 Нет загруженных документов")
            
            # Пример документа
            with st.expander("📝 Загрузить пример"):
                if st.button("Добавить пример документа"):
                    sample_text = """RAG (Retrieval-Augmented Generation) - это метод, который объединяет поиск информации с генерацией текста.
Система сначала ищет релевантные документы в базе знаний, а затем использует их как контекст для генерации ответов.
Преимущества RAG включают уменьшение галлюцинаций и возможность цитирования источников.
Гибридный поиск в RAG комбинирует семантическое сходство с ключевыми словами для лучших результатов."""
                    rag.add_document(sample_text, source="пример.txt")
                    st.success("✅ Пример добавлен!")
                    st.rerun()
    
    # Основной интерфейс
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Задайте вопрос")
        
        # История сообщений
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Отображение истории
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Ввод вопроса
        if prompt := st.chat_input("Введите ваш вопрос по документам..."):
            # Добавляем вопрос пользователя
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Генерируем ответ
            with st.chat_message("assistant"):
                with st.spinner("🔍 Ищу ответ в документах..."):
                    response = rag.query(prompt, k=k_results, show_scores=show_scores)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.subheader("ℹ️ Информация")
        
        with st.expander("📌 Как улучшить качество"):
            st.markdown("**Рекомендации:**\n\n1. **Установите лучшие модели:**\n```\nollama pull mxbai-embed-large\nollama pull gemma2:2b\nollama pull llama3.2:3b\n```\n\n2. **Качество документов:**\n- Используйте чистый текст\n- Удалите мусор\n- Разбивайте на логические части\n\n3. **Настройки поиска:**\n- 3-5 чанков оптимально\n- Включите показ оценок для отладки")
        
        with st.expander("💡 Примеры вопросов"):
            st.markdown("**Хорошие вопросы:**\n- \"Что такое RAG система?\"\n- \"Какие преимущества у гибридного поиска?\"\n- \"Как уменьшить галлюцинации?\"\n\n**Плохие вопросы:**\n- \"Что там?\" (слишком общий)\n- \"Расскажи всё\" (неконкретный)")
        
        # Кнопка очистки истории
        if st.button("🗑️ Очистить историю"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
