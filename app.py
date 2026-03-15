"""
Zero-Install RAG System - No Ollama required!
Works on Streamlit Cloud with free APIs
"""

import streamlit as st
import numpy as np
import requests
import hashlib
import json
from pathlib import Path
import tempfile
import os
from typing import List, Dict, Optional
import time

# Настройка страницы
st.set_page_config(
    page_title="RAG Чат без установки",
    page_icon="🚀",
    layout="wide"
)

class ZeroInstallRAG:
    """
    RAG система которая работает без установки дополнительного ПО
    Использует бесплатные API или встроенные функции
    """
    
    def __init__(self):
        # Пытаемся получить API ключи из секретов Streamlit
        self.openrouter_key = st.secrets.get("OPENROUTER_API_KEY", None)
        self.huggingface_key = st.secrets.get("HUGGINGFACE_API_KEY", None)
        
        # Выбираем лучший доступный бэкенд
        self.backend = self._detect_best_backend()
        
        # Хранилище документов
        self.chunks = []
        self.embeddings = None
        self.sources = []
        self.dim = 384  # Размерность эмбеддингов
        
        # Статистика
        self.api_calls = 0
        self.fallback_used = 0
        
        st.sidebar.success(f"✅ Бэкенд: {self.backend}")
    
    def _detect_best_backend(self) -> str:
        """Автоматически выбирает лучший доступный бэкенд"""
        if self.openrouter_key:
            return "openrouter"
        elif self.huggingface_key:
            return "huggingface"
        else:
            return "fallback"
    
    def _get_embedding_openrouter(self, text: str) -> List[float]:
        """Получение эмбеддингов через OpenRouter (бесплатно)"""
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "sentence-transformers/all-mpnet-base-v2",
                    "input": text[:1000]  # Ограничиваем длину
                },
                timeout=10
            )
            
            if response.status_code == 200:
                self.api_calls += 1
                return response.json()['data'][0]['embedding']
            else:
                st.warning(f"OpenRouter ошибка: {response.status_code}, использую fallback")
                self.fallback_used += 1
                return self._get_embedding_fallback(text)
                
        except Exception as e:
            st.warning(f"OpenRouter недоступен: {e}, использую fallback")
            self.fallback_used += 1
            return self._get_embedding_fallback(text)
    
    def _get_embedding_huggingface(self, text: str) -> List[float]:
        """Получение эмбеддингов через HuggingFace Inference API"""
        try:
            API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
            headers = {"Authorization": f"Bearer {self.huggingface_key}"}
            
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": text[:1000]},
                timeout=10
            )
            
            if response.status_code == 200:
                self.api_calls += 1
                return response.json()[0]
            else:
                return self._get_embedding_fallback(text)
                
        except Exception:
            return self._get_embedding_fallback(text)
    
    def _get_embedding_fallback(self, text: str) -> List[float]:
        """
        Простой эмбеддинг на основе хеша - работает всегда,
        даже без интернета!
        """
        words = text.lower().split()[:100]
        features = np.zeros(self.dim)
        
        for word in words:
            # Используем встроенный hash (всегда доступен)
            idx = abs(hash(word)) % self.dim
            features[idx] += 1
        
        # Нормализация
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.tolist()
    
    def get_embedding(self, text: str) -> List[float]:
        """Универсальный метод получения эмбеддингов"""
        if self.backend == "openrouter" and self.openrouter_key:
            return self._get_embedding_openrouter(text)
        elif self.backend == "huggingface" and self.huggingface_key:
            return self._get_embedding_huggingface(text)
        else:
            return self._get_embedding_fallback(text)
    
    def generate_answer_openrouter(self, question: str, context: str) -> str:
        """Генерация ответа через OpenRouter"""
        prompt = f"""Ты ассистент, отвечающий на вопросы на русском языке.
Используй ТОЛЬКО информацию из контекста ниже.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ (на русском, только на основе контекста):"""
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mistralai/mistral-7b-instruct:free",  # Бесплатная модель
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"❌ Ошибка API: {response.status_code}"
                
        except Exception as e:
            return f"❌ Ошибка: {str(e)}"
    
    def generate_answer_huggingface(self, question: str, context: str) -> str:
        """Генерация ответа через HuggingFace"""
        prompt = f"""<|system|>
Ты ассистент, отвечающий на русском языке. Используй только контекст.</s>
<|user|>
Контекст: {context}

Вопрос: {question}</s>
<|assistant|>"""
        
        try:
            API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
            headers = {"Authorization": f"Bearer {self.huggingface_key}"}
            
            response = requests.post(
                API_URL,
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 300,
                        "temperature": 0.3
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()[0]['generated_text']
                # Извлекаем только ответ ассистента
                if "<|assistant|>" in result:
                    return result.split("<|assistant|>")[-1].strip()
                return result
            else:
                return self._generate_answer_fallback(question, context)
                
        except Exception:
            return self._generate_answer_fallback(question, context)
    
    def _generate_answer_fallback(self, question: str, context: str) -> str:
        """
        Простой ответ на основе ключевых слов - работает всегда
        """
        # Ищем предложения с ключевыми словами из вопроса
        question_words = set(question.lower().split())
        sentences = context.replace('!', '.').replace('?', '.').split('.')
        
        best_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Считаем совпадения слов
            sentence_words = set(sentence.lower().split())
            matches = len(question_words & sentence_words)
            if matches > 0:
                best_sentences.append((sentence, matches))
        
        # Сортируем по релевантности
        best_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if best_sentences:
            answer = "📄 **На основе документов:**\n\n"
            for sent, score in best_sentences[:3]:
                answer += f"• {sent.strip()}\n\n"
            return answer
        else:
            return "❌ Не найдено информации в документах."
    
    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """Универсальный метод генерации ответа"""
        # Формируем контекст из найденных чанков
        context = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in context_chunks])
        
        if self.backend == "openrouter" and self.openrouter_key:
            return self.generate_answer_openrouter(question, context)
        elif self.backend == "huggingface" and self.huggingface_key:
            return self.generate_answer_huggingface(question, context)
        else:
            return self._generate_answer_fallback(question, context)
    
    def add_document(self, text: str, source: str = "document"):
        """Добавление документа с эмбеддингами"""
        # Простое разбиение на чанки
        chunk_size = 500
        overlap = 50
        
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if chunk and len(chunk) > 50:
                chunks.append(chunk)
        
        if not chunks:
            return 0
        
        # Прогресс бар
        progress_bar = st.progress(0)
        
        # Генерация эмбеддингов
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            chunk_embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(chunks))
        
        progress_bar.empty()
        
        # Сохраняем
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        self.sources.extend([source] * len(chunks))
        
        # Обновляем матрицу эмбеддингов
        new_embeddings = np.array(chunk_embeddings)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        return len(chunks)
    
    def add_file(self, filepath: str):
        """Добавление файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.add_document(text, source=os.path.basename(filepath))
        except Exception as e:
            st.error(f"Ошибка чтения {filepath}: {e}")
            return 0
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Поиск релевантных чанков"""
        if not self.chunks:
            return []
        
        # Получаем эмбеддинг запроса
        query_emb = np.array(self.get_embedding(query))
        
        # Косинусное сходство
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Топ-K результатов
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if not np.isnan(similarities[idx]):
                results.append({
                    'text': self.chunks[idx],
                    'source': self.sources[idx],
                    'similarity': float(similarities[idx])
                })
        
        return results
    
    def query(self, question: str, k: int = 3) -> str:
        """Полный RAG цикл"""
        # Поиск
        relevant = self.search(question, k=k)
        
        if not relevant:
            return "❌ Не найдено релевантной информации."
        
        # Генерация ответа
        answer = self.generate_answer(question, relevant)
        
        # Добавляем информацию о источниках
        answer += "\n\n---\n**📚 Источники:**\n"
        for r in relevant:
            preview = r['text'][:100] + "..." if len(r['text']) > 100 else r['text']
            answer += f"• {r['source']}: {preview}\n"
        
        return answer

def main():
    st.title("🚀 RAG Чат - НИЧЕГО НЕ НАДО УСТАНАВЛИВАТЬ!")
    st.markdown("Просто загрузите документы и задавайте вопросы")
    
    # Инициализация RAG
    if 'rag' not in st.session_state:
        st.session_state.rag = ZeroInstallRAG()
    
    rag = st.session_state.rag
    
    # Сайдбар
    with st.sidebar:
        st.header("📁 Документы")
        
        # Загрузка файлов
        uploaded_files = st.file_uploader(
            "Загрузите текстовые файлы",
            type=['txt', 'md', 'csv'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Сохраняем временно
                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as f:
                    content = uploaded_file.read().decode('utf-8', errors='ignore')
                    f.write(content)
                    temp_path = f.name
                
                # Добавляем в RAG
                count = rag.add_file(temp_path)
                if count > 0:
                    st.success(f"✅ {uploaded_file.name}: {count} чанков")
                
                # Удаляем временный файл
                os.unlink(temp_path)
        
        st.divider()
        
        # Пример документа
        with st.expander("📝 Добавить пример"):
            if st.button("Загрузить пример документа"):
                sample = """RAG (Retrieval-Augmented Generation) - это метод, который улучшает работу языковых моделей.
                
Система RAG состоит из двух основных компонентов:
1. Поисковая система - находит релевантные документы
2. Генератор - создает ответ на основе найденных документов

Преимущества RAG:
- Уменьшение галлюцинаций
- Возможность обновлять знания без переобучения
- Прозрачность - можно показать источники

Гибридный поиск объединяет семантическое сходство и ключевые слова."""
                
                rag.add_document(sample, source="пример.txt")
                st.success("✅ Пример загружен!")
                st.rerun()
        
        st.divider()
        
        # Статистика
        st.header("📊 Статистика")
        st.write(f"**Чанков:** {len(rag.chunks)}")
        st.write(f"**Бэкенд:** {rag.backend}")
        st.write(f"**API вызовов:** {rag.api_calls}")
        st.write(f"**Fallback:** {rag.fallback_used}")
    
    # Основной чат
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Отображение истории
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Ввод вопроса
    if prompt := st.chat_input("Задайте вопрос по документам..."):
        # Сохраняем вопрос
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Получаем ответ
        with st.chat_message("assistant"):
            with st.spinner("🔍 Ищу ответ..."):
                response = rag.query(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Информация о бэкенде
    if rag.backend == "fallback":
        st.info("ℹ️ Работаю в автономном режиме. Для лучших результатов добавьте API ключи в secrets!")

if __name__ == "__main__":
    main()
