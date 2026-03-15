"""
ПРОФЕССИОНАЛЬНЫЙ RAG с локальной Mistral-7B
ОТВЕТЫ ФОРМУЛИРУЮТСЯ, А НЕ КОПИРУЮТСЯ
Работает полностью локально, без API
"""

import streamlit as st
import numpy as np
import os
import re
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Проверка доступности GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"🖥️ Используется: {device.upper()}")

class MistralRAG:
    """RAG с локальной Mistral-7B для настоящего понимания"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.articles = {}
        self.embeddings = {}
        
        # Загружаем модель Mistral
        with st.spinner("🔄 Загрузка модели Mistral-7B (займёт 1-2 минуты)..."):
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            self.model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto"
            )
            st.sidebar.success("✅ Модель Mistral загружена")
        
        # Загружаем документ
        self._load_document()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Локальные эмбеддинги для поиска"""
        words = text.lower().split()
        
        # Веса для ключевых терминов
        important_terms = {
            'искусственный интеллект': 3.0,
            'федеральный закон': 3.0,
            'конституция': 3.0,
            'правовую основу': 3.0,
            'статья 2': 4.0,
            'статья 5': 3.5,
            'цели': 2.0,
            'задачи': 2.0,
            'развитие': 1.5
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
    
    def _generate_with_mistral(self, context: str, question: str) -> str:
        """
        Генерация ответа через локальную Mistral-7B
        Модель понимает контекст и формулирует ответ
        """
        
        # Формируем промпт в формате Mistral-Instruct
        messages = [
            {
                "role": "user", 
                "content": f"""Ты - эксперт по Национальной стратегии развития искусственного интеллекта РФ.

На основе фрагментов документа ниже, ответь на вопрос.
ОТВЕЧАЙ СВОИМИ СЛОВАМИ, НЕ КОПИРУЙ ТЕКСТ ИЗ ДОКУМЕНТА.
Если информации нет в документе, скажи "В документе нет информации об этом".

ФРАГМЕНТЫ ДОКУМЕНТА:
{context}

ВОПРОС: {question}

ОТВЕТ (подробно, своими словами, со ссылкой на документ):"""
            }
        ]
        
        # Применяем шаблон для Mistral
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Генерируем ответ
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.3,  # Низкая температура для точности
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Декодируем ответ
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
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
        
        # Сортируем по убыванию
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def query(self, question: str) -> str:
        """Полный RAG цикл с генерацией через Mistral"""
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
        
        # Генерируем ответ через Mistral
        answer = self._generate_with_mistral(context, question)
        
        # Добавляем источники
        answer += f"\n\n<div class='source-box'>📚 Источники: Статьи {', '.join(articles_used)}</div>"
        
        return f'<div class="answer-box">{answer}</div>'

def main():
    st.title("📄 Национальная стратегия развития ИИ")
    st.markdown("*Анализ документа с локальной Mistral-7B*")
    
    # Сайдбар с примерами вопросов
    with st.sidebar:
        st.header("❓ Примеры вопросов")
        
        examples = [
            "Какие федеральные законы составляют правовую основу стратегии?",
            "Что такое искусственный интеллект по определению стратегии?",
            "Какие цели развития искусственного интеллекта указаны в стратегии?",
            "Что такое большие фундаментальные модели и какой порог параметров указан?",
            "Какие основные принципы развития ИИ?",
            "Что говорится в статье 25?",
            "Какие вызовы стоят перед Россией в сфере ИИ?"
        ]
        
        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.prompt = example
        
        st.markdown("---")
        st.markdown(f"**Статус:** Модель Mistral-7B загружена")
    
    # Путь к файлу
    file_path = "filerag.txt"
    
    if not os.path.exists(file_path):
        st.error(f"❌ Файл {file_path} не найден!")
        return
    
    # Инициализация RAG
    if 'rag' not in st.session_state:
        st.session_state.rag = MistralRAG(file_path)
    
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
            with st.spinner("🤔 Анализирую документ..."):
                response = st.session_state.rag.query(prompt)
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
            with st.spinner("🤔 Анализирую документ..."):
                response = st.session_state.rag.query(prompt)
                st.markdown(response, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
