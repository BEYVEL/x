"""
RAG с маленькой моделью Phi-3-mini для Streamlit Cloud
"""

import streamlit as st
import numpy as np
import os
import re
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
        width: 100%;
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
</style>
""", unsafe_allow_html=True)

class PhiRAG:
    """RAG с маленькой моделью Phi-3 для ограниченных ресурсов"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.articles = {}
        self.embeddings = {}
        
        # Используем маленькую модель
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        
        with st.spinner("🔄 Загрузка Phi-3-mini (3.8B параметров)..."):
            # Квантование для экономии памяти
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            st.sidebar.success("✅ Модель Phi-3 загружена")
        
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
            'задачи': 2.0
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
    
    def _generate_with_phi(self, context: str, question: str) -> str:
        """Генерация ответа через Phi-3-mini"""
        
        # Формируем промпт
        prompt = f"""<|system|>
Ты - эксперт по Национальной стратегии развития искусственного интеллекта РФ.
Отвечай на вопросы ТОЛЬКО на основе предоставленного контекста.
Формулируй ответы СВОИМИ СЛОВАМИ, понятно и структурированно.
Если информации нет в контексте, скажи "В документе нет информации об этом".</s>
<|user|>
Контекст:
{context}

Вопрос: {question}</s>
<|assistant|>"""
        
        # Токенизируем
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.model.device)
        
        # Генерируем ответ
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Декодируем ответ
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        return response.strip()
    
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
            
            # Индексация
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
        
        similarities = []
        for num, emb in self.embeddings.items():
            emb_array = np.array(emb)
            norm_product = np.linalg.norm(emb_array) * np.linalg.norm(query_emb)
            if norm_product > 0:
                similarity = np.dot(emb_array, query_emb) / norm_product
            else:
                similarity = 0
            similarities.append((num, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def query(self, question: str) -> str:
        """Ответ на вопрос"""
        if not self.articles:
            return "❌ Документ не загружен."
        
        # Поиск
        relevant = self.search(question, k=2)
        
        if not relevant:
            return "❌ В документе не найдена информация."
        
        # Собираем контекст
        context_parts = []
        articles_used = []
        
        for num, sim in relevant:
            if sim > 0.1:
                article_text = self.articles[num]
                if len(article_text) > 800:
                    article_text = article_text[:800] + "..."
                context_parts.append(f"[Статья {num}]:\n{article_text}")
                articles_used.append(num)
        
        context = "\n\n".join(context_parts)
        
        # Генерируем ответ
        answer = self._generate_with_phi(context, question)
        
        # Добавляем источники
        answer += f"\n\n<div class='source-box'>📚 Источники: Статьи {', '.join(articles_used)}</div>"
        
        return f'<div class="answer-box">{answer}</div>'

def main():
    st.title("📄 Национальная стратегия развития ИИ")
    
    with st.sidebar:
        st.header("❓ Примеры вопросов")
        
        examples = [
            "Какие федеральные законы составляют правовую основу стратегии?",
            "Что такое искусственный интеллект по определению стратегии?",
            "Что такое большие фундаментальные модели?",
            "Какие цели развития ИИ указаны в стратегии?"
        ]
        
        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.prompt = example
    
    # Путь к файлу
    file_path = "filerag.txt"
    
    if not os.path.exists(file_path):
        st.error(f"❌ Файл {file_path} не найден!")
        return
    
    # Инициализация
    if 'rag' not in st.session_state:
        st.session_state.rag = PhiRAG(file_path)
    
    # История
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
            with st.spinner("🤔 Анализирую..."):
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
    if prompt := st.chat_input("Задайте вопрос..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Анализирую..."):
                response = st.session_state.rag.query(prompt)
                st.markdown(response, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
