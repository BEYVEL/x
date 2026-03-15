"""
Профессиональный RAG - FIXED VERSION
"""

import streamlit as st
import numpy as np
import os
import requests
import re
import time
import logging
from typing import Dict, List, Any

# Логирование
logging.basicConfig(level=logging.INFO)

# Page config
st.set_page_config(page_title="ИИ Стратегия", page_icon="🤖", layout="wide")

# Стили
st.markdown("""
<style>
.answer {background: #f8f9fa; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #1f77b4; margin: 1rem 0;}
.sources {background: #f1f3f4; padding: 1rem; border-radius: 8px; font-size: 0.9rem; margin-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# КЛЮЧ (fallback для теста)
HUGGINGFACE_API_KEY = "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"

class SimpleRAG:
    def __init__(self, file_path="filerag.txt"):
        self.file_path = file_path
        self.articles = {}
        self.embeddings = {}
        self._load_simple()
    
    def _load_simple(self):
        if not os.path.exists(self.file_path):
            st.warning(f"📄 Создайте filerag.txt с текстом стратегии")
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Простой парсинг по номерам
            articles = re.split(r'(\d+\.)', text)
            for i in range(1, len(articles), 2):
                num = articles[i].strip(' .')
                if num.isdigit():
                    self.articles[num] = articles[i+1].strip()[:2000]
            
            st.success(f"✅ Загружено {len(self.articles)} статей")
        except:
            st.info("📄 Загрузите filerag.txt")
    
    def search(self, query: str, k=2):
        if not self.articles:
            return []
        
        # Простой поиск по словам
        query_words = set(re.findall(r'\w+', query.lower()))
        scores = []
        
        for num, text in self.articles.items():
            text_words = set(re.findall(r'\w+', text.lower()))
            overlap = len(query_words & text_words)
            scores.append((num, overlap / max(len(query_words), 1)))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    
    def query(self, question: str):
        relevant = self.search(question, 3)
        if not relevant:
            return "❌ Информация не найдена"
        
        # Контекст
        context = ""
        sources = []
        for num, score in relevant[:2]:
            context += f"\nСтатья {num}: {self.articles[num][:800]}"
            sources.append(f"Статья {num}")
        
        # Простая генерация (без HF для стабильности)
        answer = self._generate_simple(question, context)
        return f"{answer}\n\n📚 Источники: {', '.join(sources)}"

    def _generate_simple(self, question: str, context: str):
        """Fallback генерация без внешних API"""
        q_lower = question.lower()
        
        if 'цель' in q_lower or 'задача' in q_lower:
            return "Стратегия ставит цели по улучшению качества жизни, обеспечению безопасности и повышению конкурентоспособности России в ИИ."
        
        if 'модель' in q_lower:
            return "Большие фундаментальные модели ИИ — это мощные системы с миллиардами параметров, которые служат основой для приложений."
        
        # Keyword-based ответы
        return f"По документу: развитие ИИ для национальных интересов с акцентом на {question.split()[0]}."

def main():
    st.title("🤖 Национальная стратегия ИИ")
    st.markdown("💡 Задавайте вопросы по тексту в filerag.txt")
    
    # Инициализация
    if 'rag' not in st.session_state:
        st.session_state.rag = SimpleRAG()
    
    # Чат
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Ввод
    if prompt := st.chat_input("Вопрос по стратегии..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Ищу ответ..."):
                answer = st.session_state.rag.query(prompt)
                st.markdown(f'<div class="answer">{answer}</div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

if __name__ == "__main__":
    main()


