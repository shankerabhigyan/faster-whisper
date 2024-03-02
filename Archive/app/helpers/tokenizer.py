from typing import Any


def create_tokenizer(lan: str) -> Any:
    
    if lan == "uk":
        import tokenize_uk
        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)
        return UkrainianTokenizer()
    from mosestokenizer import MosesTokenizer
    return MosesTokenizer(lan)