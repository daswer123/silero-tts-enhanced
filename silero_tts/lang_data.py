# lang_data.py

from silero_tts.transliterate import transliterate,reverse_transliterate

lang_data = {
    'ru': {
        'script': 'cyrillic',
        'replacements': [
            ("…", "..."),  # Model does not handle "…"
            ("*", " звёздочка "),
            ("%", " процентов "),
            (" г.", " году"),
            (" гг.", " годах"),
        ],
        'patterns': [
            (r'(\d+)[\.|,](\d+)', r'\1 и \2'),  # to make more clear stuff like 2.75%
            ("д.\s*н.\s*э.", " до нашей эры"),
            ("н.\s*э.", " нашей эры"),
        ],
    },
    'en': {
        'script': 'latin',
        'replacements': [
            ("…", "..."),  # Model does not handle "…"
            ("*", " asterisk "),
            ("%", " percent "),
        ],
        'patterns': [
            (r'(\d+)[\.|,](\d+)', r'\1 point \2'),  # to make more clear stuff like 2.75%
            ("B\.C\.", " before Christ"),
            ("A\.D\.", " anno Domini"),
        ],
    },
    'de': {
        'script': 'latin',
        'replacements': [
            ("…", "..."),  # Model does not handle "…"
            ("*", " Sternchen "),
            ("%", " Prozent "),
        ],
        'patterns': [
            (r'(\d+)[\.|,](\d+)', r'\1 Komma \2'),  # to make more clear stuff like 2,75%
            ("v\. Chr\.", " vor Christus"),
            ("n\. Chr\.", " nach Christus"),
        ],
    },
    'es': {
        'script': 'latin',
        'replacements': [
            ("…", "..."),  # Model does not handle "…"
            ("*", " asterisco "),
            ("%", " por ciento "),
        ],
        'patterns': [
            (r'(\d+)[\.|,](\d+)', r'\1 punto \2'),  # to make more clear stuff like 2.75%
            ("a\. C\.", " antes de Cristo"),
            ("d\. C\.", " después de Cristo"),
        ],
    },
    'fr': {
        'script': 'latin',
        'replacements': [
            ("…", "..."),  # Model does not handle "…"
            ("*", " astérisque "),
            ("%", " pour cent "),
        ],
        'patterns': [
            (r'(\d+)[\.|,](\d+)', r'\1 virgule \2'),  # to make more clear stuff like 2,75%
            ("av\. J\.-C\.", " avant Jésus-Christ"),
            ("ap\. J\.-C\.", " après Jésus-Christ"),
        ],
    },
    'ba': {
        'script': 'cyrillic',
        'replacements': [
            ("…", "..."),  # Model does not handle "…"
            ("*", " йондоҙ "),
            ("%", " процент "),
        ],
        'patterns': [
            (r'(\d+)[\.|,](\d+)', r'\1 нөктә \2'),  # to make more clear stuff like 2.75%
        ],
    },
    'xal': {
        'script': 'cyrillic',
        'replacements': [
            ("…", "..."),  # Model does not handle "…"
            ("*", " одн "),
            ("%", " процент "),
        ],
        'patterns': [
            (r'(\d+)[\.|,](\d+)', r'\1 цег \2'),  # to make more clear stuff like 2.75%
        ],
    },
    'tt': {
        'script': 'cyrillic',
        'replacements': [
            ("…", "..."),  # Model does not handle "…"
            ("*", " йолдызча "),
            ("%", " процент "),
        ],
        'patterns': [
            (r'(\d+)[\.|,](\d+)', r'\1 нокта \2'),  # to make more clear stuff like 2.75%
        ],
    },
    'uz': {
        'script': 'latin',
        'replacements': [
            ("…", "..."),  # Model does not handle "…"
            ("*", " yulduzcha "),
            ("%", " foiz "),
        ],
        'patterns': [
            (r'(\d+)[\.|,](\d+)', r'\1 nuqta \2'),  # to make more clear stuff like 2.75%
        ],
    },
    'ua': {
        'script': 'cyrillic',
        'replacements': [
            ("…", "..."),  # Model does not handle "…"
            ("*", " зірочка "),
            ("%", " відсотків "),
        ],
        'patterns': [
            (r'(\d+)[\.|,](\d+)', r'\1 кома \2'),  # to make more clear stuff like 2.75%
            ("до н\. е\.", " до нашої ери"),
            ("н\. е\.", " нашої ери"),
        ],
    },
    'indic': {
        'script': 'latin',
        'replacements': [
            ("…", "..."),  # Model does not handle "…"
            ("*", " nakshatr "),
            ("%", " pratishat "),
        ],
        'patterns': [
            (r'(\d+)[\.|,](\d+)', r'\1 dashamlav \2'),  # to make more clear stuff like 2.75%
        ],
    },
    'cyrillic': {
        'script': 'cyrillic',
        'replacements': [
            ("…", "..."),  # Model does not handle "…"
            ("*", " звездочка "),
            ("%", " процентов "),
        ],
        'patterns': [
            (r'(\d+)[\.|,](\d+)', r'\1 запятая \2'),  # to make more clear stuff like 2.75%
        ],
    },
}

def is_cyrillic(text):
    return any('а' <= char.lower() <= 'я' for char in text)

def is_latin(text):
    return all(char.isascii() for char in text)


def to_cyrillic(text, language):
    if lang_data[language]['script'] == 'cyrillic':
        return text
    return reverse_transliterate(text, language)

def to_latin(text, language):
    if lang_data[language]['script'] == 'latin':
        return text
    return transliterate(text, language)
