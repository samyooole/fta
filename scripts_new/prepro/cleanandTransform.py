from sentence_transformers import SentenceTransformer
from functools import wraps
import logging
import re
# Standard libraries
import os
import re
import string
import logging
import csv
from pathlib import Path
from functools import wraps
from unicodedata import normalize
from typing import List, Optional, Union, Callable

# Third party libraries
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
from spellchecker import SpellChecker
from names_dataset import NameDataset
from tqdm import tqdm
import time

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class cleanandTransform():

    """
    cleanandTransform is an object that 
    
    1. fits onto a list of strings, 
    2. processes it based on the preset filters that you wish to apply (eg. lowercase, lemmatize), then 
    3. encodes it as a batch into a set of sentence embeddings (tends to take a long time)

    Some code is copied from berknology/text-preprocessing
    """


    def __init__(self, filters, transformer_package = 'all-mpnet-base-v2'):
        
        """
        initialize a //sentence// transformer
        """
        print("initializing transformer...")
        self.transformer = SentenceTransformer(transformer_package)


        """
        initialize the filters that operate on a STRING level. strictly receives a list
        """

        self.filters = filters
        self.origtextlist = []
        self.current_text = []
        self.current_embedding = []



    def _return_empty_string_for_invalid_input(self,func):
        """ Return empty string if the input is None or empty """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'input_text' in kwargs:
                input_text = kwargs['input_text']
            else:
                try:
                    input_text = args[0]
                except IndexError as e:
                    LOGGER.exception('No appropriate positional argument is provide.')
                    raise e
            if input_text is None or len(input_text) == 0:
                return ''
            else:
                return func(*args, **kwargs)
        return wrapper


    
    def to_lower(self, input_text: str) -> str:
        """
        Convert input text to lower case
        """
        return input_text.lower()

    
    def remove_number(self, input_text: str) -> str:
        """ Remove number in the input text """
        processed_text = re.sub('\d+', '', input_text)
        return processed_text


    
    def remove_itemized_bullet_and_numbering(self, input_text: str) -> str:
        """ Remove bullets or numbering in itemized input """
        processed_text = re.sub('[(\s][0-9a-zA-Z][.)]\s+|[(\s][ivxIVX]+[.)]\s+', ' ', input_text)
        return processed_text

    
    def expand_contraction(self, input_text: str) -> str:
        """ Expand contractions in input text """
        return contractions.fix(input_text)

    
    def remove_itemized_bullet_and_numbering(self, input_text: str) -> str:
        """ Remove bullets or numbering in itemized input """
        processed_text = re.sub('[(\s][0-9a-zA-Z][.)]\s+|[(\s][ivxIVX]+[.)]\s+', ' ', input_text)
        return processed_text


    
    def remove_url(self, input_text: str) -> str:
        """ Remove url in the input text """
        return re.sub('(www|http)\S+', '', input_text)


    
    def remove_punctuation(self, input_text: str, punctuations: Optional[str] = None) -> str:
        """
        Removes all punctuations from a string, as defined by string.punctuation or a custom list.
        For reference, Python's string.punctuation is equivalent to '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~'
        """
        if punctuations is None:
            punctuations = string.punctuation
        processed_text = input_text.translate(str.maketrans('', '', punctuations))
        return processed_text


    
    def remove_special_character(self, input_text: str, special_characters: Optional[str] = None) -> str:
        """ Removes special characters """
        if special_characters is None:
            # TODO: add more special characters
            special_characters = 'å¼«¥ª°©ð±§µæ¹¢³¿®ä£'
        processed_text = input_text.translate(str.maketrans('', '', special_characters))
        return processed_text


    
    def keep_alpha_numeric(self, input_text: str) -> str:
        """ Remove any character except alphanumeric characters """
        return ''.join(c for c in input_text if c.isalnum())


    
    def remove_whitespace(self, input_text: str, remove_duplicate_whitespace: bool = True) -> str:
        """ Removes leading, trailing, and (optionally) duplicated whitespace """
        if remove_duplicate_whitespace:
            return ' '.join(re.split('\s+', input_text.strip(), flags=re.UNICODE))
        return input_text.strip()

    
    def normalize_unicode(self, input_text: str) -> str:
        """ Normalize unicode data to remove umlauts, and accents, etc. """
        processed_tokens = normalize('NFKD', input_text).encode('ASCII', 'ignore').decode('utf8')
        return processed_tokens


    
    def remove_stopword(self, input_text_or_list: Union[str, List[str]], stop_words: Optional[set] = None) -> List[str]:
        """ Remove stop words """

        if stop_words is None:
            stop_words = set(stopwords.words('english'))
        if isinstance(stop_words, list):
            stop_words = set(stop_words)
        if isinstance(input_text_or_list, str):
            tokens = word_tokenize(input_text_or_list)
            processed_tokens = [token for token in tokens if token not in stop_words]
        else:
            processed_tokens = [token for token in input_text_or_list
                                if (token not in stop_words and token is not None and len(token) > 0)]
        return " ".join(processed_tokens)

    
    
    def stem_word(self, input_text_or_list: Union[str, List[str]],
                stemmer: Optional[Union[PorterStemmer, SnowballStemmer, LancasterStemmer]] = None
                ) -> List[str]:
        """ Stem each token in a text """
        if stemmer is None:
            stemmer = PorterStemmer()
        if isinstance(input_text_or_list, str):
            tokens = word_tokenize(input_text_or_list)
            processed_tokens = [stemmer.stem(token) for token in tokens]
        else:
            processed_tokens = [stemmer.stem(token) for token in input_text_or_list if token is not None and len(token) > 0]
        return " ".join(processed_tokens)


    
    def lemmatize_word(self, input_text_or_list: Union[str, List[str]],
                    lemmatizer: Optional[WordNetLemmatizer] = None
                    ) -> List[str]:
        """ Lemmatize each token in a text by finding its base form """
        if lemmatizer is None:
            lemmatizer = WordNetLemmatizer()
        if isinstance(input_text_or_list, str):
            tokens = word_tokenize(input_text_or_list)
            processed_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        else:
            processed_tokens = [lemmatizer.lemmatize(token)
                                for token in input_text_or_list if token is not None and len(token) > 0]
        return " ".join(processed_tokens)

    """
    start of fitting operators
    """

    def init_text(self, textlist: List[str]):
        """
        - sets original textlist to the input text list
        - erases processed text and current embeddings
        """
        
        self.origtextlist = list(textlist)
        self.current_text = self.origtextlist
        self.current_embedding = []

    def set_filters(self, filters: List[str]):
        self.filters = filters

    def process_text(self):
        filters = self.filters
        for filter in tqdm(filters):
            func = eval("self."+filter)
            
            for idx, text in enumerate( self.current_text ):
                output_str = func(text)
                self.current_text[idx] = output_str

    def transform_text(self):
        text_to_encode = self.current_text

        model = self.transformer

        new_embeddings = model.encode(text_to_encode, show_progress_bar=True)

        self.current_embedding = new_embeddings


    
