import pickle
import re
from typing import List
import numpy as np
import pandas as pd
import torch
# from clang import *
from clang import cindex
# from gensim.models import Word2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
# from gensim.utils import simple_preprocess
from pathlib import Path
from tokenizers import (NormalizedString, PreTokenizedString, Tokenizer,
                        normalizers, processors)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Sequential

from tokenizers.models import BPE
from tokenizers.normalizers import (Replace, StripAccents)
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from tokenizers.pre_tokenizers import PreTokenizer
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from typing import List

# custom modules
from classifier.models import ModelArchs
from classifier.utility import Utility
from classifier.preprocess import Preprocessor


class MyTokenizer:

    def __init__(self):
        self.cidx = cindex.Index.create()

    def clang_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """ Tokkenize using clang"""
        tok = []
        tu = self.cidx.parse('tmp.c',
                             args=[''],
                             unsaved_files=[
                                 ('tmp.c', str(normalized_string.original))],
                             options=0)
        for t in tu.get_tokens(extent=tu.cursor.extent):
            spelling = t.spelling.strip()

            if spelling == '':
                continue

            # enable the below line otherwise shows invalid token error in merges.txt file
            spelling = spelling.replace(' ', '')
            tok.append(NormalizedString(spelling))

        return (tok)

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.clang_split)

# def tokenize_using_clang(self, code: str):
#     """Tokenize code using clang"""
#     index = cindex.Index.create()
#     tu = index.parse('tmp.cpp', args=['-std=c++11'],
#                      unsaved_files=[('tmp.cpp', code)])
#     tokens = []
#     for token in tu.get_tokens(extent=tu.cursor.extent):
#         # tokens.append(token.spelling)
#         spelling = token.spelling.strip()
#         if spelling == '':
#             continue
#         spelling = spelling.replace(' ', '')
#         tokens.append(NormalizedString(spelling))
#         # tokens.append(spelling)
#     return tokens


class PretrainDataset():
    def __init__(self, custom_tokenizer, dataX, max_len):
        print('Tokenizing the data...')
        my_tokenizer = custom_tokenizer
        my_tokenizer.enable_truncation(max_length=max_len)
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []
        self.words = []

        def cleaner(code):
            # Remove code comments
            pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
            code = re.sub(pat, '', code)
            code = re.sub('\n', '', code)
            code = re.sub('\t', '', code)
            return (code)

        # mydata = pd.read_pickle('data/vulberta/pretrain/drapgh.pkl')
        dataX['code'] = dataX['code'].apply(cleaner)
        # mydata.functionSource = mydata.functionSource.apply(cleaner)
        dataX = dataX.sample(frac=1)
        dataX = dataX['code']
        # lines = src_file.read_text(encoding="utf-8").splitlines()
        self.examples += [
            x.ids for x in my_tokenizer.encode_batch(dataX.tolist())]
        # self.examples += [x for x in tokenizer.encode_batch(dataX.tolist())]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


class MyEmbeddings():
    def __init__(self, config):
        self.config = config
        self.emb_path = self.config['embedding']['path']
        self.w2v_file = self.emb_path + \
            self.config['embedding']['word2vec_file']
        self.tokenizer_path = self.config['embedding']['path']
        self.w2v_file = self.config['embedding']['word2vec_file']
        self.vocab_size = self.config['embedding']['vocab_size']
        self.max_len = self.config['embedding']['max_len']

        self.emb_files = self.config['embedding']['input_files']
        self.vocab_file = self.tokenizer_path + 'tokenizer.json'

    def init_tokenizer(self):
        """Init new tokenizer"""
        my_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        my_tokenizer = Tokenizer(BPE())

        my_tokenizer.normalizer = normalizers.Sequence(
            [StripAccents(), Replace(" ", "Ä")])
        # my_tokenizer.pre_tokenizer = PreTokenizer.custom(MyTokenizer())
        my_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        my_tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            special_tokens=[
                ("<s>", 0),
                ("<pad>", 1),
                ("</s>", 2),
                ("<unk>", 3),
                ("<mask>", 4)
            ]
        )
        return my_tokenizer

    def train_tokenizer(self, tokenizer, input_files):
        """Train tokenizer"""
        print('='*40)
        # check all input_files exist
        for file in self.emb_files:
            if not Path(file).exists():
                print(f'\nFile not found: {file}')
                exit(0)

        if Path(self.vocab_file).exists():
            print(f'\nTokenizer file already exists at: {self.vocab_file}')
            tokenizer = Tokenizer.from_file(self.vocab_file)
            print('Tokenizer loaded!')
        else:

            print("Start training tokenizer...")
            st = ['<s>', '<pad>', '</s>', '<unk>', '<mask>', 'char', 'int', 'switch', 'case', 'if', 'break', 'for', 'const', 'unsigned', 'struct', 'default', 'return', 'long', 'goto', 'this', 'enum', 'bool', 'static', 'false', 'true', 'new', 'delete', 'while', 'double', 'else', 'private', 'do', 'sizeof', 'void', 'continue', '__attribute__', 'short', 'throw', 'float', 'register', '__FUNCTION__', 'static_cast', '__func__', 'class', 'try', 'dynamic_cast', 'template', 'union', 'reinterpret_cast', 'catch', 'operator', 'const_cast', 'using', 'namespace', 'typename', 'wchar_t', 'not', 'typeof', '__label__', '__PRETTY_FUNCTION__', 'auto', '__extension__', 'volatile', '__asm__', '__volatile__', 'extern', 'asm', 'signed', 'typedef', 'typeid', 'and', 'or', 'public', 'virtual', 'nullptr', '__restrict', '__asm', '__typeof__', 'xor', '__complex__', '__real__', '__imag__', 'not_eq', 'export', 'compl', '__alignof__', '__restrict__', '__cdecl', 'bitor', 'protected', 'explicit', 'friend', 'decltype', 'mutable', 'inline', '__const', '__stdcall', 'char16_t', 'char32_t', '_Decimal64', 'constexpr', 'bitand', 'alignof', 'static_assert', '__attribute', 'thread_local', '__alignof', '__builtin_va_arg', '_Decimal32', '\"', '(', '*', ',', ')', '{', ';', '->', ':', '.', '-', '=', '+', '<', '++', '+=', '==', '||', '!=', '}', '/', '!', '>=', '[', ']', '&', '::', '&&', '>', '#', '--', '<=', '-=', '|', '%', '?', '<<', '>>', '|=', '&=', '^', '~', '^=', '...', '/=', '*=', '>>=', '<<=', '%=', '##', '->*', '\\', '.*', '@', '_Exit', 'abs', 'acos', 'acosh', 'asctime', 'asin', 'asinh', 'assert', 'at_quick_exit', 'atan', 'atan2', 'atanh', 'atexit', 'atof', 'atol', 'bsearch', 'btowc', 'c16rtomb', 'c32rtomb', 'cbrt', 'ceil', 'cerr', 'cin', 'clearerr', 'clock', 'clog', 'copysign', 'cos', 'cosh', 'cout', 'ctime', 'difftime', 'div', 'errno', 'exp', 'exp2', 'expm1', 'fabs', 'fclose', 'fdim', 'feclearexcept', 'fegetenv', 'fegetexceptflag', 'fegetround', 'feholdexcept', 'feof', 'feraiseexcept', 'ferror', 'fesetenv', 'fesetexceptflag', 'fesetround', 'fetestexcept', 'feupdateenv', 'fflush', 'fgetc', 'fgetpos', 'fgets', 'fgetwc', 'fgetws', 'floor', 'fma', 'fmax', 'fmod', 'fopen', 'fprintf', 'fputc', 'fputs', 'fputwc', 'fputws', 'fread', 'free', 'freopen',
                  'frexp', 'fscanf', 'fseek', 'fsetpos', 'ftell', 'fwide', 'fwprintf', 'fwrite', 'fwscanf', 'getc', 'getchar', 'getenv', 'gets', 'getwc', 'getwchar', 'gmtime', 'hypot', 'ilogb', 'imaxabs', 'imaxdiv', 'isblank', 'iscntrl', 'isdigit', 'isgraph', 'islower', 'isprint', 'ispunct', 'isspace', 'isupper', 'iswalnum', 'iswalpha', 'iswblank', 'iswcntrl', 'iswctype', 'iswdigit', 'iswgraph', 'iswlower', 'iswprint', 'iswpunct', 'iswspace', 'iswupper', 'iswxdigit', 'isxdigit', 'labs', 'ldexp', 'ldiv', 'llabs', 'lldiv', 'llrint', 'llround', 'localeconv', 'localtime', 'log', 'log10', 'log1p', 'log2', 'logb', 'longjmp', 'lrint', 'lround', 'malloc', 'mblen', 'mbrlen', 'mbrtoc16', 'mbrtoc32', 'mbrtowc', 'mbsinit', 'mbsrtowcs', 'mbstowcs', 'mbtowc', 'memchr', 'memcmp', 'memcpy', 'memmove', 'memset', 'mktime', 'modf', 'nan', 'nearbyint', 'nextafter', 'nexttoward', 'perror', 'pow', 'printf', 'putc', 'putchar', 'puts', 'putwchar', 'qsort', 'quick_exit', 'raise', 'realloc', 'remainder', 'remove', 'remquo', 'rename', 'rewind', 'rint', 'round', 'sca', 'scalbln', 'scalbn', 'setbuf', 'setjmp', 'setlocale', 'setvbuf', 'signal', 'sin', 'sinh', 'snprintf', 'sprintf', 'sqrt', 'srand', 'sscanf', 'strcat', 'strchr', 'strcmp', 'strcoll', 'strcpy', 'strcspn', 'strerror', 'strftime', 'strlen', 'strncat', 'strncmp', 'strncpy', 'strpbrk', 'strrchr', 'strspn', 'strstr', 'strtod', 'strtoimax', 'strtok', 'strtol', 'strtoll', 'strtoull', 'strtoumax', 'strxfrm', 'swprintf', 'swscanf', 'tan', 'tanh', 'time', 'tmpfile', 'tmpnam', 'tolower', 'toupper', 'towctrans', 'towlower', 'towupper', 'trunc', 'ungetc', 'ungetwc', 'vfprintf', 'vfscanf', 'vfwprintf', 'vfwscanf', 'vprintf', 'vscanf', 'vsfscanf', 'vsnprintf', 'vsprintf', 'vsscanf', 'vswprintf', 'vwprintf', 'vwscanf', 'wcerr', 'wcin', 'wclog', 'wcout', 'wcrtomb', 'wcscat', 'wcschr', 'wcscmp', 'wcscoll', 'wcscpy', 'wcscspn', 'wcsftime', 'wcslne', 'wcsncat', 'wcsncmp', 'wcsncpy', 'wcspbrk', 'wcsrchr', 'wcsrtombs', 'wcsspn', 'wcsstr', 'wcstod', 'wcstof', 'wcstoimax', 'wcstok', 'wcstol', 'wcstold', 'wcstoll', 'wcstombs', 'wcstoul', 'wcstoull', 'wcstoumax', 'wcsxfrm', 'wctob', 'wctomb', 'wctrans', 'wctype', 'wmemchr', 'wmemcmp', 'wmemcpy', 'wmemmove', 'wmemset', 'wprintf', 'wscanf']
            trainer = BpeTrainer(vocab_size=50000, min_frequency=2,
                                 show_progress=True, special_tokens=st)
            tokenizer.train(input_files, trainer)
            tokenizer.save(self.vocab_file)
            # use this to get vocab.json and merges.txt both
            # tokenizer.model.save(tokenizer_path)
            # tokenizer.model.save_pretrained(tokenizer_path)
            # trainer.save_pretrained(trainer)
            print(f'Trained tokenizer is saved at {self.tokenizer_path}')
        tokenizer.pre_tokenizer = PreTokenizer.custom(MyTokenizer())
        print('='*40)
        return tokenizer

    def apply_padding(self, tokenizer):
        """function to apply padding"""
        print('Applying padding to the tokenizer...')
        tokenizer.enable_truncation(max_length=self.max_len)
        tokenizer.enable_padding(
            direction='right',
            pad_id=1,
            pad_type_id=0,
            pad_token='<pad>',
            length=None,
            pad_to_multiple_of=None
        )
        return tokenizer

    def get_word_index_tokenizer(self, tokenizer):
        """function to get word_index"""
        word_index = {}
        for i, token in enumerate(tokenizer.get_vocab()):
            word_index[token] = i
        return word_index

    def get_word_index_w2v(self, w2v_model):
        """function to get word index from w2v model"""
        word_index = {}
        for i, token in enumerate(w2v_model.wv.key_to_index):
            word_index[token] = i
        return word_index

    def train_word2vec(self, w2v_file, examples):
        print('Training a new Word2Vec model...')
        sentences = [x for x in examples]
        w2v_model = Word2Vec(sentences, window=5, min_count=5)
        w2v_model.wv.save(self.w2v_file)
        print(f"The Word2Vec model is saved at: {self.w2v_file}")
        return w2v_model

    def load_word2vec_model(self, w2v_file):
        """ Load the model from a pickle file"""
        w2v_model = None
        print('='*40)
        print(f"Loading Word2Vec model from: {w2v_file}")
        w2v_model = Word2Vec()
        w2v_model.wv = KeyedVectors.load(w2v_file)
        print("Word2Vec model loaded from the already trained" +
              f"file [{w2v_file}] successfully.")
        print('='*40)
        return w2v_model

    def load_w2v_emb(self, w2v_model):
        """function to make embeddings_index from w2v model file"""
        embeddings_index = dict()
        # with open(w2v_file, 'r') as w2v:
        #     for line in w2v:
        #         values = line.split()
        #         word = values[0]
        #         coefs = np.asarray(values[1:], dtype='float32')
        #         embeddings_index[word] = coefs
        embeddings_index = {word: w2v_model.wv[word]
                            for word in w2v_model.wv.key_to_index}
        return embeddings_index

    def get_embedding_matrix(self, word_index, embeddings_index, num_words,
                             emb_dim):
        """function to generate embedding_matrix from embedding_index"""
        emb_matrix_file = self.emb_path + 'embedding_matrix.npy'
        if Path(emb_matrix_file).exists():
            print(f"Embedding matrix already exists at: {emb_matrix_file}")
            embedding_matrix = np.load(emb_matrix_file)
            print('Embedding matrix loaded!')
        else:
            print('Generating embedding matrix...')
            # Initialize matrix with zeros
            embedding_matrix = np.zeros((num_words, emb_dim))
            for word, i in word_index.items():
                if i >= num_words:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is None:
                    print(f"Not found embedding for word: {word}")
                    continue
                embedding_matrix[i] = embedding_vector
            print(f'num_words: {num_words}')
            print(f'emb_dim: {emb_dim}')
            print(f'Shape of embedding_matrix: {embedding_matrix.shape}')
            print('Embedding matrix generated!')
            # save embedding_matrix
            np.save(emb_matrix_file, embedding_matrix)
        return embedding_matrix

    def vectorize_and_load_fun_data(self, df):
        """convert the functions into vectorized form"""
        prepro = Preprocessor(self.config)
        vectorized_data = self.config['vectorized_data']

        emb = MyEmbeddings(self.config)
        my_tokenizer = emb.init_tokenizer()

        my_tokenizer = emb.train_tokenizer(
            tokenizer=my_tokenizer,
            input_files=self.config['embedding']['input_files'],
            # tokenizer_path=emb_path,
        )
        # load already processed andvectorized data to skip preprocessing
        if Path(vectorized_data).exists() and Path(self.w2v_file).exists():
            X, y = prepro.load_vectorized_data(vectorized_data)
            w2v_model = emb.load_word2vec_model(self.w2v_file)
        else:
            # data = pd.read_csv(self.config['data_file'], nrows=1000)
            dataset = PretrainDataset(my_tokenizer, df, self.max_len)
            X = pad_sequences(dataset.examples)
            y = np.array(df['label'])
            print(f"Shape of X: {X.shape}, Shape of y:{y.shape}")
            prepro.save_vectorized_data(X, y, vectorized_data)

            # train w2v_model
            w2v_model = emb.train_word2vec(
                w2v_file=self.w2v_file, examples=dataset.examples)

        my_tokenizer = emb.apply_padding(tokenizer=my_tokenizer)

        word_index = emb.get_word_index_tokenizer(my_tokenizer)
        # word_index = emb.get_word_index_w2v(w2v_model=w2v_model)
        vocab_size = len(word_index) + 1
        embeddings_index = emb.load_w2v_emb(w2v_model=w2v_model)

        embedding_matrix = emb.get_embedding_matrix(
            word_index=word_index,
            embeddings_index=embeddings_index,
            num_words=len(word_index) + 1,
            emb_dim=self.max_len
        )
        return X, y, vocab_size, embedding_matrix
