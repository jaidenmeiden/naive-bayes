#!/usr/bin/env python
# coding: utf-8

# In[9]:


import math
import os
from pathlib import Path


# In[10]:


get_ipython().system('jupyter nbextension install --py jupyter_tabnine')
get_ipython().system('jupyter nbextension enable --py jupyter_tabnine')
get_ipython().system('jupyter serverextension enable --py jupyter_tabnine')


# ## Preparación del corpus de emails

# In[12]:


if not os.path.isdir('datasets'):
    print ("Directory not exist")
    get_ipython().system('git clone https://github.com/pachocamacho1990/datasets')
else:
    print ("Directory exist")


# In[13]:


if not os.path.isdir('corpus1/spam'):
    print ("Directory not exist")
    get_ipython().system(' unzip datasets/email/plaintext/corpus1.zip')
else:
    print ("Directory exist")


# In[14]:


os.listdir('corpus1/spam')


# In[15]:


data = []
clases = []
#lectura de spam data
for file in os.listdir('corpus1/spam'):
    with open('corpus1/spam/'+file, encoding='latin-1') as f:
        data.append(f.read())
        clases.append('spam')
#lectura de ham data
for file in os.listdir('corpus1/ham'):
    with open('corpus1/ham/'+file, encoding='latin-1') as f:
        data.append(f.read())
        clases.append('ham')
len(data)


# ## Construcción de modelo Naive Bayes

# ### Tokenizador de Spacy
# 
# * Documentación: https://spacy.io/api/tokenizer
# * ¿Cómo funciona el tokenizador? https://spacy.io/usage/linguistic-features#how-tokenizer-works

# In[16]:


from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

nlp = English()
tokenizer = Tokenizer(nlp.vocab)


# Aplicamos el tokenizador para nuestro texto obteneindo una lista

# In[17]:


print([t.text for t in tokenizer(data[0])])


# ### Clase principal para el algoritmo
# 
# Recuerda que la clase más probable viene dada por (en espacio de cómputo logarítmico): 
# 
# 
# $$\hat{c} = {\arg \max}_{(c)}\log{P(c)}
#  +\sum_{i=1}^n
# \log{ P(f_i \vert c)}
# $$
# 
# Donde, para evitar casos atípicos (Hay escenarios donde la probabilidad es cero, ya que una palabra especifica **NO** existe para una **categoría dada** en el corpus, lo cual NO significa que la palabra NO este en el corpus, solo que no tiene relación con la categoría especificada), usaremos el suavizado de Laplace así:
# 
# $$
# P(f_i \vert c) = \frac{C(f_i, c)+1}{C(c) + \vert V \vert}
# $$
# 
# siendo $\vert V \vert$ la longitud del vocabulario de nuestro conjunto de entrenamiento. 

# In[20]:


import numpy as np

# Ceamos una clase de Naive Bayes para clasificar
class NaiveBayesClassifier():
    # Instancias propias de la clase
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab) 

    # Función que corresponde al tokenizador
    # Renorna el texto completamente tokenizado en una lista
    def tokenize(self, doc):
        return [t.text.lower() for t in tokenizer(doc)]

    # Contador de palabras, ya que para calcular
    # las probabilidades necesito contar las palabras
    def word_counts(self, words):
        wordCount = {} # Diccionario
        
        # Recorro todas las palabras
        for w in words: 
            # Verifica si la palabra esta en el diccionario
            if w in wordCount.keys():
                wordCount[w] += 1
            else:
                wordCount[w] = 1
        return wordCount

    # Metodo 'fit' donde se entrena el modelo y 
    # debe adminir como entrada los datos de entrenamiento
    def fit(self, data, clases):
        # Define la longitud de los datos de entrenamiento
        n = len(data)
        
        # Atributo que define cuantas clases unicas hay
        # y debe ser capaz de hacer clasificación multiclase
        self.unique_clases = set(clases)
        
        # Atributo vacio para asignación de vocabulario
        self.vocab = set()
        
        # Atributo para contar cuantas veces aparece cada 
        # categoría posible en el corpus 'C(c)'.
        self.classCount = {} #C(c)
        
        # Atributo para calcular el logaritmo de la probabilidad de la categoría 'P(c)'.
        self.log_classPriorProb = {} #P(c)
        
        # Atributo para calcular los contenos de que dada una categoría se observe una 
        # palabra en algún documento de dicha categoría 'C(w|c)'.
        self.wordConditionalCounts = {} #C(w|c)
    
        # Conteos de categorías
        for c in clases:
            if c in self.classCount.keys():
                self.classCount[c] += 1
            else:
                self.classCount[c] = 1
        # Calculo de P(c)
        for c in self.classCount.keys():
            # Por cada una de las categorías en el diccionario calculo 
            # el logaritmo de las probabilidades 'prior'.
            self.log_classPriorProb[c] = math.log(self.classCount[c]/n)
            
            # Diccionario vacio para el conteo de que dada una categoría 
            # se observe una palabra en algún documento.
            self.wordConditionalCounts[c] = {}
            
        # Calculo de los conteos condicionales de cada una de 
        # las palabras ligadas a una categoría C(w|c)
        for text, c in zip(data, clases):# Recorremos dos listas simultaneamente
            # Aplicamos la función conteo que devuelve el número de veces que 
            # aparece el objeto texto en los datos.
            counts = self.word_counts(self.tokenize(text))
            for word, count in counts.items():
                # Vericamos si la palabra no esta en el vocabulario, si no esta, la agregamos
                if word not in self.vocab:
                    self.vocab.add(word)
                # Vericamos si la palabra no esta en los conteos condicionales, si no esta, la agregamos    
                if word not in self.wordConditionalCounts[c]:
                    self.wordConditionalCounts[c][word] = 0.0
                # Hacemos el conteo condicional respectivo    
                self.wordConditionalCounts[c][word] += count

    # Metodo 'predict' donde se hacen prediciones con el modelo entrenado y 
    # debe adminir los datos sobre los cuales debe predecir.
    def predict(self, data):
        # Contruyo la lista para los resultados
        results = []
        
        # Recorro todas las palabras de la data (Conjunto de datos para predecir)
        for text in data:
            # Atributo que obtiene un conjunto de palabras unicas
            words = set(self.tokenize(text))
            
            # Defino un socre de probabilidad, el cual es un diccionario donde cada categoría 
            # tiene su propia probabilidad y lo que se hace es escoger el maximo de dichas 
            # probabilidades.
            scoreProb = {}
            
            # Se recorre cada palabra de todas las palabras posibles en el conjunto tokenizado
            for word in words: 
                # Si la palabra no esta en el vocabulario se ignora, 
                # ya que no hay forma de calcular la probabilidad
                if word not in self.vocab: continue #ignoramos palabras nuevas

                # Si la palabra si esta en el vocabulario se aplica el     
                # suavizado Laplaciano para P(w|c)
                # Comienzo a recorrer todas las categorías de mi algoritmo, las cuales obtengo el metodo
                # 'self.unique_clases' definido en la función 'fit'
                for c in self.unique_clases:
                    # Se calcula el logaritmo de la probabilidad de que dada una clase tenga una 
                    # palabra (Probabilidad condicional). Dividimos dos elementos que corresponden 
                    # al suavizado de Laplace:
                    # -------------------------------------------------------------------------------------
                    # Primer elemento: Los conteos de las palabras dadas las categorías más uno => 𝐶(𝑓𝑖,𝑐)+1
                    # Para este primer elemento buscamos si la palabra 'word' existe para la categoría 'c',
                    # si no exiete devuelve '0'. Se usa la función '.get(word, 0.0)' en el dicionario.
                    # -------------------------------------------------------------------------------------
                    # Segundo elemento: Los conteos del número de veces que se observo la categoría => 𝐶(𝑐)
                    # A este segundo elemento le sumamos la longitud del vocabulario.
                    # -------------------------------------------------------------------------------------
                    log_wordClassProb = math.log(
                      (self.wordConditionalCounts[c].get(word, 0.0)+1)/(self.classCount[c]+len(self.vocab)))
                    
                    # El valor calculado anteriormente se agrega al score probabilístico de la categoría 'c', la
                    # categoría en la que esta ubicada el bucle.
                    # -------------------------------------------------------------------------------------
                    # Primero busca si ya hay un valor asignado a esa categoría, sí si, deja el valor, sí no, 
                    # asigna inicialmente el valor correspondiente calculado del logaritmo de la probabilidad 
                    # de esa categoría, luego sumamos el valor calculao del logaritmo del conteo condicional.
                    # -------------------------------------------------------------------------------------
                    scoreProb[c] = scoreProb.get(c, self.log_classPriorProb[c]) + log_wordClassProb
                    
            # Definimos el argumento de máxima probabilidad con la función 'np.argmax' y obtenemos el 
            # máximo valor en toda la lista de scores de probabilidad de los argumentos. La variable 
            # 'arg_maxprob' guada la posición donde esta la mayor probabilidad    
            arg_maxprob = np.argmax(np.array(list(scoreProb.values())))
            
            # Finalmente agregamos la máxima probabilidad correspondiente a ese argumento en los resultados,
            # buscando en el diccionario de scores (Convertimos el diccionario a una lista) por la posición 
            # previamente encontrada.
            results.append(list(scoreProb.keys())[arg_maxprob])
            
        # Una vez se hace el proceso para todos los textos en el conjunto de datos retornamos el resultado  
        return results


# ### Utilidades de Scikit Learn
# * `train_test_split`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# 
# * `accuracy_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# 
# * `precision_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
# 
# * `recall_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
data_train, data_test, clases_train, clases_test = train_test_split(data, clases, test_size=0.10, random_state=42)


# In[21]:


classifier = NaiveBayesClassifier()
classifier.fit(data_train, clases_train)


# In[22]:


clases_predict = classifier.predict(data_test)


# In[23]:


accuracy_score(clases_test, clases_predict)


# In[24]:


precision_score(clases_test, clases_predict, average=None, zero_division=1)


# `precision_score` para este caso en particular **array([0.82687927, 1.        ])**, el valor `0.82` esta diciendo que de todo lo que logre  predecir como `ham`, el **82%** es realmente `ham`. Por otra lado, el valor `1` esta diciendo que de todo lo que logre  predecir como `spam`, el **100%** es realmente `spam`.

# In[26]:


recall_score(clases_test, clases_predict, average=None, zero_division=1)


# `recall_score` para este caso en pariticular **array([1.        , 0.50967742])**, el valor `1` esta diciendo que de todo lo que en el dataset es realmente `ham`, se logro capturar ese **100%**. Por otro lado, el valor `0.509` esta diciendo que de todo lo que en el dataset es realmente `spam`, se logro capturar solo el **50%**.

# In[ ]:


get_ipython().system('jupyter nbconvert --to=python 1_NaiveBayes_class.ipynb')

