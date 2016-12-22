# -*- coding: utf-8 -*-
import xml.etree.ElementTree as et
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

class XML_parser():
    '''
    Класс для парсинга XML-файлов и преобразования данных для обучения или тестирования.
    '''
    def xml_parse(self, filename):
        root = et.parse(filename).getroot()
        twits = []
        classes = []
        for table in root.iter('table'):    # если в table содержится несколько оценок, то такие отзывы не учитываются
            aux_count = 0
            for column in table[4:]:
                if column.text != 'NULL':
                    value = column.text
                    aux_count += 1
            if aux_count == 1:      # добавление в список твитов и меток классов
                classes.append(value)
                twits.append(table[3].text)
        target = np.array(classes)      # преобразование оценок в массив меток классов
        return twits, target

    def vectorize_freq(self, twits, twits_test):
        '''
        Feature-extractor (мешок слов, признаки - частоты слов).
        '''
        vectorizer = CountVectorizer()
        extractor = vectorizer.fit_transform(twits)
        data_train = extractor.toarray()
        data_test = vectorizer.transform(twits_test).toarray()
        return data_train, data_test
    
    def vectorize_tfidf(self, twits, twits_test):
        '''
        Feature-extractor (мешок слов, признаки - tf-idf). Не используется, но может быть использована при solver='liblinear' и multi_class='ovr'.
        '''
        vectorizer = TfidfVectorizer(smooth_idf=False)
        extractor = vectorizer.fit_transform(twits)
        data_train = extractor.toarray()
        data_test = vectorizer.transform(twits_test).toarray()
        return data_train, data_test

class Classifier():
    '''
    Класс для классификации с помощью логистической регрессии.
    '''
    def __init__(self):
        self.__st_model = None
        
    def log_reg_train(self, data_train, target_train):
        '''
        Функция для обучения логистической регрессии.
        '''
        logreg = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', C=0.5)
        logreg.fit(data_train, target_train)  # обучение
        self.__st_model = pickle.dumps(logreg)  # сохранение модели
        
    def log_reg_test(self, data_test, target_test):
        '''
        Функция для тестирования.
        '''
        logreg = pickle.loads(self.__st_model)      # загрузка сохранённой модели
        predicted = logreg.predict(data_test)       # тестирование
        print('LogisticRegression: \n', classification_report(target_test, predicted))    # вывод результатов
        print('Accuracy: ', accuracy_score(target_test, predicted))

if __name__ == '__main__':
    print('Banks: \n')
    parser = XML_parser()
    twits_train, target_train = parser.xml_parse('bank_train_2016.xml')     # парсинг обучающей выборки
    twits_test, target_test = parser.xml_parse('banks_test_etalon.xml')          # парсинг тестовой выборки
    data_train, data_test = parser.vectorize_freq(twits_train, twits_test)      # преобразование в вектора
    
    classifier = Classifier()   
    train = classifier.log_reg_train(data_train, target_train)   # обучение с помощью логистической регрессии
    test = classifier.log_reg_test(data_test, target_test)      # тестирование
     
    print('Telecom: \n')
    parser = XML_parser()
    twits_train, target_train = parser.xml_parse('tkk_train_2016.xml')           # парсинг обучающей выборки
    twits_test, target_test = parser.xml_parse('tkk_test_etalon.xml')       # парсинг тестовой выборки
    data_train, data_test = parser.vectorize_freq(twits_train, twits_test)      # преобразование в вектора
      
    train = classifier.log_reg_train(data_train, target_train)   # обучение с помощью логистической регрессии
    test = classifier.log_reg_test(data_test, target_test)      # тестирование