# Tweet-sentiment-analysis
Оценка эмоциональности твитов с помощью алгоритма логистической регрессии. Оценка проводилась для банков и компаний, предоставляющих телекоммуникационные услуги. Классификация трёхклассовая: положительные, отрицательные и нейтральные отзывы.

Данные отсюда: https://goo.gl/GhX3vU

Статья о соревновании SentiRuEval-2016: http://www.dialog-21.ru/media/3410/loukachevitchnvrubtsovayv.pdf

Результаты:

Banks: 

LogisticRegression: 
              precision    recall  f1-score   support

         -1       0.62      0.51      0.56       751
          0       0.78      0.89      0.83      2164
          1       0.54      0.27      0.36       297

avg / total       0.72      0.74      0.72      3212

Accuracy:  0.741594022416

Telecom: 

LogisticRegression: 
              precision    recall  f1-score   support

         -1       0.72      0.68      0.70       902
          0       0.67      0.79      0.72       972
          1       0.43      0.16      0.23       180

avg / total       0.67      0.68      0.67      2054

Accuracy:  0.682570593963
