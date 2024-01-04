# Проект по программной инженерии

## Описание скрипта для 1 практической работы
Скрипт использует модель sentence-transformers/LaBSE. Эта модель предназначена
для оценки схожести предложений. Подробнее здесь - https://huggingface.co/sentence-transformers/LaBSE.  
В данном скрипте реализован поиск 5 самых похожих друг на друг предложения.  
Входные данные находятся в переменной sentences, для их изменения нужно отредактировать или добавить новые строки
в эту переменную.  
На выходе будет 5 наиболее похожих пар с оценкой похожести.

## Описание скрипта для 2 практической работы
Для запуска скрипта необходимо из папки 2_практическая_работа вызвать команду:  
```
streamlit run .\web_app.py
```
Ссылка на документацию streamlit - https://streamlit.io/  
Разработанное веб-приложение является легкой реализацией антиплагиата с помощью нейросети.  
Используемая модель точно такая же, как и в 1 практической работе.  

Это приложение позволяет сравнить два текста между собой по предложениям и определяет 
похожесть текстов друга на друга. Для работы выберите какими разделителями разделены предложения,
введите последовательно два текста, которые хотите сравнить в соответствующие окна и нажмите кнопку сравнить.
Будет отображаться процент похожести текстов друг на друга.  

Алгоритм сравнения просто: тексты сравниваются по предложениям, каждое с каждым. 
После сравнения для каждой пары предложения формируется оценка. 
Далее берется для каждого предложения из первого текста ищется максимальная оценка среди пар образованных с предложениями из второго текста и суммируется.
Результат делится на количество предложений в первом тексте.
Таким образом, получается более релевантная оценка похожести предложений и соответственно текстов
(к примеру перестановки предложений слабо влияют на общую оценку), 
но все же это больше демонстрационное приложение, поэтому для повсеместного применения не подходит.

## Описание скрипта для 3 практической работы
В основе веб-сервера лежит модель из первых двух практических работ  
Для запуска сервер fast api, нужно выполнить команду - uvicorn api:app --reload, предварительно установив библиотеки
из файла requirements.txt.  
Приложение запустится по умолчанию на 8000 порту, по адресу http://localhost:8000/docs , будет доступенн swagger ui.  
Приложение предоставляет 2 метода:  
1. Первый метод проводит сравнительный анализ двух текстов на сходство между собой и на выходе отдает число вероятности
сходства текстов. Метод принимает 2 обязательных параметра: first_text - первый текст для сравнения, second_text - 
второй текст для сравнения. Также метод принимает 2 необязательных параметра: separators - список разделителей предложений
по умолчанию равен ['?', '.', '!'], is_percent - булевый флаг, если передать true, то вывод вероятности сходства будет
в процента округленного до 2 знаков, с символом процента.
2. Второй метод проводит сравнительный анализ двух текстов на сходство между собой, по принципу каждое предложение с
каждым. Набор входных параметров такой же как и для первого метода. На выходе массив объектов, каждый объект состоит
из 3 полей: first_sentence - предложение из первого текста, second_sentence - предложение из второго текста,
similarity - число или строка означающая вероятность сходства предложений между собой.

## Описание для 4 практической работы
В рамках 4 практической работы, был написан Dockerfile и docker-compose.yaml файл для быстрого развертывания api и модели.  
Для локального запуска docker контейнера, необходимо из папки с docker-compose файлом выполнить команду - docker compose up.  
Также приложение было развернуто на яндекс облаке с помощью docker и swagger доступен по ссылке - http://158.160.136.193/docs

## Описание для 5 практической работы
В рамках 5 практической работы, были написаны тесты для api сделанного в рамках прошлых работ,
тесты покрывают основные сценарии использования api, такие как проверка корректности ответа от входных данных,
передача неправильных входных параметров. Для локального запуска тестов необходимо сначала установить
все зависимости из requirements_test.txt, далее из папки  с тестами выполнить команду pytest.







## Состав команды
1. Коньков Владислав Александрович - РИМ-130908
2. Копотев Никита Викторович - РИМ-130908
