# scale_rotation_invariant_match

Инвариантная к повортоу и масштабу регистрация фрагмента изображения (template) на снимке.

![alt text](https://github.com/VolshevskyAlex/scale_rotation_invariant_match/blob/main/doc/6.png)
![alt text](https://github.com/VolshevskyAlex/scale_rotation_invariant_match/blob/main/doc/6a.png)
a) шаблон поиска  
b) тайл с максимальным откликом  
c) аффинная трансформация  

![alt text](https://github.com/VolshevskyAlex/scale_rotation_invariant_match/blob/main/doc/7.png)

response map, белая окржность - максимальный отклик, красные радиусы - показывают угол поворота  

На первом проходе необходимо вызвать функцию gen_tiles(), которая запишет в файл для каждого тайла его спектр log_polar предстваления.  
На втором проходе загружается этот файл.  
Рандомно выбирается точка на снимке (в пределах roi), рандомно выбирается rotation, sclae - получаем img4search.  
Ищем по тайлам, выбираем с максимальным откликом.  
Далее, уже зная rotaion, sclae находим сдвиг.  

download.sh - загрузка тестового изображения png

requirements:
OpenCV
