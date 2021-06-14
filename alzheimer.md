# Alzheimer Notes 


### Tanvir sir's code 

**outer, inner?**
- Uses an outer and an inner cross validator. In the original code, both were 10-fold cross validators. 
- The inner fold is got from the outer-fold's training data. Classifiers are trained and results are predicted on both inner and outer fold's data separately. So, classifiers in outer fold train on 75% data, in inner fold train on (.75 x 0.75) = 56.25% data. 
- The results are appended in separate dataframe for inner and outer fold. Finally the mean is shown. 

Surprisingly, 3-fold gives 4-5% less accuracy on all classifiers than 10-fold. Why? 

**CV** = Cross validation 