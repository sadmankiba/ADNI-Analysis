# Alzheimer Notes 


### Tanvir sir's code 

**outer, inner?**
- Uses an outer and an inner cross validator. In the original code, both were 10-fold cross validators. 
- The inner fold is got from the outer-fold's training data. The classifiers run both in inner and outer fold's data. So, classifier's in outer fold train on 75% data , in inner fold train on (.75 x 0.75) = 56.25% data. 
- The results are appended, finally the mean is shown. 

Surprisingly, 3-fold gives less 4-5% less accuracy on all classifiers than 10-fold. Why? 

**CV** = Cross validation 