# Alzheimer Notes 

### Training on ADNI merge 
After filling in missing values and dropping empty DX, 
Baseline - ('CN', 828), ('Dementia', 402), ('MCI', 1053)
Last visit - ('CN', 400), ('Dementia', 548), ('MCI', 542)
All visit - ('CN', 3685), ('Dementia', 2355), ('MCI', 4715)

Number of features without CDRSB
Baseline - 22
Last visit - 20
All visit - 20

### Tanvir sir's code 

**outer, inner?**
- Uses an outer and an inner cross validator. In the original code, both were 10-fold cross validators. 
- The inner fold is got from the outer-fold's training data. Classifiers are trained and results are predicted on both inner and outer fold's data separately. So, classifiers in outer fold train on 75% data, in inner fold train on (.75 x 0.75) = 56.25% data. 
- The results are appended in separate dataframe for inner and outer fold. Finally the mean is shown. 

Surprisingly, 3-fold gives 4-5% less accuracy on all classifiers than 10-fold. Why? 

**CV** = Cross validation 