# ML
Project 1-ML
## TODO
- Some refs say to discard all the columns that have '_phi_' in it. Should test that.
- Finetune the preprocessing. You'll see in the histogram after preprocessing that a lot of values are centers in 0. Why the fuck ? should we just remove this ?
- Maybe try a polynomial augmentation ? (Or fourier augmentation,  works really well with images/sound...)
- Remove the outliers (I have no clue how to do this)
- Try 'Mean' and 'Median' for the filling od the missing values
- Actually implement some models and to some testing
- ..
---

### Done by Alex 13.10
Mainly preprocessing:
- I've read the article on the physics behind, and from it i removed the features of columns (22 to 28) because they actually do not make any sense! It's actually the columns where there are ~70% of missing values so here ya go
- The 'jet_num' columns is actually discrete valued feature that categorizes the type of 'event' (physics stuff) that occured. So I don't know If we should discard it or if we should use this info... I'll look more into this!
- Refactored the code a bit, thx Alicia for your seaborn mastery damn these plots are awesome :O

---
## Refs
- https://github.com/phunterlau/kaggle_higgs/tree/97c5b3d6ffebb772bbb6a586f00b485a2dc9355a (25th of original challenge)
- https://www.kaggle.com/competitions/higgs-boson/discussion/10425 (The winner of the original challenge)
- https://www.kaggle.com/code/sugataghosh/implementing-logistic-regression-from-scratch  (random dude)
- https://github.com/dufourc1/ML_CS433_project1/blob/master/src/run.py  (former EPFL student)