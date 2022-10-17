# ML
Project 1-ML
## TODO
### Most important
- Finetune the preprocessing. Some refs say to discard all the columns that have '_phi_' in it. Should test that. Also we might want to log transform some of the features, I think I have seen this done in a few projects but cannot find the references anyomore. Finally trhere are some features that are categorical, and so far the preprocessing function treats them like the other, which it should not so fix needed here. That is maybe why the polynomial augmentation does not work properly so far.
- You'll see in the histogram after preprocessing that a lot of values are centers in 0. Why the fuck ? should we just remove this ?
- Cross-validation (!!!)
- Refactor the code so we can do some grid-search easily (overnight on the cluster for example)
- Start the report
### Less important
- Maybe try a fourier augmentation ? ( works really well with images/sound...)
- Remove the outliers (I have no clue how to do this). Does it even make sense ?
- Try 'Mean' and 'Median' for the filling of the missing values
- ..
---

### Done by Alex 13.10
Mainly preprocessing:
- I've read the article on the physics behind, and from it i removed the features of columns (22 to 28) because they actually do not make any sense! It's actually the columns where there are ~70% of missing values so here ya go
- The 'jet_num' columns is actually discrete valued feature that categorizes the type of 'event' (physics stuff) that occured. So I don't know If we should discard it or if we should use this info... I'll look more into this!
- Refactored the code a bit, thx Alicia for your seaborn mastery damn these plots are awesome :O

### Done by Alex 17.10
- Implemented all the missing models
- Polynomial expansion (which does not seem to work great, I think preprocessing should be finetuned)
---
## Refs
- https://github.com/phunterlau/kaggle_higgs/tree/97c5b3d6ffebb772bbb6a586f00b485a2dc9355a (25th of original challenge)
- https://www.kaggle.com/competitions/higgs-boson/discussion/10425 (The winner of the original challenge)
- https://www.kaggle.com/code/sugataghosh/implementing-logistic-regression-from-scratch  (random dude)
- https://github.com/dufourc1/ML_CS433_project1/blob/master/src/run.py  (former EPFL student)