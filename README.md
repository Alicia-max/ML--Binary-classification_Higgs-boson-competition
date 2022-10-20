# ML
Project 1-ML
## TODO
### Most important
- Do standardization after polynomial augmentation
- Average the accuracies on the folds and add the error
- Start the report
- Write a proper README, comment the code, make the variable name coherent... (I'll take care of this i'm super OCD)
- Finetune the preprocessing. Some refs say to discard all the columns that have '_phi_' in it. Should test that. Also we might want to log transform some of the features, I think I have seen this done in a few projects but cannot find the references anyomore. Finally trhere are some features that are categorical, and so far the preprocessing function treats them like the other, which it should not so fix needed here. That is maybe why the polynomial augmentation does not work properly so far.
- You'll see in the histogram after preprocessing that a lot of values are centers in 0. Why the fuck ? should we just remove this ?
### Less important
- Maybe try a fourier augmentation ? ( works really well with images/sound...)
- ..
---
### Done bz Alex 20.10
- Finalized cross-validation, now we can run with list of parameters. check visu.ipynb for more info.
- Tried it on all the models. Ridge is indeed the one that works best.
- Added again the features we were not sure about. (the ones with $\phi$ in the name)
### Done by Sevda 18.10
I have observed the following:
- I tried with mean instead of median and the accuracy was slightly better but not much (~0.5%)
- I blacklisted the columns of 4-5-6 and 15-18-20 but didn’t observe an improvement. However, I tried with just removing the “_phi” features as in the winning model and it was again slightly better than the case with removing all of 4-5-6-15-18-20-and so on. Also, I remember that there was a source which does take the log of some features but the winning model that I skimmed through took “log(1+x) - log(1-x)” of a feature x if I understood correctly. I tried to understand the reasoning behind but couldn’t get so much out of it.
- I started coding K-fold cross validation but it wasn’t done so I will probably work on that on either thursday or friday. Also do you think K-fold is OK or should we do leave-one-our or something other than these?

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