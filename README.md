# ML
Project 1-ML
## TODO
- Data exploration, preprocessing and exploration
- Implement all the models
- ..
---
### Ideas for data preprocessing:
- First simply normalize the features column-wise  
- In https://www.kaggle.com/code/sugataghosh/implementing-logistic-regression-from-scratch  
the author removes the features that are composed of 30 % or more of nans. Then for the other ones, the nans are filled wit either 'Mean or 'Median'. We can try both and compare them, it could be a nice analysis for the report.
- In https://github.com/dufourc1/ML_CS433_project1/blob/master/src/run.py  
  they do polynomial feature extraction, no clue why but might be a good Idea ?
- Assuming we use these two feature processing methods, we could do an ablation study afterwards, which is generally nicely seen in ML papers