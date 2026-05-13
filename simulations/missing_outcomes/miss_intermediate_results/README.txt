## Version as of April 6 2026:
## Organization of the missing outcome intermediate results:

Each setting has a folder. 

We save results for all procedures and methods applied to a setting in that setting's folder of intermediate results.

Compilation of all results for all missing data will consume this setting specific analysis which will be saved as all_missing_data_res.csv in the miss_intermediate_results

Figure and Table preparation per setting, procedure, method and model will then consume this .csv file and the results are piped down into the miss_paper_results folder. Again, this will be grouped in folders by the setting.

Note that in the main paper, the presentation of the results on the missing outcome data analysis are by the missingness setting.

In the main paper, we present results for setting three, which is the most interesting given the gross misspecification of the analysts model and the two-layered bias + misspecification effect. It would be very challenging for a model specified by the analysts to recover the correct functional forms of the covariates and any adjustment procedure or model that performs well in terms of reducing bias under missing outcome data is guaranteed to yield an even better performance under missing outcome data scenarios specified in settings one and two (add section ref)

We compare our single stage regression adjustment with variants adopting double ML, TMLE, and AIPTW. Our choice is informed by the fact that our approach under missing outcome data includes an additional imputation step where potential outcomes are imputed for individuals missing outcomes and is a realistic choice compared to a complete case analysis which has been shown to yield biased estimates of treatment effects [cite why complete case analysis is wrong and when].

We focus on the third setting [reiterate on why the third simulation setting is the most challenging]

