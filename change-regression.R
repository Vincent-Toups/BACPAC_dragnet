library(tidyverse);
library(gbm);
source('util.R');

ex_data <- do.call(rbind,
                   Map(read_csv,
                       read_csv("derived_data/meta-data.csv") %>%
                       filter(domain=="EX" & archive==F) %>%
                       pull(file))) %>% select(-STUDYID,-DOMAIN);

dm_data <- do.call(rbind,
                   Map(function(file){
                       read_csv(file) %>% 
                       select(AGE,
                              BRTHDTC,
                              DOMAIN,
                              ETHNIC,
                              RACE,
                              RACEMULT,
                              RFPENDTC,
                              RFSTDTC,
                              SEX,
                              STUDYID,
                              USUBJID);
                   },
                       read_csv("derived_data/meta-data.csv")  %>%
                       filter(domain=="DM" & archive==F) %>%
                       pull(file))) %>% select(-STUDYID,-DOMAIN);


ex_wide_time <- pivot_wider(ex_data,
                               id_cols=c("USUBJID","EXDY"),
                               names_from="EXTRT",
                               values_from="EXCAT",
                               values_fn=function(...) 1,
                            values_fill=0) %>%
    group_by(USUBJID) %>% arrange(EXDY) %>% 
    summarize(`Mindfulness or meditation or relaxation`={
        start_ii <- min(which(`Mindfulness or meditation or relaxation`==1));
        if (start_ii == Inf) {
            -1;
        } else {
            EXDY[[start_ii]];
        }
    },
    `Exercise`={
        start_ii <- min(which(`Exercise`==1));
        if (start_ii == Inf) {
            -1;
        } else {
            EXDY[[start_ii]];
        }
    },
    `NSAIDs`={
            start_ii <- min(which(`NSAIDs`==1));
        if (start_ii == Inf) {
            -1;
        } else {
            EXDY[[start_ii]];
        }
    },
    `Opioids`={
        start_ii <- min(which(`Opioids`==1));
        if (start_ii == Inf) {
            -1;
        } else {
            EXDY[[start_ii]];
        }
    },
    `Diet or weight loss program`={
        start_ii <- min(which(`Diet or weight loss program`==1));
        if (start_ii == Inf) {
            -1;
        } else {
            EXDY[[start_ii]];
        }
    },
    `Non-spinal fusion`={
        start_ii <- min(which(`Non-spinal fusion`==1));
        if (start_ii == Inf) {
            -1;
        } else {
            EXDY[[start_ii]];
        }
    },
    `Therapy or counseling`={
        start_ii <- min(which(`Therapy or counseling`==1));
        if (start_ii == Inf) {
            -1;
        } else {
            EXDY[[start_ii]];
        }
    },
    `SSRI_SNRI`={
        start_ii <- min(which(`SSRI_SNRI`==1));
        if (start_ii == Inf) {
            -1;
        } else {
            EXDY[[start_ii]];
        }
    },
    `Acupuncture`={
        start_ii <- min(which(`Acupuncture`==1));
        if (start_ii == Inf) {
            -1;
        } else {
            EXDY[[start_ii]];
        }
    },
    `Spinal fusion`={
        start_ii <- min(which(`Spinal fusion`==1));
        if (start_ii == Inf) {
            -1;
        } else {
            EXDY[[start_ii]];
        }
    },
    `Gabapentin or pregabalin`={
        start_ii <- min(which(`Gabapentin or pregabalin`==1));
        if (start_ii == Inf) {
            -1;
        } else {
            EXDY[[start_ii]];
        }
    },
    `Tricyclic antidepressants`={
        start_ii <- min(which(`Tricyclic antidepressants`==1));
        if (start_ii == Inf) {
            -1;
        } else {
            EXDY[[start_ii]];
        }
    }) %>% 
    inner_join(read_csv("derived_data/subject-changes.csv"),by="USUBJID") %>%
    filter(`Visit Count` >= 6) %>%
    mutate(improved=1.0*(Change>5.0)) %>%
    inner_join(dm_data,by="USUBJID");

write_csv(ex_wide_time, "derived_data/ex-wide-gbm-encoded.csv");


k <- 5;

fold_ii <- ex_wide_time %>% group_by(improved) %>% mutate(fold=(floor(seq(0,0.999,length.out=length(Change))*k))+1) %>% ungroup() %>% pull(fold);


f <-  Change ~ (`Exercise`)                                +
                        (`Mindfulness or meditation or relaxation`) +
                        (`NSAIDs`)                                  +
                        (`Opioids`)                                 +
                        (`Diet or weight loss program`)             +
                        (`Non-spinal fusion`)                       +
                        (`Therapy or counseling`)                   +
                        (`SSRI_SNRI`)                               +
                        (`Acupuncture`)                             +
                        (`Spinal fusion`)                           +
                        (`Gabapentin or pregabalin`)                +
    (`Tricyclic antidepressants`) +
    AGE +
    factor(ETHNIC) +
    factor(RACE) +
    factor(SEX);


characterization <- do.call(rbind, Map(function(i){
    train_ii <- fold_ii != i#runif(nrow(ex_wide_time)) < 0.75;
    train <- ex_wide_time %>% filter(train_ii);
    test <- ex_wide_time %>% filter(!train_ii);


    model <- gbm(f,
                 distribution='gaussian',
                 data=train,
                 interaction.depth=8,
                 n.trees=10000);
    prediction = predict.gbm(model, newdata=test)
    tibble(prediction=prediction,actual=test$Change,fold=i) %>%
        mutate(error=sqrt((prediction-actual)*(prediction-actual)));
},1:k));




