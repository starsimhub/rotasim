age_results<-data.frame(age=c('0-11 m', '12-23 m', '24-59 m', '60+ m'), 
                        source=c(rep('model', 4), rep('data', 4)), 
                        cases_age=c(11.9, 27.5, 48.0, 0, 13.8, 27.7, 46.9, 11.7),
                        LCL=c(7.6, 18.5, 36.4, 0, NA, NA, NA, NA), 
                        UCL=c(26.8, 38.6, 59.1, 0, NA, NA, NA, NA))

require(ggplot2)
ggplot(data=age_results, aes(x=age, y=cases_age, fill=source))+
  geom_bar(stat='identity', position=position_dodge())+
  geom_errorbar(ymin=age_results$LCL, ymax=age_results$UCL, position=position_dodge())+
  theme_bw()+lims(y=c(0, 60))+labs(x='Age group', y='Proportion of cases in age group')

incidence_results<-data.frame(IR=c(27.6, 11.5), 
                              source=c('data', 'model'), LCL=c(NA, 9.9), UCL=c(NA, 14.9))

ggplot(data=incidence_results, aes(x=source, y=IR, fill=source))+
  geom_bar(stat='identity')+
  geom_errorbar(ymin=incidence_results$LCL, ymax=incidence_results$UCL)+
  theme_bw()+lims(y=c(0, 35))+labs(x='Source', y='Cases per 100k person years')

MLE_age_risk<-function(age, beta0=-1.421, beta1=0.171, beta2=-0.004516){
  age_centered<-min(age,60)-12
  linear_predictor<-beta0+beta1*age_centered+beta2*(age_centered^2)
  probability<-1/(1+exp(-1*linear_predictor))
  return(probability)
}

MLE_age_risk(age=6)
age_vec<-seq(from=0, to=75, by=3)
perc_vec<-numeric(length(age_vec))
for(i in 1:length(age_vec)){
  perc_vec[i]<-MLE_age_risk(age=age_vec[i])
}
age_output<-data.frame(age_month=age_vec, percent_severe=perc_vec*100)

ggplot(data=age_output, aes(x=age_month, y=percent_severe))+geom_line()+theme_bw()+
  labs(x='Age (months)', y='Predicted Percent Symptomatic')

age_results<-data.frame(age=c('0-11 m', '12-23 m', '24-59 m', '60+ m'), 
                        source=c(rep('model', 4), rep('data', 4)), 
                        cases_age=c(20.4, 15.9, 20.8, 0, 13.8, 27.7, 46.9, 11.7),
                        LCL=c(0, 0, 0, 0, NA, NA, NA, NA), 
                        UCL=c(44.4, 37.8, 49.0, 0, NA, NA, NA, NA))


ggplot(data=age_results, aes(x=age, y=cases_age, fill=source))+
  geom_bar(stat='identity', position=position_dodge())+
  geom_errorbar(ymin=age_results$LCL, ymax=age_results$UCL, position=position_dodge())+
  theme_bw()+lims(y=c(0, 60))+labs(x='Age group', y='Proportion of cases in age group')

incidence_results<-data.frame(IR=c(27.6, 3.27), 
                              source=c('data', 'model'), LCL=c(NA, 1.4), UCL=c(NA, 5.36))

ggplot(data=incidence_results, aes(x=source, y=IR, fill=source))+
  geom_bar(stat='identity')+
  geom_errorbar(ymin=incidence_results$LCL, ymax=incidence_results$UCL)+
  theme_bw()+lims(y=c(0, 35))+labs(x='Source', y='Cases per 100k person years')