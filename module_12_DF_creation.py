#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


class DataFrame_Creation:
    
    def create_result_dataframe(self,pred_1,pred_2):
        
        catagories = ['service', 'anecdotes/miscellaneous', 'food', 'price', 'ambience']
        polarity = ['negative', 'positive', 'neutral', 'conflict']
        
        # Calling DataFrame constructor on predicted outputs
        resultant_df = pd.DataFrame(list(zip(pred_1,pred_2)), columns = ["predicted_catagories" , "predicted_polarity"])
        result  = pd.crosstab(resultant_df.predicted_catagories,resultant_df.predicted_polarity ,margins = True , margins_name = "Total")
        result["Ranking"] = ( result.Total/resultant_df.shape[0]) * 5.0 
        
        if polarity[0] in result.columns:
            result[polarity[0]+" %"] = (result.negative/result.Total) * 100
            del result[polarity[0]]
        else:
            result[polarity[0]+" %"] = 0.00
        
        if polarity[1] in result.columns:
            result[polarity[1]+" %"] = (result.positive/result.Total) * 100 
            del result[polarity[1]]
        else:
            result[polarity[1]+" %"] = 0.00 
        
        if polarity[2] in result.columns:
            result[polarity[2]+" %"] = (result.neutral/result.Total) * 100
            del result[polarity[2]]
        else:
            result[polarity[2]+" %"] = 0.00
            
        if polarity[3] in result.columns:
            result[polarity[3]+" %"] = (result.conflict/result.Total) * 100 
            del result[polarity[3]]
        else:
            result[polarity[3]+" %"] = 0.00 
        
        del result["Total"]
        result.drop(result.tail(1).index,inplace=True)
        
        return result

