# Time-Series-Methods
Understanding various models that can be used for time series analysis
## Naive Approach

In this forecasting technique, we assume that the next expected point is equal to the last observed point. So we can expect a straight horizontal line as the prediction.

## Moving Average

In this technique we will take the average of counts for last few time periods only.

## Simple Exponential Smoothing

In this technique, we assign larger weights to more recent observations than to observations from the distant past.
The weights decrease exponentially as observations come from further in the past, the smallest weights are associated with the oldest observations.

## Holt’s Linear Trend Model

It is an extension of simple exponential smoothing to allow forecasting of data with a trend.
This method takes into account the trend of the dataset. The forecast function in this method is a function of level and trend.

The accuracy of the model is calculated using root mean square error for every model.

## References: 
Analytixvidhya
