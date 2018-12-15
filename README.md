# Actually Decent Weather Prediction
## CSCI 4502/5502
### Jacob Hallberg, Landon Baxter, Sara Park, Robert Renecker

The project, Actually Decent Weather Prediction, is our final project for our Data Mining course (CSCI 4502/5502) at the University of Colorado at Boulder. This project is a study on how Machine Learning techniques, specifically neural networks, can augment traditional weather prediction. 

The following information is excerpted from our final paper:

## Motivation:
Machine learning and neural networks are being used in more and more applications every day. Given the inconsistency and poor quality of weather descriptions – especially in an unpredictable state like Colorado – the group decided it would be interesting to see how neural networks could be used for describing the weather. Our team will further explore Deep Learning techniques by attempting to model various neural networks on hourly weather data and satellite images.

## Approaches and Model Creation:

### Original Approach:
The team’s original approach was to utilize the "Historical Hourly Weather Data 2012-2017" dataset from kaggle.com. For our original approach we used a total of three different deep neural networks to solve the classification tasks at hand. The first network consisted of a fully connected neural network and was used for hour-to-hour prediction. The second network we used was a recurrent neural network with LSTMs and a fully connected output layer. Finally, the third neural network was a convolutional-recurrent-fully-connected neural network. The second and third models were used for the 24 hour sequence predictions where the model is given 24 hours of weather data and is asked to predict the 25th hour's weather description.

* The kaggle dataset can be found here: https://www.kaggle.com/selfishgene/historical-hourly-weather-data\#weather\_description.csv *

### Stretch Goal Approach:
Once our team got results from our original approach, we wanted to see if we could predict the weather more accurately with image data. Therefore, in our stretch goal approach, the team utilized a new data set with satellite image data. This data is provided by the National Oceanic and Atmospheric Administration (NOAA) and contains timely images (taken in 5-10 minute increments) of weather patterns. The team then used the "Historical Hourly Weather Data 2012-2017" from our previous approach and matched the weather description from that data set with the NOAA radar image data set. The team wanted to achieve more accurate results with this image data alone as well as in conjunction with results from our original approach. We decided to tackle our stretch goal by using two models. 

For our stretch goal approach we used a total of two different deep neural networks. The first was a typical all convolutional network that made use of global averagepooling as the final layer and then applied the softmax activation function to the logits. The second was a non-traditional convolutional long short-term memory (C-LSTM) network that used the C-LSTMs to embed the sequence of temporal radar data down to a non temporal embedding layer or latent space and then used deconvolutions or convolutional transposes to build from the embedding-space to generate new radar images that denoted the model’s prediction into the future. We take n=4 hours worth of radar imaging for a total of 20 images and feed it through the model. The network then outputs the radar image prediction corresponding to the hour after the input data; in other words, it provides the set of five radar images from an hour in the future (relative to the input).

* NOAA dataset can be ordered through this link: https://www.ncdc.noaa.gov/cdo-web/search *

## Results:
Our original approaches overall performed well given the sparse data set that was used. Traditional weather forecasts use a collection of radar images and weather data collected from multiple satellites and weather stations. However, our approach made use of a single weather station and only 5 state variables to make a forecast.
The first model that was used reached an accuracy of roughly 20-30%, but given limitations in the data set such as syntactically similar labels, the definition of how the accuracy was interpreted might of been one of the causes for such a low accuracy.
Using the same interpretation, the most recent model that was used was a Long-Short Term Memory model (LSTM) that gave an accuracy of roughly 70%. 
Our stretch goal approaches had some interesting results. We found that the CNN model that was fed 5 radar images for a given hour did not perform as well as our point based approach. The CNN model had an accuracy of about 50%, in comparison to the original approach's accuracy of 70%. However, the CNN architecture was completely new and we had yet to see anything similar in any of our research efforts. This architecture has many advantages over the point based original approach because the model has spatial information spread about the Denver area, whereas the point based approach does not. Given this, exploration of additional CNN architectures would surely improve over all accuracy. As mentioned previously in the report, architecture design can be a very lengthy process and one change to the design of an architecture can drastically improve results.



## Execution Traces
The primary execution of our models was done in Jupyter notebooks. As such, you can find our execution traces in the models/\*.ipynb files. Also we took a few screenshots of execution traces for convience sake and they are placed in ./traces/.



