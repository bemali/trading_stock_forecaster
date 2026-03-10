# Plan for the project
- this project is to provide an estimate for a certain stock with the help of a forecaster model, and market sentiments. 

#### input
- a ticker

#### resources
- news items recorded in a postgres database in cloud, with some research analysis points, indicators along with the ticker
- yahoo finance to extract past records for the ticker

#### output
- a prediction: initially qualitative for the stock based on the stock
- baseline forecast: from the forecaster 
- underlying reasoning
- validation based on previous instances (overlayed on the graph perhaps?)

#### deployment
- as an azure container app (backend)
- as an azure webapp (front end)