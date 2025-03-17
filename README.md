# Predicting Stock Market Prices using Sentiment Analysis - README

## Overview

This repository explores various machine learning models to predict stock market movements based on sentiment analysis from news articles. The aim was to investigate the following: 
- Test whether applying Transfer Learning, which is fine-tuning an LLM like RoBERTa improves sentiment analysis compared to traditional NLP packages like Financial Vader
- Test the stability and effectiveness of different models and feature sets in forecasting stock price changes. 

Our analysis utilized several approaches including **Logistic Regression (Log)**, **Boosted Trees (BST)**, and **Long Short-Term Memory Networks (LSTM)**, compared against a **Baseline model**. The Baseline model in our context simply predicts that the stock price will always go up. The performance of each model was measured by the average percentage growth in stock prices.

## Results

The results of our simulation are as follows:

- **Logistic Regression (Log):** growth
- **Boosted Trees (BST):** growth
- **Long Short-Term Memory Networks (LSTM):**  growth
- **Baseline Model:** - growth

### Observations

1. **Performance Evaluation:**
   - The **LSTM** model achieved the highest average growth, suggesting it could capture complex temporal dependencies better than other models.
   - The **Boosted Trees** model also performed reasonably well, likely due to its ability to handle non-linear relationships.
   - The **Logistic Regression** model showed limited effectiveness, often performing similarly to the **Baseline model**. In some cases, it even degenerated to the level of the Baseline, particularly with tech stocks.
   
2. **Feature Analysis:**
   - Including average FinVader sentiment scores over 3/7 days led to performance decreases in the **Logistic Regression** and **BST** models, indicating these features might not have been as predictive as anticipated.
   - Lagged features (e.g., using day t-1's closing price to predict day t+1's opening price) did not significantly improve the performance of the **Logistic Regression** model.
   
3. **Sector-Specific Performance:**
   - The **Logistic Regression** model struggled with tech stocks, often reverting to the performance of the **Baseline model**. This suggests that these stocks might have unique characteristics or noise in the sentiment data that our current models couldn't adequately handle.

## Future Directions

1. **Sentiment Analysis Refinement:**
   - We need to understand how different types of articles impact sentiment and stock price predictions. Removing ‘dumb’ articles that only report stock growth and expanding our data sources to include non-financial news could improve model performance.
   - More sophisticated sentiment analysis methods should be considered, such as using **RoBERTa** or other advanced NLP techniques to capture more nuanced sentiments. Additionally, incorporating features like 'breaking news' events could add valuable context to our predictions.

2. **Sector and Stock-Specific Models:**
   - Further analysis into how specific sectors and stocks behave in relation to sentiment could lead to more tailored and effective models. For example, tech stocks like AAPL and NVIDIA may require specialized sentiment handling due to their high media coverage and the volume of noise present.
   - A breakdown by sector could highlight trends and characteristics that are not evident in a general model. This specialization could improve predictions for particular groups of stocks.

3. **Feature Engineering and Data Analysis:**
   - Continued fine-tuning of features and a deeper dive into the data will be critical. For instance, examining other forms of stock features and their lags, or experimenting with additional sentiment-derived features, could enhance model performance.
   - Investigating the temporal dynamics of sentiment and stock prices, possibly through more advanced time series analysis or deeper LSTM models, might uncover patterns not visible with current methodologies.

## Contributions

Contributions to enhance the models or explore new directions are welcome. Please fork the repository and submit a pull request with detailed explanations of the changes.

## Acknowledgements

We thank the Erdos Institute for putting on the May 2024 Data Science Boot Camp of which this project was a part, all those who took part in the judging process, and our project mentor Janco Krause.

---

Feel free to reach out with any questions or suggestions on how we can further improve our stock sentiment analysis approach.

---

For more detailed information, please refer to the individual model documentation and analysis files in the repository.

