# My Vivino Slides

## Slide 1 - Business Problem
- My Vivino currently uses a rules-based recommendation system
- This approach is hard to improve and hard to measure
- New wines suffer from the cold-start problem

## Slide 2 - Approved Project
- Build a content-based wine recommendation MVP
- Use wine descriptions, ratings, price, country, and province
- Deliver something explainable and production-friendly

## Slide 3 - Hypotheses
- Wine text and metadata can predict wine variety
- A value score can identify good affordable wines
- Similarity search can recommend wines without user history

## Slide 4 - Data Collection and Cleaning
- Source: public Wine Reviews dataset
- Removed duplicates and handled missing values
- Engineered features like description length, log price, and value score

## Slide 5 - Exploration
- Scores are concentrated in the high 80s and low 90s
- Price distribution is skewed
- A few varieties and countries dominate the catalog

## Slide 6 - Machine Learning
- Trained a classifier to predict wine variety
- Inputs: description, country, province, price, points
- Compared performance against a naive baseline

## Slide 7 - Recommendation MVP
- Built a nearest-neighbor content-based recommender
- Returns similar wines using text and metadata similarity
- Useful for cold-start inventory and product discovery

## Slide 8 - Business Impact and Production
- Can improve discovery and merchandising
- Can surface high-value wines for promotion
- DevOps can deploy this as a lightweight retraining pipeline plus API

## Slide 9 - Next Steps
- Add user behavior data for personalization
- Run A/B tests against the current recommender
- Add seasonality, food pairing, and inventory constraints
