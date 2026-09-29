# My Vivino: From Rules-Based Recommendations to a Data Science MVP

## Introduction
My Vivino has a large wine catalog and millions of users, but its recommendation logic is still mostly rules-based.
That makes the system hard to improve, hard to measure, and weak on new inventory.

## Question
Can we build a recommendation MVP that is:
- explainable
- useful for cold-start recommendations
- valuable for product leadership
- realistic for DevOps to deploy

## Hypotheses
1. Wine descriptions and metadata contain enough signal to predict wine variety.
2. A value score can identify strong wines for merchandising.
3. Nearest-neighbor similarity can generate useful recommendations without user history.

## Data
This project uses the public Wine Reviews dataset with descriptions, prices, scores, countries, provinces, wineries, and varieties.

## Method
1. Collect data
2. Clean and validate it
3. Explore the main patterns
4. Train a supervised model
5. Build a recommendation prototype
6. Simulate business impact

## Business takeaway
This project shows that My Vivino can replace static recommendation rules with a measurable and explainable MVP.
