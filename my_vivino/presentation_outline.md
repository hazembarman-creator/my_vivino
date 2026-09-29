# Presentation Outline

## Slide 1 - Why change the recommender
- The current system is rules-based
- New wines have a cold-start problem
- Leadership wants measurable improvement

## Slide 2 - Approved project
- Build a content-based recommendation MVP
- Use descriptions, price, geography, and quality signals
- Keep the solution explainable and production-friendly

## Slide 3 - Hypotheses
- Text and metadata can predict variety
- Value score can highlight affordable quality wines
- Similarity search can power cold-start recommendation

## Slide 4 - Data work
- Collect and clean the Wine Reviews dataset
- Handle missing prices and duplicate reviews
- Engineer value-oriented features

## Slide 5 - Exploration
- Scores are concentrated in the upper 80s and low 90s
- Prices are skewed
- A few countries and varieties dominate the catalog

## Slide 6 - ML results
- Compare model accuracy to a naive baseline
- Show macro F1 and top-3 accuracy
- Demonstrate that product text contains strong signal

## Slide 7 - Recommendation demo
- Pick one featured wine
- Return five similar wines
- Explain why this helps cold-start inventory

## Slide 8 - Business impact and production
- Show scenario-based revenue estimates
- Propose weekly retraining and API serving
- Recommend A/B testing against the old system
