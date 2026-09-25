# Customer Repeat Purchase Prediction

Predicting whether a customer will make a repeat purchase within 90 days of their first order, using real UK e-commerce transaction data — with an interactive Power BI dashboard for business use.

## Purpose

Online retailers can't tell in advance which first-time customers will come back. This project builds a model that estimates that likelihood from a customer's very first order, so a business could focus retention effort (discounts, reminder emails) on the customers who actually need it — instead of guessing or spending equally on everyone.

## Dataset

**Online Retail II** (UCI / Kaggle) — a UK-based online gift retailer, covering transactions from **December 2009 to December 2011**.

- ~1.07 million transaction rows, combined across two sheets (`Year 2009-2010`, `Year 2010-2011`)
- Columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`
- 5,942 unique customers across 43 countries
- Each row represents one product line item within an order (an order/invoice can span multiple rows)

**Note on recency:** this dataset is from 2009–2011. It's used here to demonstrate methodology (cleaning, feature engineering, modeling, evaluation) — not as a source of current market insight.

## The Prediction

**Target:** Binary — will this customer place a second order within 90 days of their first order?

**Unit of analysis:** the customer, not the product or the store. This dataset covers a single online retailer, so there's no multi-store comparison — the geographic dimension available is customer country.

**Why 90 days:** a fixed window makes the prediction business-actionable ("will they come back soon enough to matter"), rather than an open-ended "will they ever buy again."

**Important scoping decision:** customers whose first order fell within 90 days of the dataset's end date (after Sept 10, 2011) were excluded. They wouldn't have had a fair, complete window to show repeat behavior, and including them would have biased the labels toward "no repeat" simply due to running out of observation time. This excluded 597 customers, leaving 5,281 (later 5,252 after further data-quality fixes) for modeling.

## Data Cleaning — Decisions & Reasoning

| Issue | Decision | Reasoning |
|---|---|---|
| Missing Customer ID (~23% of rows) | Dropped | Likely guest checkouts; can't be attributed to a customer, so unusable for customer-level analysis. This is a known, unavoidable limitation — the analysis scope is customers with an identifiable account only. |
| Cancelled orders (`Invoice` starts with "C") | Dropped | Represent cancellations, not completed purchases; not relevant to predicting repeat purchase in this scope. |
| Negative/zero quantity or price | Dropped | Data entry errors or return adjustments, not real purchases. |
| Duplicate rows | Dropped | Exact duplicates carry no additional signal. |
| Non-product administrative codes (`M`, `POST`, `D`, `DOT`, `BANK CHARGES`, `C2`, `PADS`) | Dropped | **Discovered during feature engineering**, not the initial pass — an outlier check on `avg_price_per_item` surfaced a customer whose "first order" was actually a single `Manual` adjustment row worth £10,953.50, not a real purchase. Investigating this led to identifying and removing all non-product StockCodes (2,801 rows total) that had been quietly distorting spend-based features. |
| Country skew (~91% UK) | Simplified to a binary `is_uk` feature | 43 raw country categories would create many near-empty groups, adding noise rather than signal and risking overfitting to rare countries. |
| Right-skewed `total_spend` / `total_quantity` (driven partly by a legitimate large wholesale buyer) | Added `log_total_spend` and `log_total_quantity` alongside the originals | Reduces the influence of extreme-but-real values for scale-sensitive models like Logistic Regression, without discarding genuine customers. |

## Features Used

Built entirely from each customer's **first order only** — this matches the real business scenario, where a retailer only has the first order available when deciding whether to target a customer.

- `log_total_spend`, `log_total_quantity` — spend and volume (log-transformed)
- `num_distinct_products` — variety of products bought (distinct from quantity: buying 100 units of 1 product vs. 100 units across 20 products)
- `avg_price_per_item`
- `is_uk` — UK vs. non-UK
- `order_month`, `order_dayofweek` — timing/seasonality

The repeat-purchase label itself is derived from each customer's *subsequent* order history (did a second order occur within 90 days) — this is never used as an input feature, only as the answer being predicted.

## Exploratory Analysis & Statistical Test

Boxplots and rate comparisons showed weak-to-moderate positive relationships between repeat purchase and: first-order spend, product variety, and being a non-UK customer — none of them dramatic on their own.

One pattern stood out: customers whose first order was in **December** had a notably higher repeat-purchase rate (56.4%) than the overall average, with **November** the lowest (30.2%). A chi-square test confirmed this was statistically significant (χ² = 69.27, p < 0.001) — not due to chance.

## Modeling

Three models were trained and compared: **Logistic Regression** (baseline), **Random Forest**, and **XGBoost**.

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.60 | 0.595 |
| Random Forest | 0.57 | 0.597 |
| XGBoost | 0.57 | 0.596 |

**Honest finding:** all three models — despite very different levels of complexity — converged on the same modest performance (~0.59-0.60 ROC-AUC, barely above random guessing at 0.50). When simple and complex models alike hit the same ceiling, that's a signal about the *features*, not the modeling approach: first-order transactional data alone (spend, quantity, variety, timing, country) carries only weak-to-moderate signal for predicting repeat purchase in this dataset. Additional data — e.g., marketing exposure, delivery/service experience, customer demographics — would likely be needed to meaningfully improve predictive power. This is reported as a genuine finding, not a shortfall to hide.

## Model Interpretation (SHAP)

Using SHAP on the XGBoost model to see which features the model relies on most, and in which direction:

- **`log_total_spend`** (most influential): higher first-order spend generally pushes toward a "will repurchase" prediction.
- **`avg_price_per_item`**: customers who bought more, cheaper items showed a slightly higher repurchase tendency than those who bought fewer, more expensive items.
- **`log_total_quantity`**: very high total quantity slightly pushed toward *not* repurchasing — likely reflecting one-off bulk/wholesale buyers rather than repeat retail customers.
- **`num_distinct_products`**: higher product variety in the first order pushed toward repurchase, consistent with the EDA finding.
- **`order_month`, `order_dayofweek`, `is_uk`**: comparatively minor individual influence on the model's predictions.

**Correlation vs. causation:** these are statistical associations found in historical data, not proof of causation. For example, higher first-order spend correlates with repeat purchase, but this may reflect an underlying customer type (naturally more engaged shoppers) rather than spend itself causing return behavior. The model is used here for prediction and targeting, not as a definitive explanation of *why* customers return — this dataset contains transactional/behavioral signals only, with no data on customer satisfaction, service quality, or demographics that would support deeper causal claims.

## Dashboard

An interactive Power BI dashboard presents:
- Overall business metrics (total customers, repeat-purchase rate, revenue)
- Customer-level predicted repeat-purchase probability and segmentation
- Key drivers of repeat purchase, translated into plain business language
- Breakdowns by country and time period

*(Dashboard screenshots/link to be added.)*

## Tools

- **Python** (pandas, scikit-learn, XGBoost, SHAP, matplotlib/seaborn) — cleaning, feature engineering, modeling, interpretation
- **Power BI** (Power Query + DAX) — interactive dashboard
- **Jupyter Notebook**, **GitHub** — development and version control

## Project Structure

```
Repeat-Purchase-Prediction/
├── data/
│   ├── Raw/              # Original dataset (not committed — see Data section)
│   └── Processed/        # Cleaned/engineered data (not committed)
├── notebooks/
│   ├── 01_data_inspection_cleaning.ipynb
│   └── 02_modeling.ipynb
├── PowerBI/               # Dashboard file
├── Outputs/
│   ├── Figures/           # Saved EDA plots
│   └── Models/
└── README.md
```

## Limitations

- ~23% of transactions (missing Customer ID) could not be used, limiting the analysis to identifiable, registered customers.
- Dataset is UK-heavy (~91% of customers); non-UK patterns are based on a much smaller sample and carry more uncertainty.
- Data is from 2009–2011 — used for methodology demonstration, not current market analysis.
- Predictive performance is modest (ROC-AUC ~0.60); see the Modeling section for why, and what additional data would likely help.
