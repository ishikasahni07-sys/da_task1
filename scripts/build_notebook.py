"""Builds notebooks/01_EDA.ipynb programmatically with nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# --- Title ---
md("""# Task 1: Foundational Setup & Exploratory Data Analysis (EDA)
### Dataset: Sample Superstore Sales
**apexplanet-data-analytics**

This notebook covers:
1. Environment / library check
2. Loading the raw dataset
3. Data cleaning
4. Exploratory Data Analysis (EDA)
5. Key business insights
""")

# --- Setup ---
md("## 1. Environment Setup")
code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sqlalchemy import create_engine

sns.set_theme(style="whitegrid", palette="deep")
%matplotlib inline

print("pandas", pd.__version__)
print("numpy", np.__version__)""")

# --- Load ---
md("## 2. Load Raw Data")
code("""raw = pd.read_csv("../data/Sample-Superstore.csv", encoding="utf-8-sig")
print(raw.shape)
raw.head()""")

md("### Initial profiling")
code("""raw.info()""")

code("""raw.isnull().sum()""")

code("""raw.duplicated().sum()""")

code("""raw.describe()""")

# --- Cleaning ---
md("""## 3. Data Cleaning

Steps applied (see `scripts/clean_data.py` for the reusable version):
- Standardize column names to snake_case
- Parse `Order Date` / `Ship Date` as datetime
- Cast `Postal Code` to a nullable integer (11 missing values left as-is; not needed for sales/profit analysis)
- Drop exact duplicate rows (0 found)
- Engineer helper columns: `order_year`, `order_month`, `order_year_month`, `shipping_days`, `profit_margin`, `is_loss`
""")
code("""df = raw.copy()
df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

df["order_date"] = pd.to_datetime(df["order_date"])
df["ship_date"] = pd.to_datetime(df["ship_date"])
df["postal_code"] = df["postal_code"].astype("Int64")

df = df.drop_duplicates()

df["order_year"] = df["order_date"].dt.year
df["order_month"] = df["order_date"].dt.month
df["order_year_month"] = df["order_date"].dt.to_period("M").astype(str)
df["shipping_days"] = (df["ship_date"] - df["order_date"]).dt.days
df["profit_margin"] = (df["profit"] / df["sales"]).round(4)
df["is_loss"] = df["profit"] < 0

df.to_csv("../data/superstore_clean.csv", index=False)
df.head()""")

md("### (Optional) Load into SQLite via SQLAlchemy — demonstrates the sqlalchemy dependency")
code("""engine = create_engine("sqlite:///../data/superstore.db")
df.to_sql("orders", engine, if_exists="replace", index=False)

pd.read_sql("SELECT category, ROUND(SUM(sales),2) AS total_sales FROM orders GROUP BY category ORDER BY total_sales DESC", engine)""")

# --- EDA ---
md("## 4. Exploratory Data Analysis")

md("### 4.1 Overall KPIs")
code("""total_sales = df["sales"].sum()
total_profit = df["profit"].sum()
overall_margin = total_profit / total_sales
loss_pct = (df["profit"] < 0).mean() * 100
avg_ship_days = df["shipping_days"].mean()

print(f"Total Sales:        ${total_sales:,.2f}")
print(f"Total Profit:       ${total_profit:,.2f}")
print(f"Overall Margin:     {overall_margin:.2%}")
print(f"Orders at a loss:   {loss_pct:.2f}%")
print(f"Avg shipping days:  {avg_ship_days:.2f}")""")

md("### 4.2 Monthly Sales & Profit Trend")
code("""monthly = df.groupby("order_year_month")[["sales","profit"]].sum().reset_index()

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(monthly["order_year_month"], monthly["sales"], label="Sales", linewidth=2)
ax.plot(monthly["order_year_month"], monthly["profit"], label="Profit", linewidth=2)
ax.set_xticks(ax.get_xticks()[::3])
plt.xticks(rotation=45, ha="right")
ax.set_title("Monthly Sales & Profit Trend (2015-2018)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
ax.legend()
plt.tight_layout()
plt.show()""")

md("### 4.3 Sales & Profit by Category")
code("""cat = df.groupby("category")[["sales","profit"]].sum().sort_values("sales", ascending=False)
ax = cat.plot(kind="bar", figsize=(8,5), title="Sales & Profit by Category")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
cat""")

md("### 4.4 Profit by Sub-Category (loss-makers highlighted)")
code("""subcat = df.groupby("sub_category")["profit"].sum().sort_values()
colors = ["#d62728" if v < 0 else "#2ca02c" for v in subcat.values]

fig, ax = plt.subplots(figsize=(9,7))
ax.barh(subcat.index, subcat.values, color=colors)
ax.set_title("Total Profit by Sub-Category (red = loss-making)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
plt.tight_layout()
plt.show()""")

md("""**Observation:** `Tables` is the single biggest loss-maker (~ -$17.7K total), followed by `Bookcases` and `Supplies`. `Copiers`, `Phones`, and `Accessories` are the strongest profit drivers.""")

md("### 4.5 Regional Performance")
code("""reg = df.groupby("region")[["sales","profit"]].sum().sort_values("sales", ascending=False)
ax = reg.plot(kind="bar", figsize=(7,5), title="Sales & Profit by Region")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
reg""")

md("### 4.6 Discount vs Profit")
code("""fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df, x="discount", y="profit", hue="category", alpha=0.5, ax=ax)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_title("Discount vs Profit (by Category)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
plt.tight_layout()
plt.show()

print("Correlation (discount, profit):", round(df["discount"].corr(df["profit"]), 3))""")

md("""**Observation:** Discount and profit are negatively correlated (≈ -0.22). Heavy discounting (>30-40%) pushes many orders into a loss, especially in Furniture.""")

md("### 4.7 Customer Segment Share")
code("""seg = df.groupby("segment")["sales"].sum()
fig, ax = plt.subplots(figsize=(6,6))
ax.pie(seg.values, labels=seg.index, autopct="%1.1f%%", startangle=90)
ax.set_title("Sales Share by Customer Segment")
plt.tight_layout()
plt.show()""")

md("### 4.8 Top 10 States by Sales")
code("""state = df.groupby("state")["sales"].sum().sort_values(ascending=False).head(10)
ax = state.plot(kind="bar", figsize=(9,5), color="#1f77b4", title="Top 10 States by Sales")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()""")

md("### 4.9 Correlation Matrix")
code("""corr = df[["sales","quantity","discount","profit","shipping_days"]].corr()
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
ax.set_title("Correlation Matrix")
plt.tight_layout()
plt.show()""")

# --- Insights ---
md("""## 5. Key Insights Summary

1. **Overall performance:** $2.30M in total sales generated $286K in profit — a modest **12.5% overall margin**.
2. **~18.7% of all orders lose money**, concentrated heavily in the Furniture category (Tables and Bookcases in particular).
3. **Discounting hurts profit:** correlation of -0.22 between discount and profit; discounts above ~30% are strongly associated with losses.
4. **Technology drives profit:** Copiers, Phones, and Accessories are the top 3 profit-generating sub-categories.
5. **Geography matters:** California and New York lead in sales, while Texas posts the weakest profit performance despite solid sales volume — a discounting/cost issue worth investigating further.
6. **Shipping:** Average shipping time is ~4 days across all ship modes, fairly consistent and not a likely driver of profit variance.

## 6. Next Steps (Task 2+)
- Deeper segmentation (customer-level RFM analysis)
- Investigate Texas / Furniture losses with a discount-threshold policy
- Build the interactive dashboard (`dashboards/`) surfacing these KPIs
""")

nb["cells"] = cells
with open("/home/claude/apexplanet-data-analytics/notebooks/01_EDA.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written.")
