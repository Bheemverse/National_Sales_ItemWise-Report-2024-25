from fastapi import FastAPI, Form, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import uvicorn

app = FastAPI(title="Association Rules Mining API", description="API for mining association rules from sales data")

# Path to your Excel file
EXCEL_FILE = "../National_Sales_ItemWise Report 2024-25 (1).xlsx"

@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """[REMAINS SAME AS YOUR HTML ABOVE]"""
    return html_content

@app.get("/sheet-names")
async def get_sheet_names():
    try:
        xl = pd.ExcelFile(EXCEL_FILE)
        return {"sheet_names": xl.sheet_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading Excel file: {str(e)}")

@app.get("/column-names")
async def get_column_names(sheet_name: str = Query(...)):
    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name, nrows=1)
        return {"column_names": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading columns: {str(e)}")

@app.post("/mine-rules")
async def mine_rules(
    min_support: float = Form(...),
    min_confidence: float = Form(...),
    max_rules: int = Form(...),
    sheet_name: str = Form(...),
    item_column: str = Form(...),
    transaction_column: str = Form(...)
):
    try:
        # ✅ Fixed line — actual Excel read
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)

        # Validation
        if item_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Item column '{item_column}' not found")
        if transaction_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Transaction column '{transaction_column}' not found")

        # Preprocess
        df = df[[transaction_column, item_column]].dropna()
        df['value'] = 1
        basket = pd.crosstab(df[transaction_column], df[item_column])

        # Frequent itemsets
        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            rules_list = []
            for _, rule in rules.head(max_rules).iterrows():
                rules_list.append({
                    "antecedents": list(rule['antecedents']),
                    "consequents": list(rule['consequents']),
                    "support": rule['support'],
                    "confidence": rule['confidence'],
                    "lift": rule['lift']
                })
            return rules_list
        else:
            return []

    except Exception as e:
        return [{"antecedents": ["Error"], "consequents": [str(e)], "support": 0, "confidence": 0, "lift": 0}]

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
