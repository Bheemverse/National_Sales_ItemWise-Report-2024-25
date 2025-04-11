from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import uvicorn

app = FastAPI()

EXCEL_FILE = "National_Sales_ItemWise Report 2024-25 (1).xlsx"
SHEET_NAME = "Sheet1"

# 🧠 Load dataset and prepare transactions
def prepare_basket():
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    df = df[["BILLNO", "ITEMNAME"]].dropna()
    df["value"] = 1
    basket = pd.crosstab(df["BILLNO"], df["ITEMNAME"])
    basket = (basket > 0).astype(int)
    return df, basket

df_loaded, basket_matrix = prepare_basket()
frequent_itemsets = apriori(basket_matrix, min_support=0.01, use_colnames=True)
rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
rules_df["antecedents"] = rules_df["antecedents"].apply(lambda x: list(x))
rules_df["consequents"] = rules_df["consequents"].apply(lambda x: list(x))


@app.get("/", response_class=HTMLResponse)
async def index():
    itemnames = sorted(df_loaded["ITEMNAME"].unique())
    options_html = "".join([f"<option value='{name}'>{name}</option>" for name in itemnames])

    return f"""
    <html>
    <head>
        <title>ITEMNAME Association</title>
        <style>
            body {{ font-family: Arial; padding: 20px; }}
            select, button {{ padding: 8px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; }}
            th {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>ITEMNAME-Based Association Rule Viewer</h1>
        <label>Select a Product Name (ITEMNAME):</label><br/>
        <select id="itemSelector">
            {options_html}
        </select>
        <button onclick="fetchAssociations()">Get Associations</button>
        <table id="results" style="display:none;">
            <thead>
                <tr><th>Product</th><th>Associated Product</th><th>Support</th><th>Confidence</th><th>Lift</th></tr>
            </thead>
            <tbody id="resultBody"></tbody>
        </table>

        <script>
            function fetchAssociations() {{
                const item = document.getElementById("itemSelector").value;
                fetch("/get-associated-products?itemname=" + encodeURIComponent(item))
                    .then(res => res.json())
                    .then(data => {{
                        const body = document.getElementById("resultBody");
                        const table = document.getElementById("results");
                        body.innerHTML = '';
                        table.style.display = 'table';
                        if(data.length === 0) {{
                            body.innerHTML = '<tr><td colspan="5">No rules found</td></tr>';
                            return;
                        }}
                        data.forEach(rule => {{
                            body.innerHTML += `<tr>
                                <td>${{rule.Product}}</td>
                                <td>${{rule.Associated_Product}}</td>
                                <td>${{rule.Support.toFixed(4)}}</td>
                                <td>${{rule.Confidence.toFixed(4)}}</td>
                                <td>${{rule.Lift.toFixed(4)}}</td>
                            </tr>`;
                        }});
                    }});
            }}
        </script>
    </body>
    </html>
    """

# 🧠 Backend API to get associated ITEMNAMEs
@app.get("/get-associated-products")
async def get_associated_products(itemname: str):
    itemname = itemname.strip().lower()
    filtered_rules = rules_df[
        rules_df["antecedents"].apply(lambda x: itemname in [i.lower() for i in x])
    ]

    if filtered_rules.empty:
        return []

    result = []
    for _, row in filtered_rules.head(10).iterrows():
        for assoc in row["consequents"]:
            result.append({
                "Product": itemname,
                "Associated_Product": assoc,
                "Support": row["support"],
                "Confidence": row["confidence"],
                "Lift": row["lift"]
            })

    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
