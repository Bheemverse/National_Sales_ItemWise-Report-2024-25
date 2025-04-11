from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os
import io
import json
from typing import List, Optional, Dict, Any
import uvicorn
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
from pathlib import Path

app = FastAPI(title="Association Rules Mining API", 
              description="API for mining association rules from sales data")

# More robust file path handling
BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = str(BASE_DIR / "National_Sales_ItemWise Report 2024-25 (1).xlsx")

# Check if the file exists at startup and provide helpful error
if not Path(EXCEL_FILE).exists():
    print(f"WARNING: Excel file not found at {EXCEL_FILE}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir(BASE_DIR)}")
    # Look for any Excel files in the directory
    excel_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.xlsx')]
    if excel_files:
        print(f"Available Excel files: {excel_files}")
        # Use the first available Excel file
        EXCEL_FILE = str(BASE_DIR / excel_files[0])
        print(f"Using alternative Excel file: {EXCEL_FILE}")

@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Association Rules Mining</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
            .container { max-width: 800px; margin: 0 auto; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; }
            input, select { width: 100%; padding: 8px; box-sizing: border-box; }
            button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
            button:hover { background-color: #45a049; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Association Rules Mining</h1>
            <form id="miningForm" onsubmit="submitForm(event)">
                <div class="form-group">
                    <label>Minimum Support (0.0 - 1.0):</label>
                    <input type="number" name="min_support" step="0.01" min="0.01" max="1.0" value="0.01" required>
                </div>
                <div class="form-group">
                    <label>Minimum Confidence (0.0 - 1.0):</label>
                    <input type="number" name="min_confidence" step="0.01" min="0.01" max="1.0" value="0.2" required>
                </div>
                <div class="form-group">
                    <label>Max Number of Rules:</label>
                    <input type="number" name="max_rules" value="10" min="1" required>
                </div>
                <div class="form-group">
                    <label>Sheet Name:</label>
                    <input type="text" name="sheet_name" value="Sheet1" required>
                </div>
                <div class="form-group">
                    <label>Item Column Name:</label>
                    <input type="text" name="item_column" value="ITEMNAME" required>
                </div>
                <div class="form-group">
                    <label>Transaction Column Name:</label>
                    <input type="text" name="transaction_column" value="BILLNO" required>
                </div>
                <button type="submit">Generate Rules</button>
            </form>
            <div id="results" style="display:none;">
                <h2>Association Rules</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Antecedents</th>
                            <th>Consequents</th>
                            <th>Support</th>
                            <th>Confidence</th>
                            <th>Lift</th>
                        </tr>
                    </thead>
                    <tbody id="rulesBody"></tbody>
                </table>
            </div>
        </div>
        <script>
            function submitForm(event) {
                event.preventDefault();
                const form = document.getElementById('miningForm');
                const formData = new FormData(form);
                document.getElementById('results').style.display = 'block';
                document.getElementById('rulesBody').innerHTML = '';

                fetch('/mine-rules', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('rulesBody');
                    if (!Array.isArray(data)) {
                        tbody.innerHTML = '<tr><td colspan="5">Invalid response from server</td></tr>';
                        return;
                    }
                    if (data.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5">No rules found</td></tr>';
                        return;
                    }
                    data.forEach(rule => {
                        const row = `<tr>
                            <td>${rule.antecedents.join(', ')}</td>
                            <td>${rule.consequents.join(', ')}</td>
                            <td>${rule.support.toFixed(4)}</td>
                            <td>${rule.confidence.toFixed(4)}</td>
                            <td>${rule.lift.toFixed(4)}</td>
                        </tr>`;
                        tbody.innerHTML += row;
                    });
                })
                .catch(err => {
                    document.getElementById('rulesBody').innerHTML = '<tr><td colspan="5">Error fetching rules: ' + err.message + '</td></tr>';
                });
            }
        </script>
    </body>
    </html>
    """
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
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)

        if item_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Item column '{item_column}' not found")
        if transaction_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Transaction column '{transaction_column}' not found")

        if item_column.lower() == "itemname":
            def name_lookup(code): return str(code)
        elif "ITEMNAME" in df.columns:
            item_map = df[[item_column, "ITEMNAME"]].drop_duplicates().set_index(item_column)["ITEMNAME"].to_dict()
            def name_lookup(code): return item_map.get(code, str(code))
        else:
            raise HTTPException(status_code=400, detail="Missing 'ITEMNAME' column for name mapping.")

        df = df[[transaction_column, item_column]].dropna()
        df['value'] = 1
        basket = pd.crosstab(df[transaction_column], df[item_column])
        basket = (basket > 0).astype(int)

        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

            rules_list = []
            for _, rule in rules.head(max_rules).iterrows():
                antecedents = [name_lookup(code) for code in rule['antecedents']]
                consequents = [name_lookup(code) for code in rule['consequents']]

                rules_list.append({
                    "antecedents": antecedents,
                    "consequents": consequents,
                    "support": rule['support'],
                    "confidence": rule['confidence'],
                    "lift": rule['lift']
                })

            return rules_list
        else:
            return []

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing rules: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)