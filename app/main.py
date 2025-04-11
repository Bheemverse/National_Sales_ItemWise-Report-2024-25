from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os
import json
from typing import List, Optional, Dict, Any
import uvicorn

app = FastAPI(title="Association Rules Mining API", 
              description="API for mining association rules from sales data")

# Path to your Excel file
EXCEL_FILE = "../National_Sales_ItemWise Report 2024-25 (1).xlsx"

@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Association Rules Mining</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
                h1 { color: #333; }
                .container { max-width: 800px; margin: 0 auto; }
                .form-group { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; }
                input, select { width: 100%; padding: 8px; box-sizing: border-box; }
                button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
                button:hover { background-color: #45a049; }
                .results { margin-top: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Association Rules Mining</h1>
                <form id="miningForm" onsubmit="submitForm(event)">
                    <div class="form-group">
                        <label for="min_support">Minimum Support (0.0-1.0):</label>
                        <input type="number" id="min_support" name="min_support" step="0.01" min="0.01" max="1.0" value="0.01" required>
                    </div>
                    <div class="form-group">
                        <label for="min_confidence">Minimum Confidence (0.0-1.0):</label>
                        <input type="number" id="min_confidence" name="min_confidence" step="0.01" min="0.01" max="1.0" value="0.2" required>
                    </div>
                    <div class="form-group">
                        <label for="max_rules">Maximum Number of Rules:</label>
                        <input type="number" id="max_rules" name="max_rules" min="1" value="10" required>
                    </div>
                    <div class="form-group">
                        <label for="sheet_name">Sheet Name:</label>
                        <input type="text" id="sheet_name" name="sheet_name" placeholder="Sheet1" required>
                    </div>
                    <div class="form-group">
                        <label for="item_column">Item Column Name:</label>
                        <input type="text" id="item_column" name="item_column" placeholder="Item Name" required>
                    </div>
                    <div class="form-group">
                        <label for="transaction_column">Transaction Column Name:</label>
                        <input type="text" id="transaction_column" name="transaction_column" placeholder="Invoice Number" required>
                    </div>
                    <button type="submit">Generate Rules</button>
                </form>
                
                <div id="results" class="results" style="display:none;">
                    <h2>Association Rules</h2>
                    <div id="loading" style="display:none;">Loading...</div>
                    <table id="rulesTable">
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
                    
                    document.getElementById('loading').style.display = 'block';
                    document.getElementById('results').style.display = 'block';
                    document.getElementById('rulesBody').innerHTML = '';
                    
                    fetch('/mine-rules', {
                        method: 'POST',
                        body: formData,
                    })
                    .then(response => response.json())
                    .then(data => {
                        displayRules(data);
                        document.getElementById('loading').style.display = 'none';
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        document.getElementById('loading').style.display = 'none';
                        document.getElementById('rulesBody').innerHTML = '<tr><td colspan="5">Error fetching rules: ' + error.message + '</td></tr>';
                    });
                }
                
                function displayRules(rules) {
                    const tbody = document.getElementById('rulesBody');
                    if (rules.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5">No rules found</td></tr>';
                        return;
                    }
                    
                    tbody.innerHTML = rules.map(rule => `
                        <tr>
                            <td>${Array.isArray(rule.antecedents) ? rule.antecedents.join(', ') : rule.antecedents}</td>
                            <td>${Array.isArray(rule.consequents) ? rule.consequents.join(', ') : rule.consequents}</td>
                            <td>${rule.support.toFixed(4)}</td>
                            <td>${rule.confidence.toFixed(4)}</td>
                            <td>${rule.lift.toFixed(4)}</td>
                        </tr>
                    `).join('');
                }
                
                // Get available sheet names when the page loads
                window.addEventListener('load', () => {
                    fetch('/sheet-names')
                    .then(response => response.json())
                    .then(data => {
                        const sheetInput = document.getElementById('sheet_name');
                        if (data.sheet_names && data.sheet_names.length > 0) {
                            sheetInput.value = data.sheet_names[0];
                            
                            // Also try to get column names for the first sheet
                            fetch(`/column-names?sheet_name=${data.sheet_names[0]}`)
                            .then(response => response.json())
                            .then(colData => {
                                if (colData.column_names && colData.column_names.length > 0) {
                                    // Try to guess item and transaction columns
                                    const itemCol = colData.column_names.find(col => 
                                        col.toLowerCase().includes('item') || 
                                        col.toLowerCase().includes('product') || 
                                        col.toLowerCase().includes('name'));
                                        
                                    const transCol = colData.column_names.find(col => 
                                        col.toLowerCase().includes('invoice') || 
                                        col.toLowerCase().includes('transaction') || 
                                        col.toLowerCase().includes('order') ||
                                        col.toLowerCase().includes('bill'));
                                        
                                    if (itemCol) document.getElementById('item_column').value = itemCol;
                                    if (transCol) document.getElementById('transaction_column').value = transCol;
                                }
                            });
                        }
                    });
                });
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
async def get_column_names(sheet_name: str = Query(..., description="Excel sheet name")):
    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name, nrows=1)
        return {"column_names": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading columns: {str(e)}")

@app.post("/mine-rules")
async def mine_rules(
    min_support: float = Form(..., description="Minimum support threshold"),
    min_confidence: float = Form(..., description="Minimum confidence threshold"),
    max_rules: int = Form(..., description="Maximum number of rules to return"),
    sheet_name: str = Form(..., description="Excel sheet name"),
    item_column: str = Form(..., description="Column containing item names"),
    transaction_column: str = Form(..., description="Column containing transaction IDs")
):
    try:
        # Read the Excel file
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
        
        # Ensure required columns exist
        if item_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Item column '{item_column}' not found")
        if transaction_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Transaction column '{transaction_column}' not found")
        
        # Create a binary matrix: 1 if item exists in a transaction
df = df[[transaction_column, item_column]].dropna()

# Crosstab-based encoding (alternative to TransactionEncoder)
df['value'] = 1
basket = pd.crosstab(df[transaction_column], df[item_column])

# Apply Apriori algorithm
frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

# Generate rules if any frequent itemsets exist
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

        
        # Encode the transactions
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Apply Apriori algorithm
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        
        # Generate association rules
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            # Convert the rules to a list of dictionaries for JSON response
            rules_list = []
            for _, rule in rules.head(max_rules).iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
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
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
