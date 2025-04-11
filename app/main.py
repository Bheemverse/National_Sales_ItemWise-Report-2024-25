from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os
import io
import json
import traceback
import logging
from typing import List, Optional, Dict, Any
import uvicorn
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Association Rules Mining API", 
              description="API for mining association rules from sales data")

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            #loading { display: none; padding: 10px; background-color: #f8f9fa; margin-top: 10px; }
            .error { color: red; padding: 10px; border: 1px solid #ffcccc; background-color: #fff0f0; }
            .filter-container { margin-top: 20px; margin-bottom: 20px; display: flex; flex-direction: column; gap: 10px; }
            .search-container { display: flex; gap: 10px; }
            .search-box { flex-grow: 1; }
            .filtered-count { font-size: 0.9em; color: #666; margin-top: 5px; }
            .highlight { background-color: #e6ffe6; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Association Rules Mining</h1>
            <form id="miningForm" onsubmit="submitForm(event)">
                <div class="form-group">
                    <label>Minimum Support (0.0 - 1.0):</label>
                    <input type="number" name="min_support" id="min_support" step="0.001" min="0.001" max="1.0" value="0.005" required>
                </div>
                <div class="form-group">
                    <label>Minimum Confidence (0.0 - 1.0):</label>
                    <input type="number" name="min_confidence" id="min_confidence" step="0.01" min="0.01" max="1.0" value="0.1" required>
                </div>
                <div class="form-group">
                    <label>Max Number of Rules:</label>
                    <input type="number" name="max_rules" id="max_rules" value="20" min="1" required>
                </div>
                <div class="form-group">
                    <label>Sheet Name:</label>
                    <input type="text" name="sheet_name" id="sheet_name" value="Sheet1" required>
                </div>
                <div class="form-group">
                    <label>Item Column Name:</label>
                    <input type="text" name="item_column" id="item_column" value="ITEMNAME" required>
                </div>
                <div class="form-group">
                    <label>Transaction Column Name:</label>
                    <input type="text" name="transaction_column" id="transaction_column" value="BILLNO" required>
                </div>
                <button type="submit" id="submitBtn">Generate Rules</button>
            </form>
            <div id="loading">Generating rules... This may take a moment.</div>
            <div id="errorMsg" class="error" style="display:none;"></div>
            <div id="results" style="display:none;">
                <h2>Association Rules</h2>
                <div class="filter-container">
                    <h3>Filter by Product</h3>
                    <div class="search-container">
                        <input type="text" id="productSearch" class="search-box" placeholder="Search for a product..." oninput="filterProductDropdown()">
                        <select id="productFilter" onchange="filterRulesByProduct()">
                            <option value="">Show All Products</option>
                        </select>
                    </div>
                    <div class="filtered-count" id="filteredCount"></div>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Product (Antecedent)</th>
                            <th>Recommended Products (Consequents)</th>
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
            // Store all rules for filtering
            let allRules = [];
            let allProducts = [];
            
            function submitForm(event) {
                event.preventDefault();
                const form = document.getElementById('miningForm');
                const formData = new FormData(form);
                
                // Show loading, hide results and errors
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.getElementById('errorMsg').style.display = 'none';
                document.getElementById('submitBtn').disabled = true;

                fetch('/mine-rules', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(err => {
                            throw new Error(err.detail || 'Server error');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    // Hide loading, enable button
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('submitBtn').disabled = false;
                    
                    if (!Array.isArray(data)) {
                        showError('Invalid response from server');
                        return;
                    }
                    
                    // Store all rules
                    allRules = data;
                    
                    // Populate product filter
                    populateProductFilter(data);
                    
                    // Display rules
                    displayRules(data);
                    document.getElementById('results').style.display = 'block';
                    
                    // Clear search box
                    document.getElementById('productSearch').value = '';
                })
                .catch(err => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('submitBtn').disabled = false;
                    showError('Error fetching rules: ' + err.message);
                });
            }
            
            function showError(message) {
                const errorDiv = document.getElementById('errorMsg');
                errorDiv.textContent = message;
                errorDiv.style.display = 'block';
            }
            
            function populateProductFilter(rules) {
                const productSet = new Set();
                
                // Extract all products that appear as antecedents
                rules.forEach(rule => {
                    if (rule.antecedents && rule.antecedents.length === 1) {
                        productSet.add(rule.antecedents[0]);
                    }
                });
                
                // Sort products alphabetically
                allProducts = Array.from(productSet).sort();
                
                // Populate dropdown
                updateProductDropdown(allProducts);
            }
            
            function updateProductDropdown(products) {
                const productSelect = document.getElementById('productFilter');
                productSelect.innerHTML = '<option value="">Show All Products</option>';
                
                products.forEach(product => {
                    const option = document.createElement('option');
                    option.value = product;
                    option.textContent = product;
                    productSelect.appendChild(option);
                });
                
                // Update filtered count
                document.getElementById('filteredCount').textContent = 
                    products.length === allProducts.length ? 
                    `Showing all ${products.length} products` : 
                    `Filtered: ${products.length} of ${allProducts.length} products`;
            }
            
            function filterProductDropdown() {
                const searchTerm = document.getElementById('productSearch').value.toLowerCase();
                if (!searchTerm) {
                    updateProductDropdown(allProducts);
                    return;
                }
                
                // Filter products based on search term
                const filteredProducts = allProducts.filter(product => 
                    product.toLowerCase().includes(searchTerm)
                );
                
                updateProductDropdown(filteredProducts);
                
                // If only one product matches exactly, select it
                if (filteredProducts.length === 1) {
                    document.getElementById('productFilter').value = filteredProducts[0];
                    filterRulesByProduct();
                }
            }
            
            function filterRulesByProduct() {
                const selectedProduct = document.getElementById('productFilter').value;
                
                if (!selectedProduct) {
                    displayRules(allRules);
                    return;
                }
                
                // Filter rules where the selected product is an antecedent
                const filteredRules = allRules.filter(rule => 
                    rule.antecedents.length === 1 && 
                    rule.antecedents[0] === selectedProduct
                );
                
                displayRules(filteredRules);
                
                // Update the filtered count
                document.getElementById('filteredCount').textContent = 
                    `Showing ${filteredRules.length} rules for "${selectedProduct}"`;
            }
            
            function displayRules(rules) {
                const tbody = document.getElementById('rulesBody');
                tbody.innerHTML = '';
                
                if (rules.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5">No rules found</td></tr>';
                    return;
                }
                
                rules.forEach(rule => {
                    const row = document.createElement('tr');
                    
                    // Antecedents cell
                    const antCell = document.createElement('td');
                    antCell.textContent = rule.antecedents.join(', ');
                    row.appendChild(antCell);
                    
                    // Consequents cell
                    const conCell = document.createElement('td');
                    conCell.textContent = rule.consequents.join(', ');
                    row.appendChild(conCell);
                    
                    // Support cell
                    const supCell = document.createElement('td');
                    supCell.textContent = rule.support.toFixed(4);
                    row.appendChild(supCell);
                    
                    // Confidence cell
                    const confCell = document.createElement('td');
                    confCell.textContent = rule.confidence.toFixed(4);
                    row.appendChild(confCell);
                    
                    // Lift cell
                    const liftCell = document.createElement('td');
                    liftCell.textContent = rule.lift.toFixed(4);
                    row.appendChild(liftCell);
                    
                    tbody.appendChild(row);
                });
            }
            
            // Load sheet names on page load
            window.addEventListener('load', function() {
                fetch('/sheet-names')
                .then(response => response.json())
                .then(data => {
                    if (data.sheet_names && data.sheet_names.length > 0) {
                        document.getElementById('sheet_name').value = data.sheet_names[0];
                        
                        // Get column names for the first sheet
                        fetch(`/column-names?sheet_name=${data.sheet_names[0]}`)
                        .then(response => response.json())
                        .then(colData => {
                            if (colData.column_names && colData.column_names.length > 0) {
                                // Try to guess appropriate column names
                                const itemCol = colData.column_names.find(col => 
                                    col.toLowerCase().includes('item') || 
                                    col.toLowerCase().includes('product') || 
                                    col.toLowerCase().includes('name'));
                                    
                                const transCol = colData.column_names.find(col => 
                                    col.toLowerCase().includes('invoice') || 
                                    col.toLowerCase().includes('transaction') || 
                                    col.toLowerCase().includes('bill') ||
                                    col.toLowerCase().includes('order'));
                                    
                                if (itemCol) document.getElementById('item_column').value = itemCol;
                                if (transCol) document.getElementById('transaction_column').value = transCol;
                            }
                        })
                        .catch(err => {
                            console.error('Error fetching column names:', err);
                        });
                    }
                })
                .catch(err => {
                    console.error('Error fetching sheet names:', err);
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
        logger.error(f"Error reading Excel file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading Excel file: {str(e)}")

@app.get("/column-names")
async def get_column_names(sheet_name: str = Query(...)):
    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name, nrows=1)
        return {"column_names": df.columns.tolist()}
    except Exception as e:
        logger.error(f"Error reading columns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading columns: {str(e)}")

@app.post("/mine-rules")
async def mine_rules(
    min_support: float = Form(0.005, description="Minimum support threshold"),
    min_confidence: float = Form(0.1, description="Minimum confidence threshold"),
    max_rules: int = Form(20, description="Maximum number of rules to return"),
    sheet_name: str = Form(..., description="Excel sheet name"),
    item_column: str = Form(..., description="Column containing item names"),
    transaction_column: str = Form(..., description="Column containing transaction IDs")
):
    try:
        logger.info(f"Processing rules with params: support={min_support}, confidence={min_confidence}, sheet={sheet_name}")
        
        # Read the Excel file
        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
        
        # Log data shape
        logger.info(f"Data shape: {df.shape}")
        
        # Ensure required columns exist
        if item_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Item column '{item_column}' not found")
        if transaction_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Transaction column '{transaction_column}' not found")
        
        # Respect user's column choice but log the choice
        original_item_column = item_column
        logger.info(f"Using item column: {item_column}")
            
        # Print unique values in the item column (first 10) for verification
        unique_items = df[item_column].unique()
        logger.info(f"Sample of unique items from {item_column}: {', '.join(str(x) for x in unique_items[:10])}")
        
        # Filter only needed columns for efficiency
        df = df[[transaction_column, item_column]].dropna()
        
        # Create the basket format - use optimized method for larger datasets
        if df.shape[0] > 10000:
            # For larger datasets: group first, then one-hot encode
            logger.info("Using optimized method for large dataset")
            basket = df.groupby([transaction_column, item_column]).size().reset_index(name='count')
            basket_wide = basket.pivot_table(index=transaction_column, columns=item_column, values='count', fill_value=0)
            basket_encoded = (basket_wide > 0).astype(int)
        else:
            # For smaller datasets: use cross-tab approach
            logger.info("Using standard method for dataset")
            basket_encoded = pd.crosstab(df[transaction_column], df[item_column])
            basket_encoded = (basket_encoded > 0).astype(int)
        
        logger.info(f"Basket shape: {basket_encoded.shape}")
        
        # Apply Apriori algorithm
        frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
        logger.info(f"Found {len(frequent_itemsets)} frequent itemsets")
        
        # Generate association rules
        rules_list = []
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            logger.info(f"Generated {len(rules)} association rules")
            
            # Process rules and ensure at least 5 consequents per product
            product_to_rules = {}
            
            # First pass: organize rules by product
            for _, rule in rules.iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                
                # Log a sample of the generated rules for debugging
                if _ < 5:  # Log first 5 rules
                    logger.info(f"Rule {_}: {antecedents} -> {consequents}")
                
                # Ensure we're only working with single-item antecedents for clear recommendations
                if len(antecedents) == 1:
                    product = antecedents[0]
                    
                    if product not in product_to_rules:
                        product_to_rules[product] = []
                        
                    product_to_rules[product].append({
                        "antecedents": antecedents,
                        "consequents": consequents,
                        "support": float(rule['support']),
                        "confidence": float(rule['confidence']),
                        "lift": float(rule['lift'])
                    })
            
            logger.info(f"Rules organized for {len(product_to_rules)} unique products")
            
            # Second pass: ensure minimum of 5 consequents per product
            all_products = df[item_column].unique()
            
            for product in all_products:
                if product not in product_to_rules or sum(len(rule["consequents"]) for rule in product_to_rules[product]) < 5:
                    # We need more consequents for this product
                    logger.info(f"Finding more consequents for product: {product}")
                    
                    # Get transactions containing this product
                    transactions_with_product = set(df[df[item_column] == product][transaction_column])
                    
                    if transactions_with_product:
                        # Find items that co-occur with this product
                        co_occurring = df[
                            df[transaction_column].isin(transactions_with_product) & 
                            (df[item_column] != product)
                        ]
                        
                        if not co_occurring.empty:
                            # Count co-occurrences
                            co_occurring_counts = co_occurring[item_column].value_counts()
                            
                            # Calculate metrics
                            total_txns = df[transaction_column].nunique()
                            product_txns = len(transactions_with_product)
                            
                            # Create new rules for top co-occurring items
                            new_rules = []
                            for co_item, count in co_occurring_counts.items():
                                # Skip if already in our rules
                                if product in product_to_rules and any(
                                    co_item in rule["consequents"] for rule in product_to_rules[product]
                                ):
                                    continue
                                
                                # Calculate metrics
                                support = count / total_txns
                                confidence = count / product_txns
                                
                                # Calculate lift
                                co_item_txns = df[df[item_column] == co_item][transaction_column].nunique()
                                expected = (product_txns / total_txns) * (co_item_txns / total_txns)
                                lift = (support / expected) if expected > 0 else 1.0
                                
                                new_rules.append({
                                    "antecedents": [product],
                                    "consequents": [co_item],
                                    "support": float(support),
                                    "confidence": float(confidence),
                                    "lift": float(lift)
                                })
                            
                            # Sort by lift and take top rules
                            new_rules.sort(key=lambda x: x["lift"], reverse=True)
                            
                            # Add to product rules
                            if product not in product_to_rules:
                                product_to_rules[product] = []
                            
                            # Calculate how many more rules we need
                            current_consequents = sum(len(rule["consequents"]) for rule in product_to_rules[product])
                            needed = max(0, 5 - current_consequents)
                            
                            # Add as many new rules as needed
                            product_to_rules[product].extend(new_rules[:needed])
            
            # Combine all rules
            for product_rules in product_to_rules.values():
                rules_list.extend(product_rules)
                
            # Log a sample of the final rules for debugging
            for i, rule in enumerate(rules_list[:5]):
                logger.info(f"Final rule {i}: {rule['antecedents']} -> {rule['consequents']}")
            
            # Sort by lift and limit to max_rules
            rules_list.sort(key=lambda x: x["lift"], reverse=True)
            if max_rules > 0 and len(rules_list) > max_rules:
                rules_list = rules_list[:max_rules]
                
            logger.info(f"Returning {len(rules_list)} rules")
        
        return rules_list
    
    except Exception as e:
        error_msg = f"Error processing rules: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)