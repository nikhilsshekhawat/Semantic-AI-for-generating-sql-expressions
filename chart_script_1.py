import plotly.graph_objects as go
import json

# Data from the JSON
data = {"models": [{"name": "Arctic-Text2SQL-R1", "accuracy": 91.2}, {"name": "DataGPT-SQL-7B", "accuracy": 87.2}, {"name": "GPT-4", "accuracy": 85.3}, {"name": "SQLformer", "accuracy": 84.1}, {"name": "AI2SQL Falcon-7B", "accuracy": 82.1}, {"name": "RESDSQL-3B", "accuracy": 79.9}, {"name": "DIN-SQL", "accuracy": 78.5}, {"name": "T5-Large", "accuracy": 78.2}, {"name": "Vanna AI", "accuracy": 75.6}, {"name": "CodeT5", "accuracy": 73.4}]}

# Extract model names and accuracy scores
models = [item["name"] for item in data["models"]]
accuracies = [item["accuracy"] for item in data["models"]]

# Brand colors
colors = ['#1FB8CD', '#FFC185', '#ECEBD5', '#5D878F', '#D2BA4C', '#B4413C', '#964325', '#944454', '#13343B', '#DB4545']

# Create horizontal bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=accuracies,
    y=models,
    orientation='h',
    marker_color=colors,
    text=[f"{acc}%" for acc in accuracies],
    textposition='outside',
    cliponaxis=False
))

fig.update_layout(
    title="AI Model Text-to-SQL Accuracy"
)

fig.update_xaxes(title="Accuracy (%)")
fig.update_yaxes(title="AI Models")

fig.write_image("ai_models_accuracy.png")