import plotly.graph_objects as go
import pandas as pd
import json

# Load the data
data = {"pipeline_steps": [{"step": 1, "name": "User Query Input", "description": "Natural language query", "example": "Show me sales by region"}, {"step": 2, "name": "Query Preprocessing", "description": "Parse and analyze", "example": "Extract intent and entities"}, {"step": 3, "name": "Schema Retrieval", "description": "Find relevant tables", "example": "sales, regions tables"}, {"step": 4, "name": "AI Model Inference", "description": "Generate SQL", "example": "SELECT region, SUM(amount)..."}, {"step": 5, "name": "SQL Validation", "description": "Check syntax", "example": "Validate query structure"}, {"step": 6, "name": "Query Execution", "description": "Run on database", "example": "Execute SQL query"}, {"step": 7, "name": "Result Processing", "description": "Format results", "example": "Return formatted data"}]}

# Create DataFrame
df = pd.DataFrame(data['pipeline_steps'])

# Abbreviate names to fit 15 character limit
df['short_name'] = df['name'].apply(lambda x: x[:15] if len(x) > 15 else x)

# Create horizontal bar chart
fig = go.Figure()

# Add bars with different colors from the brand palette
colors = ['#1FB8CD', '#FFC185', '#ECEBD5', '#5D878F', '#D2BA4C', '#B4413C', '#964325']

fig.add_trace(go.Bar(
    x=[1] * len(df),  # All bars same length to show flow
    y=df['short_name'],
    orientation='h',
    marker_color=colors[:len(df)],
    text=df['step'],
    textposition='inside',
    textfont=dict(size=16, color='white'),
    hovertemplate='<b>%{y}</b><br>Step %{text}<extra></extra>',
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title='AI Text-to-SQL Pipeline Workflow',
    xaxis_title='Pipeline Flow',
    yaxis_title='Process Steps',
    showlegend=False,
    yaxis=dict(categoryorder='array', categoryarray=df['short_name'].tolist()),
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
)

# Save the chart
fig.write_image('ai_sql_pipeline.png')